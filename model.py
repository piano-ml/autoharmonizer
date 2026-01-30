import os
import math
import pickle
import zipfile
import numpy as np
from config import *
from tqdm import trange
from keras import Model
from keras.utils import Sequence, to_categorical
from keras.layers import Input, Embedding, Concatenate
from keras.layers import TimeDistributed
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.callbacks import ModelCheckpoint
from keras.metrics import F1Score

class DataGenerator(Sequence):
    """
    Générateur optimisé : Envoie des entiers (indices) au lieu de one-hot vectors.
    Cela réduit drastiquement l'utilisation de la RAM et de la bande passante CPU->GPU.
    """
    def __init__(self, 
                 input_melody_left, input_melody_right, 
                 input_beat_left, input_beat_right, 
                 input_key_left, input_key_right,
                 input_chord_left,
                 output_chord,
                 chord_nums,
                 batch_size=BATCH_SIZE, 
                 shuffle=True,
                 **kwargs):
        super().__init__(**kwargs)
        # Stockage brut en entiers (uint8/uint16) pour économiser la RAM
        self.input_melody_left = np.asarray(input_melody_left, dtype=np.uint8)
        self.input_melody_right = np.asarray(input_melody_right, dtype=np.uint8)
        self.input_beat_left = np.asarray(input_beat_left, dtype=np.uint8)
        self.input_beat_right = np.asarray(input_beat_right, dtype=np.uint8)
        self.input_key_left = np.asarray(input_key_left, dtype=np.uint8)
        self.input_key_right = np.asarray(input_key_right, dtype=np.uint8)
        self.input_chord_left = np.asarray(input_chord_left, dtype=np.uint16)
        
        # Le target reste en one-hot ou sparse selon le choix de loss. 
        # Ici on le garde en uint16 et on le convertit à la volée pour la categorical_crossentropy
        self.output_chord = np.asarray(output_chord, dtype=np.uint16)
        
        self.chord_nums = chord_nums
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = len(self.input_melody_left)
        self.indices = np.arange(self.n_samples)
        self.on_epoch_end()
    
    def __len__(self):
        return math.ceil(self.n_samples / self.batch_size)
    
    def __getitem__(self, index):
        indices = self.indices[index*self.batch_size : (index+1)*self.batch_size]
        
        # On renvoie les indices directement. Les couches Embedding du modèle feront le travail vectoriel.
        X = {
            "input_melody_left": self.input_melody_left[indices],
            "input_melody_right": self.input_melody_right[indices],
            "input_beat_left": self.input_beat_left[indices],
            "input_beat_right": self.input_beat_right[indices],
            "input_key_left": self.input_key_left[indices],
            "input_key_right": self.input_key_right[indices],
            "input_chord_left": self.input_chord_left[indices]
        }
        
        # Seul l'output est converti en one-hot pour correspondre à 'categorical_crossentropy'
        y = to_categorical(self.output_chord[indices], num_classes=self.chord_nums)

        return X, y
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


def create_training_data(segment_length=SEGMENT_LENGTH, chord_types_path=CHORD_TYPES_PATH, corpus_path=CORPUS_PATH,  val_ratio=VAL_RATIO):
    # Load corpus
    with open(corpus_path, "rb") as filepath:
        data_corpus = pickle.load(filepath)
    # Load chord types
    with open(chord_types_path, "rb") as filepath:
        chord_types = pickle.load(filepath)

    chord_types_dict = {chord_types[i]: i for i in range(len(chord_types))}

    # 1. Estimation de la taille pour pré-allocation
    total_train_size = 0
    total_val_size = 0
    np.random.seed(0)
    
    # Simple passe pour compter
    for songs in data_corpus:
        is_train = np.random.rand() > val_ratio
        for song in songs:
            size_incr = len(song[0]) # Nombre de segments potentiels (simplifié)
            # En réalité c'est un peu moins à cause des bords, mais mieux vaut allouer trop que pas assez
            if is_train: total_train_size += size_incr
            else: total_val_size += size_incr
    
    # Allocation mémoire optimisée (uint8/16)
    def create_arrays(size):
        return {
            "mel_l": np.zeros((size, segment_length), dtype=np.uint8),
            "mel_r": np.zeros((size, segment_length), dtype=np.uint8),
            "beat_l": np.zeros((size, segment_length), dtype=np.uint8),
            "beat_r": np.zeros((size, segment_length), dtype=np.uint8),
            "key_l": np.zeros((size, segment_length), dtype=np.uint8),
            "key_r": np.zeros((size, segment_length), dtype=np.uint8),
            "chord_l": np.zeros((size, segment_length), dtype=np.uint16),
            "out": np.zeros(size, dtype=np.uint16)
        }

    train_data = create_arrays(total_train_size)
    val_data = create_arrays(total_val_size)

    # Pointeurs
    t_idx, v_idx = 0, 0
    np.random.seed(0) # Reset seed pour reproduire la répartition

    print(f"Processing corpus into sequences (Segment: {segment_length})...")
    
    for songs_idx in trange(len(data_corpus)):
        songs = data_corpus[songs_idx]
        target_dict = train_data if (np.random.rand() > val_ratio) else val_data
        
        for song in songs:
            # Padding
            pad = [0] * segment_length
            s_mel = np.array(pad + song[0] + pad, dtype=np.uint8)
            s_beat = np.array(pad + song[1] + pad, dtype=np.uint8)
            s_key = np.array(pad + song[2] + pad, dtype=np.uint8)
            s_chord = np.array(pad + [chord_types_dict[c] for c in song[3]] + pad, dtype=np.uint16)
            
            # Vectorized slicing serait plus rapide, mais la boucle est lisible
            limit = len(s_mel) - segment_length
            
            # Optimisation: on prépare les batchs de cette chanson
            # C'est une fenêtre glissante
            for idx in range(segment_length, limit):
                # Selection dictionnary ref (train ou val)
                c_idx = t_idx if (target_dict is train_data) else v_idx
                
                target_dict["mel_l"][c_idx] = s_mel[idx-segment_length:idx]
                target_dict["mel_r"][c_idx] = s_mel[idx:idx+segment_length][::-1] # Reverse future
                target_dict["beat_l"][c_idx] = s_beat[idx-segment_length:idx]
                target_dict["beat_r"][c_idx] = s_beat[idx:idx+segment_length][::-1]
                target_dict["key_l"][c_idx] = s_key[idx-segment_length:idx]
                target_dict["key_r"][c_idx] = s_key[idx:idx+segment_length][::-1]
                target_dict["chord_l"][c_idx] = s_chord[idx-segment_length:idx]
                target_dict["out"][c_idx] = s_chord[idx]
                
                if target_dict is train_data: t_idx += 1
                else: v_idx += 1

    # Final Trimming
    def trim(d, count):
        return [d["mel_l"][:count], d["mel_r"][:count], 
                d["beat_l"][:count], d["beat_r"][:count], 
                d["key_l"][:count], d["key_r"][:count], 
                d["chord_l"][:count], d["out"][:count]]

    print(f"Training samples: {t_idx}, Validation samples: {v_idx}")
    return trim(train_data, t_idx), trim(val_data, v_idx)


def build_model(segment_length, rnn_size, num_layers, dropout, weights_path=None, chord_types_path=CHORD_TYPES_PATH, training=True):
    
    with open(chord_types_path, "rb") as filepath:
        chord_types = pickle.load(filepath)
    num_chords = len(chord_types)

    # --- INPUTS ---
    i_mel_l = Input(shape=(segment_length,), name='input_melody_left')  # (Batch, 32)
    i_mel_r = Input(shape=(segment_length,), name='input_melody_right') # (Batch, 32)
    i_beat_l = Input(shape=(segment_length,), name='input_beat_left')
    i_beat_r = Input(shape=(segment_length,), name='input_beat_right')
    i_key_l = Input(shape=(segment_length,), name='input_key_left')
    i_key_r = Input(shape=(segment_length,), name='input_key_right')
    i_chord_l = Input(shape=(segment_length,), name='input_chord_left')

    # --- EMBEDDINGS (Shared weights) ---
    # Un "Embedding" transforme un entier (note 60) en un vecteur dense de taille N.
    # C'est beaucoup plus efficace que One-Hot + Dense.
    emb_mel = Embedding(input_dim=129, output_dim=64, name="emb_melody") # 0-128
    emb_beat = Embedding(input_dim=17, output_dim=8, name="emb_beat")    # Beat strength
    emb_key = Embedding(input_dim=25, output_dim=8, name="emb_key")      # Keys
    emb_chord = Embedding(input_dim=num_chords+1, output_dim=64, name="emb_chord")

    # --- FUSION (EARLY CONCATENATION) ---
    # On colle la Mélodie + Beat + Key ENSEMBLE avant le LSTM.
    # Le LSTM verra un vecteur complet représentant "Note X sur Temps Y en Tonalité Z".
    
    # Left Context (Past)
    x_mel_l = emb_mel(i_mel_l)
    x_beat_l = emb_beat(i_beat_l)
    x_key_l = emb_key(i_key_l)
    # Fusion Left
    feat_left = Concatenate(axis=-1)([x_mel_l, x_beat_l, x_key_l]) # shape=(Batch, 32, 64+8+8=80)

    # Right Context (Future)
    x_mel_r = emb_mel(i_mel_r)
    x_beat_r = emb_beat(i_beat_r)
    x_key_r = emb_key(i_key_r)
    # Fusion Right
    feat_right = Concatenate(axis=-1)([x_mel_r, x_beat_r, x_key_r])

    # Chord History
    x_chord_l = emb_chord(i_chord_l)

    # --- RECURRENT LAYERS (LSTM) ---
    
    # Context Branch (Left & Right processed somewhat in parallel but merged conceptually)
    # On utilise separate LSTMs car 'Right' est inversé temporellement par rapport au sens musical
    
    curr_l = feat_left
    curr_r = feat_right
    curr_c = x_chord_l

    for idx in range(num_layers):
        ret_seq = (idx < num_layers - 1) # Return sequences for all except last layer? 
        # Actually for 'merge' strategy at the end, usually we just want the final context vector
        # BUT if we want deep stacking, we propagate sequences.
        # Let's keep sequence propagation until the very last aggregation or use specific architecture.
        # Original model flattened everything at the end. Here we keep it standard.
        
        # Layer Config
        lstm_layer = LSTM(rnn_size, return_sequences=ret_seq, dropout=dropout)
        
        curr_l = lstm_layer(curr_l) # Shared weights or separate? Original had separate.
        # Separate is better because "Future" semantics are different from "Past".
        curr_r = LSTM(rnn_size, return_sequences=ret_seq, dropout=dropout)(curr_r)
        curr_c = LSTM(rnn_size, return_sequences=ret_seq, dropout=dropout)(curr_c)

        if ret_seq:
            curr_l = BatchNormalization()(curr_l)
            curr_r = BatchNormalization()(curr_r)
            curr_c = BatchNormalization()(curr_c)

    # --- MERGE & PREDICTION ---
    # Au final, on a 3 vecteurs contextuels: Passé Mélodique, Futur Mélodique, Passé Harmonique
    merge = Concatenate()([curr_l, curr_r, curr_c])
    
    merge = Dense(rnn_size, activation='relu')(merge)
    merge = BatchNormalization()(merge)
    merge = Dropout(dropout)(merge)
    
    output = Dense(num_chords, activation='softmax', name='output_chord')(merge)

    model = Model(
        inputs=[i_mel_l, i_mel_r, i_beat_l, i_beat_r, i_key_l, i_key_r, i_chord_l],
        outputs=output
    )

    f1 = F1Score(average='macro', name='f1_score')
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy', f1])
    
    if weights_path is None:
        model.summary()
    elif os.path.exists(weights_path):
        # Loading weights might fail if architecture changed significantly
        # User should probably delete old weights file
        try:
             model.load_weights(weights_path)
        except:
             print("Could not load weights due to architecture mismatch. Starting fresh.")

    return model


def train_model(data, data_val, segment_length=SEGMENT_LENGTH, 
                rnn_size=RNN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT,
                epochs=EPOCHS, verbose=1, weights_path=WEIGHTS_PATH):

    with open(CHORD_TYPES_PATH, "rb") as filepath:
        chord_nums = len(pickle.load(filepath))
        
    model = build_model(segment_length, rnn_size, num_layers, dropout)

    # Checkpoint
    monitor = 'val_loss' if len(data_val[0]) > 0 else 'loss'
    checkpoint = ModelCheckpoint(filepath=weights_path, monitor=monitor,
                                 verbose=0, save_best_only=True, mode='min')

    # Generators
    # Note: data indices match the return tuple of create_training_data
    # 0:MelL, 1:MelR, 2:BeatL, 3:BeatR, 4:KeyL, 5:KeyR, 6:ChdL, 7:Out
    
    train_gen = DataGenerator(data[0], data[1], data[2], data[3], data[4], data[5], data[6], 
                              data[7], chord_nums)
    
    val_gen = None
    if len(data_val[0]) > 0:
        val_gen = DataGenerator(data_val[0], data_val[1], data_val[2], data_val[3], data_val[4], data_val[5], data_val[6], 
                                data_val[7], chord_nums)

    history = model.fit(x=train_gen, validation_data=val_gen,
                        epochs=epochs, verbose=verbose, callbacks=[checkpoint])
    return history, model


def plot_history(history, model, train_size, val_size):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n[WARN] Matplotlib not found. Please install it: pip install matplotlib")
        return

    # Extract metrics
    acc = history.history.get('accuracy', [])
    val_acc = history.history.get('val_accuracy', [])
    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])
    
    epochs = range(1, len(acc) + 1)
    
    # Model Stats
    total_params = model.count_params()
    # Calculate trainable params
    trainable_params = np.sum([np.prod(v.shape) for v in model.trainable_weights])
    non_trainable_params = total_params - trainable_params

    plt.figure(figsize=(12, 7)) # Taller to fit text at bottom
    
    # Plot Accuracy
    if acc:
        plt.subplot(1, 2, 1)
        plt.plot(epochs, acc, 'b-', label='Training')
        if val_acc: plt.plot(epochs, val_acc, 'r-', label='Validation')
        plt.title('Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
    
    # Plot Loss
    if loss:
        plt.subplot(1, 2, 2)
        plt.plot(epochs, loss, 'b-', label='Training')
        if val_loss: plt.plot(epochs, val_loss, 'r-', label='Validation')
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
    
    # Add textual info banner
    info_text = (
        f"Dataset: {train_size:,} train sequences / {val_size:,} val sequences\n"
        f"Model Params: {total_params:,} Total | "
        f"{int(trainable_params):,} Trainable | "
        f"{int(non_trainable_params):,} Non-traininable"
    )
    
    plt.figtext(0.5, 0.05, info_text, ha='center', fontsize=11, 
                bbox={"facecolor":"#e6f2ff", "alpha":0.8, "pad":8, "edgecolor":"#b3d9ff"})

    plt.subplots_adjust(bottom=0.2) # Make room for text
    plt.savefig('training_history.png')
    print("\n[INFO] Training plots saved to 'training_history.png'")


def append_history_to_csv(history, model, train_size, val_size, csv_path="training_history.csv"):
    import csv
    import time
    
    # Check if file exists to write header
    file_exists = os.path.isfile(csv_path)
    
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # Header
        if not file_exists:
            writer.writerow([
                "Timestamp", "Dataset Size (Train)", "Dataset Size (Val)", "Params Count",
                "RNN_SIZE", "NUM_LAYERS", "BATCH_SIZE", 
                "Best Val Loss", "Train Loss @ Best", "Best Val Acc", "Train Acc @ Best", "Best Epoch"
            ])
            
        # Metrics logic: The model uses 'val_loss' for checkpointing (min).
        val_losses = history.history.get('val_loss', [])
        train_losses = history.history.get('loss', [])
        val_accs = history.history.get('val_accuracy', [])
        train_accs = history.history.get('accuracy', [])

        best_val_loss = -1
        train_loss_at_best = -1
        best_val_acc = -1
        train_acc_at_best = -1
        best_epoch = -1

        if val_losses:
            best_val_loss = min(val_losses)
            best_idx = val_losses.index(best_val_loss)
            best_epoch = best_idx + 1
            
            # Retrieve corresponding metrics at that specific epoch
            train_loss_at_best = train_losses[best_idx] if len(train_losses) > best_idx else -1
            best_val_acc = val_accs[best_idx] if len(val_accs) > best_idx else -1
            train_acc_at_best = train_accs[best_idx] if len(train_accs) > best_idx else -1
        
        writer.writerow([
            time.strftime("%Y-%m-%d %H:%M:%S"),
            train_size, val_size, model.count_params(),
            RNN_SIZE, NUM_LAYERS, BATCH_SIZE,
            round(best_val_loss, 5), round(train_loss_at_best, 5),
            round(best_val_acc, 5), round(train_acc_at_best, 5),
            best_epoch
        ])
    
    print(f"\n[INFO] Training history appended to '{csv_path}'")


if __name__ == "__main__":
    data, data_val = create_training_data()
    
    train_size = len(data[0])
    val_size = len(data_val[0])
    
    history, model = train_model(data, data_val)
    plot_history(history, model, train_size, val_size)
    append_history_to_csv(history, model, train_size, val_size)

    # Zip weights for easy download/archiving
    if os.path.exists(WEIGHTS_PATH):
        zip_path = WEIGHTS_PATH + '.zip'
        print(f"\n[INFO] Zipping weights to {zip_path}...")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(WEIGHTS_PATH, os.path.basename(WEIGHTS_PATH))
        print("[INFO] Done.")

import os
import warnings
import pickle
import numpy as np
from config import *
from music21 import *
from tqdm import trange
from copy import deepcopy
from model import build_model
from samplings import gamma_sampling
from loader import get_filenames, convert_files

# force CPU-only and suppress TF warnings
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")

# Load chord types once
with open(CHORD_TYPES_PATH, "rb") as fp:
    chord_types = pickle.load(fp)

# Inférence pas à pas (Autoregressive)
# On ne peut pas "batcher" comme avant car chaque accord dépend du précédent généré
INFER_BATCH_SIZE = 1

def generate_chord(chord_model, melody_data, beat_data, key_data,
                   segment_length=SEGMENT_LENGTH, rhythm_gamma=RHYTHM_DENSITY,
                   chord_per_bar=CHORD_PER_BAR):

    chord_types_dict = {chord_types[i]: i for i in range(len(chord_types))}
    # Mapping inverse si besoin (pas utilisé ici pour la prédiction)

    chord_data_list = []

    for song_idx, song_melody in enumerate(melody_data):
        # Prepare inputs (Integer integers, not One-Hot)
        padded_melody = segment_length*[0] + song_melody + segment_length*[0]
        padded_beat   = segment_length*[0] + beat_data[song_idx]   + segment_length*[0]
        padded_key    = segment_length*[0] + key_data[song_idx]    + segment_length*[0]

        # Output buffer (start with zeros/rests)
        # Assuming 0 is 'R' or the padding index.
        # Check chord_types order ideally, but usually 0 is generic.
        song_chord    = segment_length * [0]

        n_steps = len(padded_melody) - 2*segment_length

        # On itère pas à pas
        for t in trange(segment_length, len(padded_melody)-segment_length,
                        desc=f"Song {song_idx+1} [{n_steps} steps]"):

            # Context Indices
            left_start, left_end = t-segment_length, t
            right_start, right_end = t, t+segment_length

            # --- PREPARE INPUTS (Indices) ---
            # 1. Melody
            p_mel_l = np.array([padded_melody[left_start:left_end]], dtype=np.uint8)
            p_mel_r = np.array([padded_melody[right_start:right_end][::-1]], dtype=np.uint8) # Reverse future

            # 2. Beat
            p_beat_l = np.array([padded_beat[left_start:left_end]], dtype=np.uint8)
            p_beat_r = np.array([padded_beat[right_start:right_end][::-1]], dtype=np.uint8)

            # 3. Key
            p_key_l = np.array([padded_key[left_start:left_end]], dtype=np.uint8)
            p_key_r = np.array([padded_key[right_start:right_end][::-1]], dtype=np.uint8)

            # 4. Chord History
            # IMPORTANT: C'est ici que l'autoregressivité joue.
            # On prend les 'segment_length' derniers accords générés
            hist_chord = song_chord[-segment_length:]
            p_chord_l = np.array([hist_chord], dtype=np.uint16)

            # --- MODEL PREDICION ---

            inputs = {
                "input_melody_left": p_mel_l,
                "input_melody_right": p_mel_r,
                "input_beat_left": p_beat_l,
                "input_beat_right": p_beat_r,
                "input_key_left": p_key_l,
                "input_key_right": p_key_r,
                "input_chord_left": p_chord_l
            }

            pred_probs = chord_model.predict(inputs, verbose=0)[0] # Shape (num_chords,)

            # --- SAMPLING STRATEGY ---
            prev_chord_idx = song_chord[-1]
            current_beat = padded_beat[t]

            if chord_per_bar:
                # Force change only on downbeat (beat 4 in some encoding? or 1?)
                # Assuming standard music21: beat strength 1.0 -> 4 int conversion
                gamma = 1 if current_beat == 4 and prev_chord_idx != song_chord[-1] else 0
            else:
                gamma = rhythm_gamma

            # Applying sampling constraints
            tuned_probs = gamma_sampling(pred_probs, [[prev_chord_idx]], [gamma], return_probs=True)

            # Greedy choice (argmax) or probabilistic sample?
            # Usually argmax for stability unless temperature is needed.
            chosen_chord_idx = np.argmax(tuned_probs)

            song_chord.append(chosen_chord_idx)

        chord_data_list.append(song_chord[segment_length:])

    return chord_data_list

def watermark(score, filename, water_mark=WATER_MARK):
    if water_mark:
        score.metadata = metadata.Metadata()
        score.metadata.title = filename
        score.metadata.composer = 'harmonized by AutoHarmonizer'
    return score

def export_music(score, beat_data, chord_data, filename,
                 repeat_chord=REPEAT_CHORD, outputs_path=OUTPUTS_PATH,
                 water_mark=WATER_MARK):

    harmony_list = []
    offset = 0.0
    base = os.path.basename(filename)
    stem = '.'.join(base.split('.')[:-1])

    for idx, song_ch in enumerate(chord_data):
        # Convert indices back to chord names
        labels = [chord_types[int(c)].replace('N.C.', 'R').replace('bpedal', '-pedal') for c in song_ch]
        pre = None
        for t, lbl in enumerate(labels):
            if lbl != 'R' and (lbl != pre or (repeat_chord and beat_data[idx][t] == 4)):
                cs = harmony.ChordSymbol(lbl)
                cs.offset = offset
                harmony_list.append(cs)
            offset += 0.25
            pre = lbl

    new_measures = []
    offsets = []
    h_idx = 0
    for m in score:
        if isinstance(m, stream.Measure):
            new_m = deepcopy(m)
            offsets.append(m.offset)
            elems = []
            for el in new_m:
                while h_idx < len(harmony_list) and el.offset + m.offset >= harmony_list[h_idx].offset:
                    harmony_list[h_idx].offset -= m.offset
                    elems.append(harmony_list[h_idx])
                    h_idx += 1
                elems.append(el)
            new_m.elements = elems
            new_measures.append(new_m)

    final_score = stream.Score(new_measures)
    for i, m in enumerate(final_score):
        m.offset = offsets[i]

    if water_mark:
        final_score = watermark(final_score, stem)

    output_file = f"{outputs_path}/{stem}.mxl"
    final_score.write('mxl', fp=output_file)
    print(f"Exported to {output_file}")

if __name__ == "__main__":
    print("Loading Inputs...")
    files = get_filenames(input_dir=INPUTS_PATH)
    if not files:
        print("No files found in inputs/")
        exit()

    data = convert_files(files, fromDataset=False)

    print("Loading Model...")
    # Weights are loaded inside build_model if path is provided
    model = build_model(SEGMENT_LENGTH, RNN_SIZE, NUM_LAYERS, DROPOUT,
                        weights_path=WEIGHTS_PATH, training=False)

    print("Generating harmony...")
    for md, bd, kd, score_obj, fname in data:
        print(f"Processing {os.path.basename(fname)}...")
        chords = generate_chord(model, md, bd, kd)

        # DEBUG: Analyse des accords générés pour la première chanson
        unique_chords = np.unique([c for song in chords for c in song])
        print(f"DEBUG: Unique Indices Predicted: {unique_chords}")

        # Convert indices to readable labels to check if only 'R'
        labels_sample = [chord_types[int(c)] for song in chords for c in song if chord_types[int(c)] != 'R']
        print(f"DEBUG: Sample chords (excluding 'R'): {labels_sample[:20]}")
        print(f"DEBUG: Total 'R' (Rest) vs Total Indices: {sum(1 for song in chords for c in song if chord_types[int(c)] == 'R')} / {sum(len(s) for s in chords)}")

        export_music(score_obj, bd, chords, fname)

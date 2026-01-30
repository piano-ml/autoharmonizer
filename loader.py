import os
import pickle
import tarfile
import numpy as np
from copy import deepcopy
from tqdm import trange
from music21 import *
from config import *

def quant_score(score):
    
    for element in score.flat:
        onset = np.ceil(element.offset/0.25)*0.25

        if isinstance(element, note.Note) or isinstance(element, note.Rest) or isinstance(element, chord.Chord):
            offset = np.ceil((element.offset+element.quarterLength)/0.25)*0.25
            element.quarterLength = offset - onset

        element.offset = onset

    return score


def get_filenames(input_dir):

    # Automatic Dataset Extraction
    if input_dir == DATASET_PATH and not os.path.exists(input_dir):
        archive_name = DATASET_ARCHIVE
        if os.path.exists(archive_name):
            print(f"[LOADER] Dataset folder not found. Extracting {archive_name}...")
            try:
                # Check structure
                has_root = False
                with tarfile.open(archive_name, "r:gz") as tar:
                    m = tar.next()
                    if m and m.name.startswith("dataset"):
                        has_root = True
                
                # Extract
                with tarfile.open(archive_name, "r:gz") as tar:
                    if has_root:
                         tar.extractall(path=".")
                    else:
                         os.makedirs(input_dir, exist_ok=True)
                         tar.extractall(path=input_dir)
                print("[LOADER] Extraction complete.")
            except Exception as e:
                print(f"[LOADER] Error extracting dataset: {e}")
    
    # Use list comprehension for better performance
    filenames = [
        os.path.join(dirpath, this_file)
        for dirpath, dirlist, filelist in os.walk(input_dir)
        for this_file in filelist
        if input_dir != DATASET_PATH or os.path.splitext(this_file)[-1] in EXTENSION
    ]
    return filenames


def melody_reader(score):
    # Pre-calculate total length for efficient memory allocation
    total_length = 0
    elements_data = []
    sharps = 0
    chord_token = 'R'
    
    for element in score.flat:
        if isinstance(element, note.Note):
            token = element.pitch.midi
            duration = int(element.quarterLength*4)
            beat = int(element.beatStrength*4)
            if duration > 0:  # Skip zero-length notes
                elements_data.append((token, beat, sharps, chord_token, duration))
                total_length += duration
            
        elif isinstance(element, note.Rest):
            token = 0
            duration = int(element.quarterLength*4)
            beat = int(element.beatStrength*4)
            if duration > 0:  # Skip zero-length rests
                elements_data.append((token, beat, sharps, chord_token, duration))
                total_length += duration
            
        elif isinstance(element, chord.Chord) and not isinstance(element, harmony.ChordSymbol):
            notes = [n.pitch.midi for n in element.notes]
            if notes:  # Ensure there are notes
                token = max(notes)  # max() is faster than sort() + [-1]
                duration = int(element.quarterLength*4)
                beat = int(element.beatStrength*4)
                if duration > 0:  # Skip zero-length chords
                    elements_data.append((token, beat, sharps, chord_token, duration))
                    total_length += duration
            
        elif isinstance(element, harmony.ChordSymbol):
            chord_token = element.figure
            
        elif isinstance(element, key.Key) or isinstance(element, key.KeySignature):
            sharps = element.sharps+8
    
    # Preallocate arrays with appropriate dtypes
    melody_txt = np.zeros(total_length, dtype=np.uint8)
    beat_txt = np.zeros(total_length, dtype=np.uint8)
    key_txt = np.zeros(total_length, dtype=np.uint8)
    chord_txt = []
    
    # Fill arrays efficiently
    idx = 0
    for token_val, beat_val, key_val, chord_val, duration in elements_data:
        melody_txt[idx:idx+duration] = token_val
        beat_txt[idx:idx+duration] = beat_val
        key_txt[idx:idx+duration] = key_val
        chord_txt.extend([chord_val] * duration)
        idx += duration
    
    # Convert to lists for compatibility with existing code
    return melody_txt.tolist(), beat_txt.tolist(), key_txt.tolist(), chord_txt


def convert_files(filenames, fromDataset=True):

    print('\nConverting %d files...' %(len(filenames)))
    failed_list = []
    data_corpus = []

    for filename_idx in trange(len(filenames)):

        # Read this music file
        filename = filenames[filename_idx]
        
        try:
            
            score = converter.parse(filename)
            score = score.parts[0]
            if not fromDataset:
                original_score = deepcopy(score)
            song_data = []
            melody_data = []
            beat_data = []
            key_data = []

            score = quant_score(score)
            melody_txt, beat_txt, key_txt, chord_txt = melody_reader(score)

            if fromDataset:
                if len(melody_txt)==len(beat_txt) and len(beat_txt)==len(key_txt) and len(key_txt)==len(chord_txt):
                    song_data.append((melody_txt, beat_txt, key_txt, chord_txt))
                
                else:
                    failed_list.append((filename, 'length mismatch'))
                    song_data = []
                    break

            else:
                if len(melody_txt)!=len(beat_txt) or len(melody_txt)!=len(key_txt):
                    min_len = min(len(melody_txt), len(beat_txt))
                    melody_txt = melody_txt[:min_len]
                    beat_txt = beat_txt[:min_len]
                    key_txt = key_txt[:min_len]
                    
                melody_data.append(melody_txt)
                beat_data.append(beat_txt)
                key_data.append(key_txt)
            
            if not fromDataset:
                data_corpus.append((melody_data, beat_data, key_data, original_score, filename))
            
            elif len(song_data)>0:
                data_corpus.append(song_data)

        except Exception as e:
            failed_list.append((filename, e))

    print('Successfully converted %d files.' %(len(filenames)-len(failed_list)))
    if len(failed_list)>0:
        print('Failed numbers: '+str(len(failed_list)))
        print('Failed to process: \n')
        for failed_file in failed_list:
            print(failed_file)

    if fromDataset:
        chord_types = [song[3] for songs in data_corpus for song in songs]
        chord_types = [item for sublist in chord_types for item in sublist]
        chord_types = list(set(chord_types))
        
        # Only remove 'R' if it exists in the list
        if 'R' in chord_types:
            chord_types.remove('R')
            chord_types = ['R'] + chord_types
        elif len(chord_types) == 0:
            # If no chord types found, use default
            chord_types = ['R']
        
        print(f"Found {len(chord_types)} unique chord types")

        with open(CHORD_TYPES_PATH, "wb") as filepath:
            pickle.dump(chord_types, filepath)

        with open(CORPUS_PATH, "wb") as filepath:
            pickle.dump(data_corpus, filepath)
    
    else:
        return data_corpus


if __name__ == '__main__':

    # Clean up old artifacts to ensure fresh training data
    print("[LOADER] Cleaning up old binary files and weights...")
    for path in [CORPUS_PATH, CHORD_TYPES_PATH, WEIGHTS_PATH]:
        if os.path.exists(path):
            try:
                os.remove(path)
                print(f"  - Deleted {path}")
            except OSError as e:
                print(f"  - Error deleting {path}: {e}")

    filenames = get_filenames(input_dir=DATASET_PATH)
    convert_files(filenames)
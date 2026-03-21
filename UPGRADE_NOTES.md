##  Changes made


### 1 - Architecture Changes (Conceptual)

- **Fused Inputs**: Melody, Beat, and Key are now concatenated *before* the LSTM layers. The model now processes `[Note + Rhythm + Key]` as a single unified context rather than separate isolated streams.

- **Embeddings**: Replaced expensive `OneHot + Dense` layers with native `keras.layers.Embedding`.
  - drastically reduces memory usage (RAM/VRAM).
  - improves semantic representation of musical concepts.


### 2 - Technical Changes

#### 1. Requirements.txt
- TensorFlow: 2.14.0 → ≥2.18.0 (native numpy2 support)
- NumPy: Explicitly added ≥2.0.0
- music21: 7.3.3 → ≥9.1.0 (numpy2 compatibility)


#### 2. Memory Optimizations (model.py)

*DataGenerator*
- Conversion to numpy arrays with optimized dtypes (uint8/uint16)
- Added `on_epoch_end()` for efficient shuffling via indices
- Indexing by indices instead of repeated slicing
- Memory reduction: ~8x for MIDI values (uint8 vs int64)


*create_training_data()*
- Pre-allocation of numpy arrays (avoids repeated append)
- Optimized dtypes: uint8 for melody/beat/key, uint16 for chords
- Direct use of numpy arrays instead of Python lists
- Final trim to free unused space



#### 3 Other Technical Optimizations

- **DataGenerator**:
  - Now yields integer indices (`uint8`/`uint16`) instead of massive one-hot float arrays.
  - Conversion to vectors happens on the GPU via Embedding layers.
  - Fixes CPU->GPU bandwidth bottlenecks and RAM saturation.

- **Code Refactoring**:
  - Modularized `model.py` structure.
  - Removed redundant logic in data loading.

- **Code repo**:
  - git repository with non essentials artefacts (.bin) + generation workflow as code0
  - migrate dataset to .tgz

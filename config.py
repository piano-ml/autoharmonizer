# Path setting
DATASET_PATH = "dataset"
DATASET_ARCHIVE = "dataset.tgz"
CORPUS_PATH = "data_corpus.bin"
CHORD_TYPES_PATH = 'chord_types.bin'
WEIGHTS_PATH = 'weights.keras'
INPUTS_PATH = "inputs"
OUTPUTS_PATH = "outputs"

# 'loader.py'
EXTENSION = ['.musicxml', '.xml', '.mxl']

# '.model.py'
VAL_RATIO = 0.1
DROPOUT = 0.2
# RAM Impact: Increases input data size linearly.
SEGMENT_LENGTH = 32

# Model capacity: Higher (256/512) = "smarter" but slower/riskier overfitting; Lower (64/128) = faster/lighter.
# RAM Impact: High. Quadratic growth in weights. Best = 64 for current dataset.
RNN_SIZE = 128

# Network Depth: More layers (3-4) = can learn complex patterns; fewer (1-2) = simpler, faster training.
# RAM Impact: Linear growth in weights. Best = 3 for current dataset.
NUM_LAYERS = 3

# Training Batch: Higher (256+) = Faster epochs but less generalization; Lower (32/64) = Better accuracy but slower.
# RAM Impact: Very High. Linear growth. Best = 32 for current dataset.
BATCH_SIZE = 32

# Use training history to adjust usefull epochs is at lowest val_loss
EPOCHS = 3

# 'harmonizor.py'
RHYTHM_DENSITY = 0.5
CHORD_PER_BAR = False
REPEAT_CHORD = False
WATER_MARK = False

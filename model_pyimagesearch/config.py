# Import the necessary packages
import torch
import os

TRAINING_TYPE = "BINARY"
# TRAINING_TYPE = "MULTICLASS"

CONTINUE_TRAINING = False

ENCODER_CHANNELS = (3, 128, 256, 512, 1024)
DECODER_CHANNELS = (1024, 512, 256, 128)

INTERPOLATE = False

INPUT_IMAGE_WIDTH = int(960/2)
INPUT_IMAGE_HEIGHT = int(736/2)
OUTPUT_IMAGE_WIDTH = int(960/2)
OUTPUT_IMAGE_HEIGHT = int(736/2)

INIT_LR = 0.001
NUM_EPOCHS = 200
BATCH_SIZE = 16

TEST_SPLIT = 0.15
SPLIT_SEED = 12

# Base path of the datasets
BINARY_DATASET_PATH = "dataset/binary_dataset/binary_dataset"
MULTICLASS_DATASET_PATH = "dataset/classes_dataset/classes_dataset"

# Define the path to the images and masks dataset
IMAGE_BINARY_DATASET_PATH = os.path.join(BINARY_DATASET_PATH, "original_images")
MASK_BINARY_DATASET_PATH = os.path.join(BINARY_DATASET_PATH, "images_semantic")
IMAGE_MULTICLASS_DATASET_PATH = os.path.join(MULTICLASS_DATASET_PATH, "original_images")
MASK_MULTICLASS_DATASET_PATH = os.path.join(MULTICLASS_DATASET_PATH, "label_images_semantic")

# Define threshold to filter weak predictions
PRED_THRESHOLD = 0.5
PRED_THRESHOLDS_MULTI = (0.5, 0.5, 0.5, 0.5, 0.5)

# Mask values
DICT_MASKS = ["obstacles", "water", "soft-surfaces", "moving_objects", "landing-zones"]
MASK_VALUES = (89, 106, 184, 104, 169)
PLOT_MASK_VALUES = (0, 64, 128, 192, 255)

# Determine the device to be used for training and evaluation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Determine if we will be pinning memory during data loading
PIN_MEMORY = True if DEVICE == "cuda" else False

if TRAINING_TYPE == 'BINARY':
    NB_CLASSES = 1
elif TRAINING_TYPE == 'MULTICLASS':
    NB_CLASSES = 5

# Define the path to the base output directory
BASE_OUTPUT = "output"

# Define the path to the output serialized model, model training plot, and testing image paths
BINARY_MODEL_PATH = os.path.join(BASE_OUTPUT, "binary_unet.pth")
BINARY_PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "binary_plot.png"])
MULTICLASS_MODEL_PATH = os.path.join(BASE_OUTPUT, "multiclass_unet.pth")
MULTICLASS_PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "multiclass_plot.png"])
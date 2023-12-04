# Import the necessary packages
import torch
import os

# Base path of the dataset
BINARY_DATASET_PATH = "dataset/binary_dataset/binary_dataset"
MULTICLASS_DATASET_PATH = "dataset/classes_dataset/classes_dataset"

# Define the path to the images and masks dataset
IMAGE_BINARY_DATASET_PATH = os.path.join(BINARY_DATASET_PATH, "original_images")
MASK_BINARY_DATASET_PATH = os.path.join(BINARY_DATASET_PATH, "images_semantic")
IMAGE_MULTICLASS_DATASET_PATH = os.path.join(MULTICLASS_DATASET_PATH, "original_images")
MASK_MULTICLASS_DATASET_PATH = os.path.join(MULTICLASS_DATASET_PATH, "label_images_semantic")

# Define the test split
TEST_SPLIT = 0.15

# Determine the device to be used for training and evaluation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Determine if we will be pinning memory during data loading
PIN_MEMORY = True if DEVICE == "cuda" else False

# Define the size of the encoder and decoder channels
ENCODER_CHANNELS = (3, 64, 128, 256, 512)
DECODER_CHANNELS = (512, 256, 128, 64)

# Initialize learning rate, number of epochs to train for, and the batch size
INIT_LR = 0.004
NUM_EPOCHS = 30
BATCH_SIZE = 64

# Define the input image dimensions
INPUT_IMAGE_WIDTH = int(960/2)
INPUT_IMAGE_HEIGHT = int(736/2)

# Mask values
MASK_VALUES = ()

# Define threshold to filter weak predictions
PRED_THRESHOLD = 0.5

# Define the path to the base output directory
BASE_OUTPUT = "output"

# Define the path to the output serialized model, model training plot, and testing image paths
BINARY_MODEL_PATH = os.path.join(BASE_OUTPUT, "binary_unet_test.pth")
BINARY_PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "binary_plot.png"])
BINARY_TEST_PATHS = os.path.sep.join([BASE_OUTPUT, "binary_test_paths.txt"])

MULTICLASS_MODEL_PATH = os.path.join(BASE_OUTPUT, "multiclass_unet_test.pth")
MULTICLASS_PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "multiclass_plot.png"])
MULTICLASS_TEST_PATHS = os.path.sep.join([BASE_OUTPUT, "multiclass_test_paths.txt"])
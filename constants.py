# Parameters for semantic segmentation
from pathlib import Path


WINDOW_SIZE = (256, 256)  # Patch size

STRIDE = 32  # Stride for testing
IN_CHANNELS = 3  # Number of input channels (e.g. RGB)
FOLDER = "./data/"  # Default data folder path
BATCH_SIZE = 10
# BATCH_SIZE = 4 # For backbone ViT-Huge

LABELS = ["background", "buildings"]  # Binary classification: background vs buildings
N_CLASSES = len(LABELS)  # Number of classes
CACHE = True  # Store the dataset in-memory

DATASET_DIR = Path("./data/Aarvi")
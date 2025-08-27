# Parameters
## SwinFusion
WINDOW_SIZE = (256, 256)  # Patch size

STRIDE = 32  # Stride for testing
IN_CHANNELS = 3  # Number of input channels (e.g. RGB)
FOLDER = "/media/lscsc/nas/xianping/ISPRS_dataset/"  # Replace with your "/path/to/the/ISPRS/dataset/folder/"
BATCH_SIZE = 10
# BATCH_SIZE = 4 # For backbone ViT-Huge

LABELS = ["roads", "buildings", "low veg.", "trees", "cars", "clutter"]  # Label names
N_CLASSES = len(LABELS)  # Number of classes
CACHE = True  # Store the dataset in-memory

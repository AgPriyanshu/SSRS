import numpy as np
from sklearn.metrics import confusion_matrix
import random
import torch
import torch.nn.functional as F
import itertools
from torchvision.utils import make_grid
from PIL import Image
from skimage import io
import os
from constants import N_CLASSES

WEIGHTS = torch.ones(N_CLASSES)  # Weights for class balancing

# Default color palette for semantic segmentation
palette = {
    0: (0, 0, 0),        # Background (black)
    1: (255, 0, 0),      # Class 1 (red)
    2: (0, 255, 0),      # Class 2 (green)
    3: (0, 0, 255),      # Class 3 (blue)
    4: (255, 255, 0),    # Class 4 (yellow)
    5: (255, 0, 255),    # Class 5 (magenta)
    6: (0, 255, 255),    # Class 6 (cyan)
}

invert_palette = {v: k for k, v in palette.items()}

# Legacy configuration variables (for backward compatibility)
# These will be overridden by the new config system
MODEL = "UNetformer"
MODE = "Train"
DATASET = "default"
IF_SAM = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Default values (will be overridden by config)
train_ids = ["train_001", "train_002", "train_003"]
test_ids = ["test_001", "test_002"]
Stride_Size = 64
epochs = 50
save_epoch = 1

# These will be set by the dataset configuration
DATA_FOLDER = None
DSM_FOLDER = None  
LABEL_FOLDER = None
ERODED_FOLDER = None


def convert_to_color(arr_2d, palette=palette):
    """Numeric labels to RGB-color encoding"""
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)

    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i

    return arr_3d


def convert_from_color(arr_3d, palette=invert_palette):
    """RGB-color encoding to grayscale labels"""
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i

    return arr_2d


def save_img(tensor, name):
    tensor = tensor.cpu().permute((1, 0, 2, 3))
    im = make_grid(tensor, normalize=True, scale_each=True, nrow=8, padding=2).permute(
        (1, 2, 0)
    )
    im = (im.data.numpy() * 255.0).astype(np.uint8)
    Image.fromarray(im).save(name + ".jpg")





# Example dataset loading and visualization (commented out)
# img = io.imread('./data/images/img_001.jpg')
# fig = plt.figure()
# fig.add_subplot(121)
# plt.imshow(img)
#
# # Load the ground truth
# gt = io.imread('./data/labels/label_001.png')
# fig.add_subplot(122)
# plt.imshow(gt)
# plt.show()
#
# # Convert ground truth to numerical format
# array_gt = convert_from_color(gt)
# print("Ground truth in numerical format has shape ({},{}) : \n".format(*array_gt.shape[:2]), array_gt)


# Utils


def get_random_pos(img, window_shape):
    """Extract of 2D random patch of shape window_shape in the image"""
    w, h = window_shape
    W, H = img.shape[-2:]
    x1 = random.randint(0, W - w - 1)
    x2 = x1 + w
    y1 = random.randint(0, H - h - 1)
    y2 = y1 + h
    return x1, x2, y1, y2


def CrossEntropy2d(input, target, weight=None, size_average=True):
    """2D version of the cross entropy loss"""
    dim = input.dim()
    if dim == 2:
        return F.cross_entropy(input, target, weight, size_average)
    elif dim == 4:
        output = input.view(input.size(0), input.size(1), -1)
        output = torch.transpose(output, 1, 2).contiguous()
        output = output.view(-1, output.size(2))
        target = target.view(-1)
        return F.cross_entropy(output, target, weight, size_average)
    else:
        raise ValueError("Expected 2 or 4 dimensions (got {})".format(dim))


def accuracy(input, target):
    return 100 * float(np.count_nonzero(input == target)) / target.size


def sliding_window(top, step=10, window_size=(20, 20)):
    """Slide a window_shape window across the image with a stride of step"""
    for x in range(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            yield x, y, window_size[0], window_size[1]


def count_sliding_window(top, step=10, window_size=(20, 20)):
    """Count the number of windows in an image"""
    c = 0
    for x in range(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            c += 1
    return c


def grouper(n, iterable):
    """Browse an iterator by chunk of n elements"""
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


def metrics(predictions, gts, label_values=None):
    if label_values is None:
        from constants import LABELS
        label_values = LABELS
    cm = confusion_matrix(gts, predictions, labels=range(len(label_values)))

    print("Confusion matrix :")
    print(cm)
    # Compute global accuracy
    total = sum(sum(cm))
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy *= 100 / float(total)
    print("%d pixels processed" % (total))
    print("Total accuracy : %.2f" % (accuracy))

    Acc = np.diag(cm) / cm.sum(axis=1)
    for l_id, score in enumerate(Acc):
        print("%s: %.4f" % (label_values[l_id], score))
    print("---")

    # Compute F1 score
    F1Score = np.zeros(len(label_values))
    for i in range(len(label_values)):
        try:
            F1Score[i] = 2.0 * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]))
        except:
            # Ignore exception if there is no element in class i for test set
            pass
    print("F1Score :")
    for l_id, score in enumerate(F1Score):
        print("%s: %.4f" % (label_values[l_id], score))
    print("mean F1Score: %.4f" % (np.nanmean(F1Score[:5])))
    print("---")

    # Compute kappa coefficient
    total = np.sum(cm)
    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / float(total * total)
    kappa = (pa - pe) / (1 - pe)
    print("Kappa: %.4f" % (kappa))

    # Compute MIoU coefficient
    MIoU = np.diag(cm) / (np.sum(cm, axis=1) + np.sum(cm, axis=0) - np.diag(cm))
    print(MIoU)
    MIoU = np.nanmean(MIoU[:5])
    print("mean MIoU: %.4f" % (MIoU))
    print("---")

    return MIoU

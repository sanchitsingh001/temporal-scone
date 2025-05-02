import scipy.io
import numpy as np

# Load splits (train/val/test IDs are 1-indexed)
mat = scipy.io.loadmat("setid.mat")
train_ids = mat["trnid"][0]    # Train IDs
val_ids = mat["valid"][0]      # Validation IDs
test_ids = mat["tstid"][0]     # Test IDs

# Load labels (1-indexed)
labels = scipy.io.loadmat("imagelabels.mat")["labels"][0]

# Get class IDs present in ALL splits
train_classes = set(labels[train_ids - 1])
val_classes = set(labels[val_ids - 1])
test_classes = set(labels[test_ids - 1])
common_classes = sorted(train_classes & val_classes & test_classes)

print(f"Total common classes: {len(common_classes)}")  # Should be 102
import random

random.seed(42)  # For reproducibility
sampled_classes = random.sample(common_classes, 10)
print("Sampled classes:", sampled_classes)

import os
from shutil import copy

# Create output directories
os.makedirs("flowers10/train", exist_ok=True)
os.makedirs("flowers10/val", exist_ok=True)
os.makedirs("flowers10/test", exist_ok=True)

def filter_images(split_ids, split_name):
    for img_id in split_ids:
        class_id = labels[img_id - 1]
        if class_id in sampled_classes:
            src = f"jpg/image_{img_id:05d}.jpg"
            dst = f"flowers10/{split_name}/class_{class_id}/image_{img_id}.jpg"
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            copy(src, dst)

# Copy images for the 10 classes
filter_images(train_ids, "train")
filter_images(val_ids, "val")
filter_images(test_ids, "test")
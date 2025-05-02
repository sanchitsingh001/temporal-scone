import os
import shutil
import random
from tqdm import tqdm

# Set seed for reproducibility
random.seed(42)

# Original split location
source_root = "cub200_split"
output_root = "cub200_split_10"

# Get full list of class folders from CUB
all_classes = sorted(os.listdir(os.path.join(source_root, "train")))

# Select 10 classes
sampled_classes = random.sample(all_classes, 10)
print("Selected classes:", sampled_classes)

# Create new output dirs
for split in ['train', 'test']:
    for i in range(10):
        os.makedirs(os.path.join(output_root, split, f"class_{i}"), exist_ok=True)

# Copy sampled classes and rename to class_0 to class_9
for new_idx, class_name in enumerate(sampled_classes):
    for split in ['train', 'test']:
        src_dir = os.path.join(source_root, split, class_name)
        dst_dir = os.path.join(output_root, split, f"class_{new_idx}")
        print(f"Copying {split}/{class_name} → {split}/class_{new_idx}")
        for fname in tqdm(os.listdir(src_dir), desc=f"{split}/class_{new_idx}", leave=False):
            shutil.copyfile(os.path.join(src_dir, fname), os.path.join(dst_dir, fname))

print("\n✅ Done. 10-class CUB subset is ready at:", output_root)

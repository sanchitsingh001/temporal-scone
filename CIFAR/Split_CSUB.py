import os
import shutil
from tqdm import tqdm

# Paths (change these as needed)
cub_root = "CUB_200_2011"  # Directory after extracting CUB_200_2011.tgz
output_root = "cub200_split"  # Where train/ and test/ folders will be created

# Read metadata
with open(os.path.join(cub_root, "images.txt")) as f:
    id_to_path = dict(line.strip().split() for line in f)

with open(os.path.join(cub_root, "image_class_labels.txt")) as f:
    id_to_class = {img_id: int(cls_id) for img_id, cls_id in (line.strip().split() for line in f)}

with open(os.path.join(cub_root, "train_test_split.txt")) as f:
    id_to_split = {img_id: int(is_train) for img_id, is_train in (line.strip().split() for line in f)}

# Create output dirs
for split in ['train', 'test']:
    for class_id in range(1, 201):  # 200 classes, indexed from 1
        class_dir = os.path.join(output_root, split, f"class_{class_id:03d}")
        os.makedirs(class_dir, exist_ok=True)

# Copy files
images_dir = os.path.join(cub_root, "images")
print("Copying files...")

for img_id, rel_path in tqdm(id_to_path.items(), total=len(id_to_path)):
    class_id = id_to_class[img_id]
    is_train = id_to_split[img_id]
    split = 'train' if is_train == 1 else 'test'
    src_path = os.path.join(images_dir, rel_path)
    dst_path = os.path.join(output_root, split, f"class_{class_id:03d}", os.path.basename(rel_path))
    shutil.copyfile(src_path, dst_path)

print(" Done. Output structure:")
print(f"{output_root}/train/class_001/...")
print(f"{output_root}/test/class_001/...")


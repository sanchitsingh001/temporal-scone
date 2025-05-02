# import os
# import cv2
# import numpy as np
# from imagecorruptions import get_corruption_names, corrupt
# from tqdm import tqdm
# from multiprocessing import Pool, cpu_count
# import warnings
# warnings.filterwarnings("ignore", category=UserWarning)  # Suppress OpenCV warnings

# # Configuration
# ALL_CORRUPTIONS = get_corruption_names()  # All 15+ corruptions
# SEVERITY = 3
# BASE_DIR = "flowers10"
# OUTPUT_DIR = "flowers102_corrupted_npy"
# NUM_PROCESSES = max(1, cpu_count() - 1)  # Use all cores except one

# def process_single_image(args):
#     """Process one image and save all corruptions as .npy (parallel worker)"""
#     img_path, output_class_dir = args
#     try:
#         image = cv2.imread(img_path)
#         if image is None:
#             raise ValueError(f"Failed to read {img_path}")
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
#         corruptions_dict = {}
#         for corruption in ALL_CORRUPTIONS:
#             try:
#                 corrupted = corrupt(image, corruption_name=corruption, severity=SEVERITY)
#                 corruptions_dict[corruption] = corrupted
#             except Exception as e:
#                 print(f"Skipping {corruption} for {os.path.basename(img_path)}: {str(e)}")
        
#         # Save as compressed .npy
#         output_path = os.path.join(
#             output_class_dir,
#             f"{os.path.splitext(os.path.basename(img_path))[0]}.npz"
#         )
#         np.savez_compressed(
#             output_path,
#             original=image,
#             **corruptions_dict,
#             allow_pickle=False  # Safer and smaller files
#         )
#         return True
#     except Exception as e:
#         print(f"Critical error processing {img_path}: {str(e)}")
#         return False

# def apply_parallel(input_dir, output_dir):
#     """Process all images in a directory using multiprocessing"""
#     os.makedirs(output_dir, exist_ok=True)
#     tasks = []
    
#     for class_name in os.listdir(input_dir):
#         class_dir = os.path.join(input_dir, class_name)
#         output_class_dir = os.path.join(output_dir, class_name)
#         os.makedirs(output_class_dir, exist_ok=True)
        
#         # Prepare tasks for parallel processing
#         tasks.extend([
#             (os.path.join(class_dir, img_name), output_class_dir)
#             for img_name in os.listdir(class_dir)
#             if img_name.lower().endswith(('.jpg', '.jpeg', '.png'))
#         ])
    
#     # Process with progress bar
#     with Pool(processes=NUM_PROCESSES) as pool:
#         results = list(tqdm(
#             pool.imap(process_single_image, tasks),
#             total=len(tasks),
#             desc="Processing images"
#         ))
    
#     print(f"Success rate: {sum(results)}/{len(results)}")

# def verify_integrity(file_path):
#     """Check if an .npz file contains all expected keys"""
#     try:
#         data = np.load(file_path)
#         required_keys = ['original'] + ALL_CORRUPTIONS
#         missing = [k for k in required_keys if k not in data]
#         if missing:
#             print(f"Missing keys in {file_path}: {missing}")
#             return False
#         return True
#     except Exception as e:
#         print(f"Corrupted file {file_path}: {str(e)}")
#         return False

# # --- Execute Pipeline ---
# if __name__ == "__main__":
#     # Step 1: Apply corruptions in parallel
#     print(f"Starting processing with {NUM_PROCESSES} cores...")
#     apply_parallel(f"{BASE_DIR}/train", f"{OUTPUT_DIR}/train")
#     apply_parallel(f"{BASE_DIR}/val", f"{OUTPUT_DIR}/val")
#     apply_parallel(f"{BASE_DIR}/test", f"{OUTPUT_DIR}/test")
    
#     # Step 2: Verify integrity (optional)
#     print("\nVerifying file integrity...")
#     bad_files = []
#     for root, _, files in os.walk(OUTPUT_DIR):
#         for file in files:
#             if file.endswith(".npz"):
#                 if not verify_integrity(os.path.join(root, file)):
#                     bad_files.append(os.path.join(root, file))
    
#     print(f"Integrity check complete. Bad files: {len(bad_files)}")
#     if bad_files:
#         print("Sample bad files:", bad_files[:3])



import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from imagecorruptions import get_corruption_names, corrupt
import argparse
import shutil
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Default: Use all corruptions
CORRUPTIONS = get_corruption_names()

def resize_and_corrupt_image(args):
    img_path, class_name, class_idx, temp_dir, severity = args
    try:
        # Resize to 32x32
        img = Image.open(img_path).convert("RGB").resize((32, 32), Image.BILINEAR)
        img_np = np.array(img)

        corruption_dict = {}
        for c in CORRUPTIONS:
            try:
                corrupted = corrupt(img_np, corruption_name=c, severity=severity)
                corruption_dict[c] = corrupted
            except Exception as e:
                print(f"Error: {img_path} - {c}: {e}")
        
        # Save as .npz
        out_dir = os.path.join(temp_dir, class_name)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, os.path.splitext(os.path.basename(img_path))[0] + '.npz')
        np.savez_compressed(out_path, original=img_np, **corruption_dict)
        return class_idx
    except Exception as e:
        print(f"Critical error on {img_path}: {e}")
        return None

def collect_npz(temp_dir):
    corruption_arrays = {name: [] for name in CORRUPTIONS}
    labels = []

    class_names = sorted(os.listdir(temp_dir))
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    for class_name in class_names:
        class_path = os.path.join(temp_dir, class_name)
        for fname in os.listdir(class_path):
            if not fname.endswith('.npz'):
                continue
            try:
                fpath = os.path.join(class_path, fname)
                data = np.load(fpath)
                for c in CORRUPTIONS:
                    corruption_arrays[c].append(data[c])
                labels.append(class_to_idx[class_name])
            except Exception as e:
                print(f"Skipping {fname}: {e}")
    return corruption_arrays, labels

def save_to_npy(arrays, labels, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for name, arr_list in arrays.items():
        np.save(os.path.join(output_dir, f"{name}.npy"), np.stack(arr_list))
    np.save(os.path.join(output_dir, "labels.npy"), np.array(labels))
    print(f"\n✅ Saved all corrupted .npy files to {output_dir}")

def main(input_dir, output_dir, severity):
    temp_dir = os.path.join(output_dir, "_temp_npz")
    os.makedirs(temp_dir, exist_ok=True)
    print(f"\n Using temp directory: {temp_dir}")

    class_names = sorted(os.listdir(input_dir))
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    tasks = []
    for class_name in class_names:
        class_path = os.path.join(input_dir, class_name)
        for fname in os.listdir(class_path):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                tasks.append((os.path.join(class_path, fname), class_name, class_to_idx[class_name], temp_dir, severity))

    print(f"\n Starting corruption (severity {severity}) on {len(tasks)} images using {max(1, cpu_count()-1)} cores...")
    with Pool(processes=max(1, cpu_count()-1)) as pool:
        list(tqdm(pool.imap(resize_and_corrupt_image, tasks), total=len(tasks)))

    print("\nConverting .npz files to final .npy format...")
    arrays, labels = collect_npz(temp_dir)
    save_to_npy(arrays, labels, output_dir)

    shutil.rmtree(temp_dir)
    print(f" Removed temporary directory: {temp_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Path to input folder with class subdirs')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to save .npy files')
    parser.add_argument('--severity', type=int, default=3, choices=range(1, 6),
                        help='Severity level of corruption (1-5)')
    args = parser.parse_args()

    main(args.input_dir, args.output_dir, args.severity)

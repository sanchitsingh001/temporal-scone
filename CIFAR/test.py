import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from load_any_dataset import load_cifar
from make_datasets import make_datasets
import numpy as np

def compare_tensors(tensor1, tensor2, name="", is_label=False):
    """Compare two tensors and print their statistics"""
    print(f"\nComparing {name}:")
    print(f"Shape: {tensor1.shape} vs {tensor2.shape}")
    
    if not is_label:
        print(f"Mean: {tensor1.mean():.4f} vs {tensor2.mean():.4f}")
        print(f"Std: {tensor1.std():.4f} vs {tensor2.std():.4f}")
        print(f"Min: {tensor1.min():.4f} vs {tensor2.min():.4f}")
        print(f"Max: {tensor1.max():.4f} vs {tensor2.max():.4f}")
        print(f"Are equal: {torch.allclose(tensor1, tensor2, rtol=1e-3, atol=1e-3)}")
    else:
        # For labels, compare unique values and their counts
        unique1, counts1 = torch.unique(tensor1, return_counts=True)
        unique2, counts2 = torch.unique(tensor2, return_counts=True)
        print("Unique values in first tensor:", unique1.tolist())
        print("Counts in first tensor:", counts1.tolist())
        print("Unique values in second tensor:", unique2.tolist())
        print("Counts in second tensor:", counts2.tolist())
        print("Are equal:", torch.equal(tensor1, tensor2))

def compare_loaders(loader1, loader2, name):
    """Compare two dataloaders"""
    print(f"\n=== Comparing {name} ===")
    
    # Handle None loaders
    if loader1 is None or loader2 is None:
        print(f"One or both loaders are None: {loader1 is None} vs {loader2 is None}")
        return
    
    # Get batches from both loaders
    batch1 = next(iter(loader1))
    batch2 = next(iter(loader2))
    
    images1, labels1 = batch1
    images2, labels2 = batch2
    
    # Compare images
    compare_tensors(images1, images2, f"{name} Images")
    
    # Compare labels
    compare_tensors(labels1, labels2, f"{name} Labels", is_label=True)
    
    # Compare loader properties
    print(f"\nLoader properties for {name}:")
    print(f"Batch size: {loader1.batch_size} vs {loader2.batch_size}")
    print(f"Number of workers: {loader1.num_workers} vs {loader2.num_workers}")
    print(f"Dataset size: {len(loader1.dataset)} vs {len(loader2.dataset)}")
    print(f"Number of batches: {len(loader1)} vs {len(loader2)}")

def test_cifar10_loading():
    # State parameters (matching make_datasets.py defaults)
    state = {
        'batch_size': 128,
        'prefetch': 4,
        'seed': 42
    }

    print("\n=== Loading CIFAR10 using make_datasets.py ===")
    # Using make_datasets.py with CIFAR10 for both in and out distribution
    make_datasets_loaders = make_datasets(in_dset='cifar10', aux_out_dset='lsun_c', test_out_dset='lsun_c', state ={'batch_size': 128, 'prefetch': 4, 'seed': 42}, alpha=0.5, pi_1=0.5, pi_2=0.1, cortype='gaussian_noise')

    
    print("\n=== Loading CIFAR10 using load_cifar ===")
    # Using load_cifar with default parameters
    load_cifar_loaders = load_cifar()
    
    # Compare each loader pair
    loader_names = [
        "train_loader_in",
        "train_loader_aux_in",
        "train_loader_aux_in_cor",
        "train_loader_aux_out",
        "test_loader_in",
        "test_loader_cor",
        "test_loader_out",
        "valid_loader_in",
        "valid_loader_aux"
    ]
    
    for name, loader1, loader2 in zip(loader_names, make_datasets_loaders, load_cifar_loaders):
        compare_loaders(loader1, loader2, name)

if __name__ == "__main__":
    test_cifar10_loading()

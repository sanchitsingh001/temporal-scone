import torch
import torchvision.transforms as trn
import torchvision.datasets as dset
from torch.utils.data import DataLoader, TensorDataset, Dataset, Subset, ConcatDataset
import numpy as np
import os
import random
import torch.nn.functional as F
import time

def apply_corruption(image, corruption_type, severity=1):
    """
    Apply corruption to an image.
    
    Args:
        image (torch.Tensor): Input image tensor
        corruption_type (str): Type of corruption to apply
        severity (int): Severity level of corruption (1-5)
    
    Returns:
        torch.Tensor: Corrupted image
    """
    if corruption_type == 'gaussian_noise':
        noise = torch.randn_like(image) * (0.1 * severity)
        return torch.clamp(image + noise, 0, 1)
    
    elif corruption_type == 'shot_noise':
        noise = torch.poisson(image * (10 * severity)) / (10 * severity)
        return torch.clamp(noise, 0, 1)
    
    elif corruption_type == 'impulse_noise':
        mask = torch.rand_like(image) < (0.1 * severity)
        noise = torch.rand_like(image)
        return torch.where(mask, noise, image)
    
    elif corruption_type == 'defocus_blur':
        kernel_size = 2 * severity + 1
        kernel = torch.ones(1, 1, kernel_size, kernel_size) / (kernel_size * kernel_size)
        kernel = kernel.to(image.device)
        return F.conv2d(image.unsqueeze(0), kernel, padding=kernel_size//2).squeeze(0)
    
    elif corruption_type == 'glass_blur':
        # Simplified glass blur effect
        kernel_size = 2 * severity + 1
        kernel = torch.ones(1, 1, kernel_size, kernel_size) / (kernel_size * kernel_size)
        kernel = kernel.to(image.device)
        blurred = F.conv2d(image.unsqueeze(0), kernel, padding=kernel_size//2).squeeze(0)
        return 0.7 * image + 0.3 * blurred
    
    elif corruption_type == 'motion_blur':
        kernel_size = 2 * severity + 1
        kernel = torch.zeros(1, 1, kernel_size, kernel_size)
        kernel[0, 0, :, kernel_size//2] = 1.0 / kernel_size
        kernel = kernel.to(image.device)
        return F.conv2d(image.unsqueeze(0), kernel, padding=kernel_size//2).squeeze(0)
    
    elif corruption_type == 'zoom_blur':
        scales = [1.0 + 0.1 * i * severity for i in range(5)]
        blurred = torch.zeros_like(image)
        for scale in scales:
            size = int(image.shape[-1] * scale)
            resized = F.interpolate(image.unsqueeze(0), size=(size, size), mode='bilinear', align_corners=False)
            resized = F.interpolate(resized, size=image.shape[-2:], mode='bilinear', align_corners=False)
            blurred += resized.squeeze(0)
        return blurred / len(scales)
    
    elif corruption_type == 'snow':
        snow_layer = torch.rand_like(image) * (0.1 * severity)
        return torch.clamp(image + snow_layer, 0, 1)
    
    elif corruption_type == 'frost':
        frost = torch.rand_like(image) * (0.1 * severity)
        return torch.clamp(image + frost, 0, 1)
    
    elif corruption_type == 'fog':
        fog = torch.ones_like(image) * (0.1 * severity)
        return torch.clamp(image + fog, 0, 1)
    
    elif corruption_type == 'brightness':
        return torch.clamp(image * (1.0 + 0.1 * severity), 0, 1)
    
    elif corruption_type == 'contrast':
        mean = torch.mean(image, dim=[1, 2], keepdim=True)
        return torch.clamp((image - mean) * (1.0 + 0.1 * severity) + mean, 0, 1)
    
    elif corruption_type == 'elastic':
        # Simplified elastic transform
        noise = torch.randn(2, image.shape[-2], image.shape[-1]) * (0.1 * severity)
        noise = noise.to(image.device)
        grid = torch.stack(torch.meshgrid(torch.arange(image.shape[-2]), torch.arange(image.shape[-1]))).float()
        grid = grid.to(image.device)
        grid = grid + noise
        grid = grid.permute(1, 2, 0).unsqueeze(0)
        return F.grid_sample(image.unsqueeze(0), grid, align_corners=False).squeeze(0)
    
    elif corruption_type == 'pixelate':
        scale = 1.0 / (1.0 + 0.1 * severity)
        size = int(image.shape[-1] * scale)
        downsampled = F.interpolate(image.unsqueeze(0), size=(size, size), mode='nearest')
        return F.interpolate(downsampled, size=image.shape[-2:], mode='nearest').squeeze(0)
    
    elif corruption_type == 'jpeg':
        # Simplified JPEG compression
        return torch.clamp(image + torch.randn_like(image) * (0.1 * severity), 0, 1)
    
    else:
        raise ValueError(f"Unknown corruption type: {corruption_type}")

class CorruptedDataset(Dataset):
    def __init__(self, dataset, corruption_type, severity=1):
        self.dataset = dataset
        self.corruption_type = corruption_type
        self.severity = severity
        
    def __getitem__(self, index):
        image, label = self.dataset[index]
        corrupted_image = apply_corruption(image, self.corruption_type, self.severity)
        return corrupted_image, label
        
    def __len__(self):
        return len(self.dataset)

def load_cifar(dataset='cifar10', batch_size=128, num_workers=4, alpha=0.5, pi_1=0.5, pi_2=0.1, cortype='gaussian_noise', seed=42):
    """
    Load CIFAR10 dataset with corrupted data using on-the-fly corruptions.
    Matches the behavior of make_datasets.py for CIFAR10.
    
    Args:
        dataset (str): Dataset name (only 'cifar10' supported)
        batch_size (int): Batch size for dataloaders
        num_workers (int): Number of workers for dataloaders
        alpha (float): Proportion of training data to use for training
        pi_1 (float): Proportion of validation data to use for validation
        pi_2 (float): Proportion of validation data to use for validation
        cortype (str): Type of corruption to apply
        seed (int): Random seed for reproducibility
        
    Returns:
        tuple: (train_loader_in, train_loader_aux_in, train_loader_aux_in_cor, train_loader_aux_out,
                test_loader_in, test_loader_cor, test_loader_out, valid_loader_in, valid_loader_aux)
    """
    if dataset != 'cifar10':
        raise ValueError("Only CIFAR10 is supported in this version")
        
    # Set random seed
    rng = np.random.RandomState(seed)
    
    print('building datasets...')
    
    # Load CIFAR10 data
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])
    
    train_data_in_orig_cifar = dset.CIFAR10(root='./data', train=True, transform=transform, download=True)
    test_in_data_cifar = dset.CIFAR10(root='./data', train=False, transform=transform, download=True)
    
    # Split training data
    idx = np.array(range(len(train_data_in_orig_cifar)))
    rng.shuffle(idx)
    
    train_len = int(alpha * len(train_data_in_orig_cifar))
    train_idx = idx[:train_len]
    aux_idx = idx[int(0.5*len(train_data_in_orig_cifar)):]
    
    train_in_data = Subset(train_data_in_orig_cifar, train_idx)
    aux_in_data = Subset(train_data_in_orig_cifar, aux_idx)
    
    # Create corrupted versions of the datasets
    train_in_cor_data = CorruptedDataset(train_in_data, cortype)
    aux_in_cor_data = CorruptedDataset(aux_in_data, cortype)
    test_data_cor = CorruptedDataset(test_in_data_cifar, cortype)
    
    # Load out-of-distribution data (LSUN-C)
    out_data = dset.ImageFolder(root='../data/LSUN/',
                               transform=trn.Compose([trn.ToTensor(), 
                                                    trn.Normalize(mean, std),
                                                    trn.RandomCrop(32, padding=4)]))
    
    # Split OOD data
    idx_out = np.array(range(len(out_data)))
    rng.shuffle(idx_out)
    train_len_out = int(0.7 * len(out_data))
    aux_out_data = Subset(out_data, idx_out[:train_len_out])
    test_out_data = Subset(out_data, idx_out[train_len_out:])
    
    # Create validation splits
    aux_in_valid_size = int(0.3 * len(aux_in_data))
    valid_in_size = int(0.1 * len(aux_in_data))
    
    idx_in = np.array(range(len(aux_in_data)))
    rng.shuffle(idx_in)
    
    train_aux_in_idx = idx_in[aux_in_valid_size + valid_in_size:]
    valid_in_idx = idx_in[aux_in_valid_size:aux_in_valid_size + valid_in_size]
    
    train_aux_in_data_final = Subset(aux_in_data, train_aux_in_idx)
    valid_in_data = Subset(aux_in_data, valid_in_idx)
    
    # Create corrupted validation data
    valid_cor_data = CorruptedDataset(valid_in_data, cortype)
    
    # Create validation set for OOD data
    aux_out_valid_size = int(0.3 * len(aux_out_data))
    idx_out = np.array(range(len(aux_out_data)))
    rng.shuffle(idx_out)
    
    train_aux_out_idx = idx_out[aux_out_valid_size:]
    valid_out_idx = idx_out[:aux_out_valid_size]
    
    train_aux_out_data_final = Subset(aux_out_data, train_aux_out_idx)
    valid_out_data = Subset(aux_out_data, valid_out_idx)
    
    # Combine validation sets
    valid_aux_data = ConcatDataset([valid_in_data, valid_cor_data, valid_out_data])
    
    # Create dataloaders with matching parameters
    train_loader_in = DataLoader(
        train_in_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    train_loader_aux_in = DataLoader(
        train_aux_in_data_final,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    train_loader_aux_in_cor = DataLoader(
        train_in_cor_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    train_loader_aux_out = DataLoader(
        train_aux_out_data_final,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    valid_loader_in = DataLoader(
        valid_in_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    valid_loader_aux = DataLoader(
        valid_aux_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    test_loader_in = DataLoader(
        test_in_data_cifar,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader_cor = DataLoader(
        test_data_cor,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader_out = DataLoader(
        test_out_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return (train_loader_in, train_loader_aux_in, train_loader_aux_in_cor, train_loader_aux_out,
            test_loader_in, test_loader_cor, test_loader_out, valid_loader_in, valid_loader_aux) 
    

def load_Imagenette(dataset='imagenette', batch_size=128, num_workers=4, alpha=0.5, pi_1=0.5, pi_2=0.1, cortype='gaussian_noise', seed=42):
    """
    Load Imagenette dataset with corrupted data using on-the-fly corruptions.
    Matches the behavior of make_datasets.py for Imagenette.
    
    Args:
        dataset (str): Dataset name (only 'imagenette' supported)
        batch_size (int): Batch size for dataloaders
        num_workers (int): Number of workers for dataloaders
        alpha (float): Proportion of training data to use for training
        pi_1 (float): Proportion of validation data to use for validation
        pi_2 (float): Proportion of validation data to use for validation
        cortype (str): Type of corruption to apply
        seed (int): Random seed for reproducibility
        
    Returns:
        tuple: (train_loader_in, train_loader_aux_in, train_loader_aux_in_cor, train_loader_aux_out,
                test_loader_in, test_loader_cor, test_loader_out, valid_loader_in, valid_loader_aux)
    """
    if dataset != 'imagenette':
        raise ValueError("Only Imagenette is supported in this version")
        
    # Set random seed
    rng = np.random.RandomState(seed)
    
    print('building datasets...')
    
    # Load Imagenette data with appropriate transforms
    mean = [0.485, 0.456, 0.406]  # ImageNet statistics
    std = [0.229, 0.224, 0.225]
    
    # Transform to resize images to 32x32 to match CIFAR size
    transform = trn.Compose([
        trn.Resize((32, 32)),  # Force square resize
        trn.ToTensor(),
        trn.Normalize(mean, std)
    ])
    
    # Clean up existing dataset directory if it exists
    imagenette_dir = os.path.join('../data', 'imagenette2-160')
    if os.path.exists(imagenette_dir):
        print(f"Removing existing directory: {imagenette_dir}")
        import shutil
        shutil.rmtree(imagenette_dir)
        # Wait a moment to ensure directory is fully removed
        time.sleep(1)
    
    # Load both train and val datasets at once to avoid directory recreation issues
    print("Downloading and loading Imagenette dataset...")
    try:
        # Create a temporary dataset to trigger download
        temp_dataset = dset.Imagenette(
            root='../data',
            split='train',
            size='160px',
            download=True,
            transform=transform
        )
        del temp_dataset  # Clean up temporary dataset
        
        # Now load the actual datasets
        train_data_in_orig = dset.Imagenette(
            root='../data',
            split='train',
            size='160px',
            download=False,  # Don't download again
            transform=transform
        )
        
        test_in_data = dset.Imagenette(
            root='../data',
            split='val',
            size='160px',
            download=False,  # Don't download again
            transform=transform
        )
        
        print(f"Dataset loaded successfully. Training samples: {len(train_data_in_orig)}, Validation samples: {len(test_in_data)}")
        
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        raise
    
    # Split training data
    idx = np.array(range(len(train_data_in_orig)))
    rng.shuffle(idx)
    
    train_len = int(alpha * len(train_data_in_orig))
    train_idx = idx[:train_len]
    aux_idx = idx[int(0.5*len(train_data_in_orig)):]
    
    train_in_data = Subset(train_data_in_orig, train_idx)
    aux_in_data = Subset(train_data_in_orig, aux_idx)
    
    # Create corrupted versions of the datasets
    train_in_cor_data = CorruptedDataset(train_in_data, cortype)
    aux_in_cor_data = CorruptedDataset(aux_in_data, cortype)
    test_data_cor = CorruptedDataset(test_in_data, cortype)
    
    # Load out-of-distribution data (LSUN-C)
    out_data = dset.ImageFolder(
        root='../data/LSUN/',
        transform=trn.Compose([
            trn.Resize((32, 32)),  # Force square resize
            trn.ToTensor(),
            trn.Normalize(mean, std)
        ])
    )
    
    # Split OOD data
    idx_out = np.array(range(len(out_data)))
    rng.shuffle(idx_out)
    train_len_out = int(0.7 * len(out_data))
    aux_out_data = Subset(out_data, idx_out[:train_len_out])
    test_out_data = Subset(out_data, idx_out[train_len_out:])
    
    # Create validation splits
    aux_in_valid_size = int(0.3 * len(aux_in_data))
    valid_in_size = int(0.1 * len(aux_in_data))
    
    idx_in = np.array(range(len(aux_in_data)))
    rng.shuffle(idx_in)
    
    train_aux_in_idx = idx_in[aux_in_valid_size + valid_in_size:]
    valid_in_idx = idx_in[aux_in_valid_size:aux_in_valid_size + valid_in_size]
    
    train_aux_in_data_final = Subset(aux_in_data, train_aux_in_idx)
    valid_in_data = Subset(aux_in_data, valid_in_idx)
    
    # Create corrupted validation data
    valid_cor_data = CorruptedDataset(valid_in_data, cortype)
    
    # Create validation set for OOD data
    aux_out_valid_size = int(0.3 * len(aux_out_data))
    idx_out = np.array(range(len(aux_out_data)))
    rng.shuffle(idx_out)
    
    train_aux_out_idx = idx_out[aux_out_valid_size:]
    valid_out_idx = idx_out[:aux_out_valid_size]
    
    train_aux_out_data_final = Subset(aux_out_data, train_aux_out_idx)
    valid_out_data = Subset(aux_out_data, valid_out_idx)
    
    # Combine validation sets
    valid_aux_data = ConcatDataset([valid_in_data, valid_cor_data, valid_out_data])
    
    # Create dataloaders with matching parameters
    train_loader_in = DataLoader(
        train_in_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # Drop last incomplete batch
    )
    
    train_loader_aux_in = DataLoader(
        train_aux_in_data_final,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    train_loader_aux_in_cor = DataLoader(
        train_in_cor_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    train_loader_aux_out = DataLoader(
        train_aux_out_data_final,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    valid_loader_in = DataLoader(
        valid_in_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    valid_loader_aux = DataLoader(
        valid_aux_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    test_loader_in = DataLoader(
        test_in_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    test_loader_cor = DataLoader(
        test_data_cor,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    test_loader_out = DataLoader(
        test_out_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    return (train_loader_in, train_loader_aux_in, train_loader_aux_in_cor, train_loader_aux_out,
            test_loader_in, test_loader_cor, test_loader_out, valid_loader_in, valid_loader_aux) 

def load_cinic10(dataset='cinic10', batch_size=128, num_workers=4, alpha=0.5, pi_1=0.5, pi_2=0.1, cortype='gaussian_noise', seed=42):
    """
    Load CINIC-10 dataset with corrupted data using on-the-fly corruptions.
    Args:
        dataset (str): Dataset name (only 'cinic10' supported)
        batch_size (int): Batch size for dataloaders
        num_workers (int): Number of workers for dataloaders
        alpha (float): Proportion of training data to use for training
        pi_1 (float): Proportion of validation data to use for validation
        pi_2 (float): Proportion of validation data to use for validation
        cortype (str): Type of corruption to apply
        seed (int): Random seed for reproducibility
    Returns:
        tuple: (train_loader_in, train_loader_aux_in, train_loader_aux_in_cor, train_loader_aux_out,
                test_loader_in, test_loader_cor, test_loader_out, valid_loader_in, valid_loader_aux)
    """
    if dataset != 'cinic10':
        raise ValueError("Only cinic10 is supported in this version")
    
    rng = np.random.RandomState(seed)
    print('building CINIC-10 datasets...')
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    transform = trn.Compose([
        trn.Resize((32, 32)),
        trn.ToTensor(),
        trn.Normalize(mean, std)
    ])
    # Load datasets
    train_data_in_orig = dset.ImageFolder(root='../data/CINIC/train', transform=transform)
    valid_data_in_orig = dset.ImageFolder(root='../data/CINIC/valid', transform=transform)
    test_in_data = dset.ImageFolder(root='../data/CINIC/test', transform=transform)
    # Split training data
    idx = np.array(range(len(train_data_in_orig)))
    rng.shuffle(idx)
    train_len = int(alpha * len(train_data_in_orig))
    train_idx = idx[:train_len]
    aux_idx = idx[int(0.5*len(train_data_in_orig)):]
    train_in_data = Subset(train_data_in_orig, train_idx)
    aux_in_data = Subset(train_data_in_orig, aux_idx)
    # Create corrupted versions
    train_in_cor_data = CorruptedDataset(train_in_data, cortype)
    aux_in_cor_data = CorruptedDataset(aux_in_data, cortype)
    test_data_cor = CorruptedDataset(test_in_data, cortype)
    # For OOD, use LSUN-C as in other loaders
    out_data = dset.ImageFolder(root='../data/LSUN/',
                               transform=trn.Compose([
                                   trn.Resize((32, 32)),
                                   trn.ToTensor(),
                                   trn.Normalize(mean, std)
                               ]))
    idx_out = np.array(range(len(out_data)))
    rng.shuffle(idx_out)
    train_len_out = int(0.7 * len(out_data))
    aux_out_data = Subset(out_data, idx_out[:train_len_out])
    test_out_data = Subset(out_data, idx_out[train_len_out:])
    # Create validation splits
    aux_in_valid_size = int(0.3 * len(aux_in_data))
    valid_in_size = int(0.1 * len(aux_in_data))
    idx_in = np.array(range(len(aux_in_data)))
    rng.shuffle(idx_in)
    train_aux_in_idx = idx_in[aux_in_valid_size + valid_in_size:]
    valid_in_idx = idx_in[aux_in_valid_size:aux_in_valid_size + valid_in_size]
    train_aux_in_data_final = Subset(aux_in_data, train_aux_in_idx)
    valid_in_data = Subset(aux_in_data, valid_in_idx)
    valid_cor_data = CorruptedDataset(valid_in_data, cortype)
    aux_out_valid_size = int(0.3 * len(aux_out_data))
    idx_out = np.array(range(len(aux_out_data)))
    rng.shuffle(idx_out)
    train_aux_out_idx = idx_out[aux_out_valid_size:]
    valid_out_idx = idx_out[:aux_out_valid_size]
    train_aux_out_data_final = Subset(aux_out_data, train_aux_out_idx)
    valid_out_data = Subset(aux_out_data, valid_out_idx)
    valid_aux_data = ConcatDataset([valid_in_data, valid_cor_data, valid_out_data])
    # Dataloaders
    train_loader_in = DataLoader(
        train_in_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    train_loader_aux_in = DataLoader(
        train_aux_in_data_final,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    train_loader_aux_in_cor = DataLoader(
        train_in_cor_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    train_loader_aux_out = DataLoader(
        train_aux_out_data_final,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    valid_loader_in = DataLoader(
        valid_in_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    valid_loader_aux = DataLoader(
        valid_aux_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    test_loader_in = DataLoader(
        test_in_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    test_loader_cor = DataLoader(
        test_data_cor,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    test_loader_out = DataLoader(
        test_out_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    return (train_loader_in, train_loader_aux_in, train_loader_aux_in_cor, train_loader_aux_out,
            test_loader_in, test_loader_cor, test_loader_out, valid_loader_in, valid_loader_aux) 
    

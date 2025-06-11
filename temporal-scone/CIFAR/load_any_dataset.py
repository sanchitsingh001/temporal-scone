import torch
import torchvision.transforms as trn
import torchvision.datasets as dset
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
import os
import random
from make_datasets import train_valid_split

def load_any_dataset(dataset='cifar10', batch_size=128, num_workers=4, alpha=0.5, pi_1=0.3, pi_2=0.3, cortype='gaussian_noise', seed=42):
    """
    Load CIFAR10 dataset with corrupted data using pre-computed corruptions.
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
    
    # Load original CIFAR10
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])
    
    train_data = dset.CIFAR10(root='./data', train=True, transform=transform, download=True)
    test_data = dset.CIFAR10(root='./data', train=False, transform=transform, download=True)
    
    # Load corrupted CIFAR10
    from dataloader.corcifar10Loader import CorCIFARDataset
    train_cor_data = CorCIFARDataset('train', cortype, '../data/CorCIFAR10_train')
    test_cor_data = CorCIFARDataset('test', cortype, '../data/CorCIFAR10_test')
    
    # Split training data
    idx = np.array(range(len(train_data)))
    rng.shuffle(idx)
    train_len = int(alpha * len(train_data))
    train_idx = idx[:train_len]
    aux_idx = idx[int(0.5*len(train_data)):]
    
    train_in_data = torch.utils.data.Subset(train_data, train_idx)
    aux_in_data = torch.utils.data.Subset(train_data, aux_idx)
    
    # Split corrupted data
    idx_cor = np.array(range(len(train_cor_data)))
    rng.shuffle(idx_cor)
    train_len_cor = int(alpha * len(train_cor_data))
    train_idx_cor = idx_cor[:train_len_cor]
    aux_idx_cor = idx_cor[int(0.5*len(train_cor_data)):]
    
    train_in_cor_data = torch.utils.data.Subset(train_cor_data, train_idx_cor)
    aux_in_cor_data = torch.utils.data.Subset(train_cor_data, aux_idx_cor)
    
    # Create validation split
    # Since we don't have OOD data, we'll create a dummy dataset with the same length as aux_in_data
    dummy_out_data = torch.utils.data.Subset(train_data, aux_idx[:len(aux_idx)//10])  # Use a small subset as dummy
    
    # Get all validation splits
    test_in_data, test_in_data_cor, valid_in_data, valid_aux_data, train_aux_in_data_final, train_aux_in_data_cor_final, train_aux_out_data_final = train_valid_split(
        test_data, test_cor_data, aux_in_data, aux_in_cor_data, dummy_out_data, rng, pi_1, pi_2
    )
    
    # Create dataloaders
    train_loader_in = DataLoader(train_in_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    train_loader_aux_in = DataLoader(train_aux_in_data_final, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    train_loader_aux_in_cor = DataLoader(train_aux_in_data_cor_final, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    train_loader_aux_out = DataLoader(train_aux_out_data_final, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    test_loader_in = DataLoader(test_in_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader_cor = DataLoader(test_in_data_cor, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader_out = None  # No OOD data for testing
    
    valid_loader_in = DataLoader(valid_in_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    valid_loader_aux = DataLoader(valid_aux_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return (train_loader_in, train_loader_aux_in, train_loader_aux_in_cor, train_loader_aux_out,
            test_loader_in, test_loader_cor, test_loader_out, valid_loader_in, valid_loader_aux) 
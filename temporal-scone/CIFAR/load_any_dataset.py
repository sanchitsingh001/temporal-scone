import torch
import torchvision.transforms as trn
import torchvision.datasets as dset
from torch.utils.data import DataLoader, TensorDataset, Dataset, Subset, ConcatDataset
import numpy as np
import os
import random

def load_any_dataset(dataset='cifar10', batch_size=128, num_workers=4, alpha=0.5, pi_1=0.5, pi_2=0.1, cortype='gaussian_noise', seed=42):
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
    
    print('building datasets...')
    
    # Load CIFAR10 data
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])
    
    train_data_in_orig_cifar = dset.CIFAR10(root='./data', train=True, transform=transform, download=True)
    test_in_data_cifar = dset.CIFAR10(root='./data', train=False, transform=transform, download=True)
    
    # Load corrupted CIFAR10 data
    from dataloader.corcifar10Loader import CorCIFARDataset
    aux_data_cor_orig = CorCIFARDataset('train', cortype, '../data/CorCIFAR10_train')
    test_data_cor = CorCIFARDataset('test', cortype, '../data/CorCIFAR10_test')
    
    # Split training data
    idx = np.array(range(len(train_data_in_orig_cifar)))
    rng.shuffle(idx)
    
    train_len = int(alpha * len(train_data_in_orig_cifar))
    train_idx = idx[:train_len]
    aux_idx = idx[int(0.5*len(train_data_in_orig_cifar)):]
    
    train_in_data = Subset(train_data_in_orig_cifar, train_idx)
    aux_in_data = Subset(train_data_in_orig_cifar, aux_idx)
    
    # Split corrupted data
    idx_cor = np.array(range(len(aux_data_cor_orig)))
    rng.shuffle(idx_cor)
    train_len_cor = int(alpha * len(aux_data_cor_orig))
    train_idx_cor = idx_cor[:train_len_cor]
    aux_idx_cor = idx_cor[int(0.5*len(aux_data_cor_orig)):]
    
    train_in_cor_data = Subset(aux_data_cor_orig, train_idx_cor)
    aux_in_cor_data = Subset(aux_data_cor_orig, aux_idx_cor)
    
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
    
    # Split corrupted validation data
    aux_in_cor_valid_size = int(0.3 * len(aux_in_cor_data))
    valid_cor_size = int(0.1 * len(aux_in_cor_data))
    
    idx_cor = np.array(range(len(aux_in_cor_data)))
    rng.shuffle(idx_cor)
    
    train_aux_in_cor_idx = idx_cor[aux_in_cor_valid_size + valid_cor_size:]
    valid_cor_idx = idx_cor[aux_in_cor_valid_size:aux_in_cor_valid_size + valid_cor_size]
    
    train_aux_in_data_cor_final = Subset(aux_in_cor_data, train_aux_in_cor_idx)
    valid_cor_data = Subset(aux_in_cor_data, valid_cor_idx)
    
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
        train_aux_in_data_cor_final,
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
from make_datasets import *

# state = {
#     "batch_size": 128,
#     "prefetch": 2,
#     "seed": 42
# }

# city_loaders = make_shifted_batches('cifar10', T=3, batch_size=state['batch_size'], seed=state['seed'], num_workers=state['prefetch'])

# city_t, city_t1, city_t2 = city_loaders[0], city_loaders[1], city_loaders[2]

# print("Batches in each city split:")
# print(f"City t    : {len(city_loaders[0])} batches")
# print(f"City t+1  : {len(city_loaders[1])} batches")
# print(f"City t+2  : {len(city_loaders[2])} batches")

from torch.utils.data import Subset, DataLoader
import numpy as np

def split_loader_into_cities(dataloader, T=3, seed=42):
    rng = np.random.default_rng(seed)
    dataset = dataloader.dataset
    indices = np.arange(len(dataset))
    rng.shuffle(indices)
    city_splits = np.array_split(indices, T)
    return [
        DataLoader(
            Subset(dataset, split),
            batch_size=dataloader.batch_size,
            shuffle=True,  # or False depending on use
            num_workers=dataloader.num_workers,
            pin_memory=True,
            drop_last=False
        )
        for split in city_splits
    ]

loaders = make_datasets(in_dset='cifar10', aux_out_dset='lsun_c', test_out_dset='lsun_c', state ={'batch_size': 128, 'prefetch': 4, 'seed': 42}, alpha=0.5, pi_1=0.5, pi_2=0.1, cortype='gaussian_noise')

train_loader_in, train_loader_aux_in, train_loader_aux_in_cor, \
train_loader_aux_out, test_loader_in, test_loader_cor, \
test_loader_out, valid_loader_in, valid_loader_aux = loaders


T = 3  # number of city splits

train_loader_in_cities       = split_loader_into_cities(train_loader_in, T)
train_loader_aux_in_cities   = split_loader_into_cities(train_loader_aux_in, T)
train_loader_aux_cor_cities  = split_loader_into_cities(train_loader_aux_in_cor, T)
train_loader_aux_out_cities  = split_loader_into_cities(train_loader_aux_out, T)


print(len(train_loader_in_cities[0]) ) # city 0
print(len(train_loader_in_cities[1]))  # city 1
print(len(train_loader_in_cities[2]))  # city 2


loaders = zip(train_loader_in, train_loader_aux_in, train_loader_aux_in_cor, train_loader_aux_out)



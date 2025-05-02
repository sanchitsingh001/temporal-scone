from make_datasets import load_any_data, load_CIFAR, load_CorCifar, load_any_cor, load_in_mixCifar_data, load_in_mixAny_data, make_datasets, make_any_Dataset

import numpy as np

rng = np.random.default_rng(42)

y = make_any_Dataset(in_dset='../data/flowers10/', aux_out_dset='lsun_c', test_out_dset='lsun_c', state ={'batch_size': 128, 'prefetch': 4, 'seed': 42}, alpha=0.5, pi_1=0.5, pi_2=0.1, cortype='gaussian_noise')

print(type(y))


x= make_datasets(in_dset='cifar10', aux_out_dset='lsun_c', test_out_dset='lsun_c', state ={'batch_size': 128, 'prefetch': 4, 'seed': 42}, alpha=0.5, pi_1=0.5, pi_2=0.1, cortype='gaussian_noise')

print(type(x))





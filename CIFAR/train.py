# -*- coding: utf-8 -*-
from sklearn.metrics import det_curve, accuracy_score, roc_auc_score
from make_datasets import *
from models.wrn_ssnd import *
from torch.utils.data import DataLoader, Subset

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from models.mlp import *

import wandb

import seaborn as sns
import matplotlib.pyplot as plt

# for t-sne plot
from time import time
import pandas as pd

from matplotlib import offsetbox
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import seaborn as sns
import plotly.graph_objects as go


from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

import sklearn.metrics as sk

#import warning
#warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')

if __package__ is None:
    import sys
    from os import path

    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

'''
This code implements training and testing functions. 
'''


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

parser = argparse.ArgumentParser(description='Tunes a CIFAR Classifier with OE',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('dataset', type=str, choices=['cifar10', 'cifar100', 'MNIST','cinic10'],
                    default='MNIST', # default cifar10
                    help='Choose between CIFAR-10, CIFAR-100, MNIST.')
parser.add_argument('--model', '-m', type=str, default='mlp', # default allconv
                    choices=['allconv', 'wrn', 'densenet', 'mlp'], help='Choose architecture.')
# Optimization options
parser.add_argument('--epochs', '-e', type=int, default=10,
                    help='Number of epochs to train.')
parser.add_argument('--learning_rate', '-lr', type=float,
                    default=0.001, help='The initial learning rate.')
parser.add_argument('--batch_size', '-b', type=int,
                    default=128, help='Batch size.')
parser.add_argument('--oe_batch_size', type=int,
                    default=256, help='Batch size.')
parser.add_argument('--test_bs', type=int, default=200)
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-d', type=float,
                    default=0.0005, help='Weight decay (L2 penalty).')
# WRN Architecture
parser.add_argument('--layers', default=40, type=int,
                    help='total number of layers')
parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
parser.add_argument('--droprate', default=0.3, type=float,
                    help='dropout probability')
# Checkpoints
parser.add_argument('--results_dir', type=str,
                    default='results', help='Folder to save .pkl results.')
parser.add_argument('--checkpoints_dir', type=str,
                    default='checkpoints', help='Folder to save .pt checkpoints.')

parser.add_argument('--load_pretrained', type=str, default=None, help='Load pretrained model to test or resume training.')
#parser.add_argument('--load_pretrained', type=str,
#                    default='snapshots/pretrained', help='Load pretrained model to test or resume training.')
parser.add_argument('--test', '-t', action='store_true',
                    help='Test only flag.')

# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--gpu_id', type=int, default=1, help='Which GPU to run on.')
parser.add_argument('--prefetch', type=int, default=4,
                    help='Pre-fetching threads.')
# EG specific
parser.add_argument('--score', type=str, default='SSND', help='SSND|OE|energy|VOS')
parser.add_argument('--seed', type=int, default=1,
                    help='seed for np(tinyimages80M sampling); 1|2|8|100|107')
parser.add_argument('--classification', type=boolean_string, default=True)

# dataset related
parser.add_argument('--aux_out_dataset', type=str, default='FashionMNIST', choices=['svhn', 'lsun_c', 'lsun_r',
                    'isun', 'dtd', 'places', 'tinyimages_300k', 'FashionMNIST'],
                    help='Auxiliary out of distribution dataset')
parser.add_argument('--test_out_dataset', type=str, default='FashionMNIST', choices=['svhn', 'lsun_c', 'lsun_r',
                    'isun', 'dtd', 'places', 'tinyimages_300k', 'FashionMNIST'],
                    help='Test out of distribution dataset')
parser.add_argument('--pi_1', type=float, default=0.5,
                    help='pi in ssnd framework, proportion of ood data in auxiliary dataset')
parser.add_argument('--pi_2', type=float, default=0.5,
                    help='pi in ssnd framework, proportion of ood data in auxiliary dataset')

parser.add_argument('--pseudo', type=float, default=0.01,
                    help='pseudo regulazier')

parser.add_argument('--start_epoch', type=int, default=50,
                    help='start epoches')

parser.add_argument('--cortype', type=str, default='gaussian_noise', help='corrupted type of images')
###scone/woods/woods_nn specific
parser.add_argument('--in_constraint_weight', type=float, default=1,
                    help='weight for in-distribution penalty in loss function')
parser.add_argument('--out_constraint_weight', type=float, default=1,
                    help='weight for out-of-distribution penalty in loss function')
parser.add_argument('--ce_constraint_weight', type=float, default=1,
                    help='weight for classification penalty in loss function')
parser.add_argument('--false_alarm_cutoff', type=float,
                    default=0.05, help='false alarm cutoff')

parser.add_argument('--lr_lam', type=float, default=1, help='learning rate for the updating lam (SSND_alm)')
parser.add_argument('--ce_tol', type=float,
                    default=1.5, help='tolerance for the loss constraint')

parser.add_argument('--penalty_mult', type=float,
                    default=1.5, help='multiplicative factor for penalty method')

parser.add_argument('--constraint_tol', type=float,
                    default=0, help='tolerance for considering constraint violated')


parser.add_argument('--eta', type=float, default=1.0, help='woods with margin loss')

parser.add_argument('--alpha', type=float, default=0.05, help='number of labeled samples')

# Energy Method Specific
parser.add_argument('--m_in', type=float, default=-25.,
                    help='margin for in-distribution; above this value will be penalized')
parser.add_argument('--m_out', type=float, default=-5.,
                    help='margin for out-distribution; below this value will be penalized')
parser.add_argument('--T', default=1., type=float, help='temperature: energy|Odin')  # T = 1 suggested by energy paper

#energy vos method
parser.add_argument('--energy_vos_lambda', type=float, default=2, help='energy vos weight')

# OE specific
parser.add_argument('--oe_lambda', type=float, default=.5, help='OE weight')


# parse argument
args = parser.parse_args()

# method_data_name gives path to the model
if args.score in ['woods_nn']:
    method_data_name = "{}_{}_{}_{}_{}_{}_{}_{}_{}".format(args.score,
                                               str(args.in_constraint_weight),
                                               str(args.out_constraint_weight),
                                               str(args.ce_constraint_weight),
                                               str(args.false_alarm_cutoff),
                                               str(args.lr_lam),
                                               str(args.penalty_mult),
                                               str(args.pi_1),
                                               str(args.pi_2))
elif args.score == "energy":
    method_data_name = "{}_{}_{}_{}_{}".format(args.score,
                                      str(args.m_in),
                                      str(args.m_out),
                                      str(args.pi_1),
                                      str(args.pi_2))
elif args.score == "OE":
    method_data_name = "{}_{}_{}_{}".format(args.score,
                                   str(args.oe_lambda),
                                   str(args.pi_1),
                                   str(args.pi_2))
elif args.score == "energy_vos":
    method_data_name = "{}_{}_{}_{}".format(args.score,
                                   str(args.energy_vos_lambda),
                                   str(args.pi_1),
                                   str(args.pi_2))
elif args.score in ['scone', 'woods']:
    method_data_name = "{}_{}_{}_{}_{}_{}_{}_{}_{}".format(args.score,
                                            str(args.in_constraint_weight),
                                            str(args.out_constraint_weight),
                                            str(args.false_alarm_cutoff),
                                            str(args.ce_constraint_weight),
                                            str(args.lr_lam),
                                            str(args.penalty_mult),
                                            str(args.pi_1),
                                            str(args.pi_2))


state = {k: v for k, v in args._get_kwargs()}
print(state)

#save wandb hyperparameters
# wandb.config = state

wandb.init(project="oodproject", config=state)
state['wandb_name'] = wandb.run.name

# store train, test, and valid FPR95
state['fpr95_train'] = []
state['fpr95_valid'] = []
state['fpr95_valid_clean'] = []
state['fpr95_test'] = []

state['auroc_train'] = []
state['auroc_valid'] = []
state['auroc_valid_clean'] = []
state['auroc_test'] = []

state['val_wild_total'] = []
state['val_wild_class_as_in'] = []

# in-distribution classification accuracy
state['train_accuracy'] = []
state['valid_accuracy'] = []
state['valid_accuracy_clean'] = []
state['valid_accuracy_cor'] = []
state['valid_accuracy_clean_cor'] = []
state['test_accuracy'] = []
state['test_accuracy_cor'] = []

# store train, valid, and test OOD scores
state['OOD_scores_P0_train'] = []
state['OOD_scores_PX_train'] = []
state['OOD_scores_P0_valid'] = []
state['OOD_scores_PX_valid'] = []
state['OOD_scores_P0_valid_clean'] = []
state['OOD_scores_PX_valid_clean'] = []
state['OOD_scores_P0_test'] = []
state['OOD_scores_Ptest'] = []

# optimization constraints
state['in_dist_constraint'] = []
state['train_loss_constraint'] = []

state['atc_city_0'] = []
state['atc_city_1'] = []
state['atc_city_2'] = []
state['atc_diff_01'] = []
state['atc_per_epoch'] = []
device = torch.device("cuda" if torch.cuda.is_available() and args.ngpu > 0 else "cpu")
print("Using device:", device)


def split_loader_into_cities(dataset, batch_size=128, num_workers=4, T=3, seed=42):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(dataset))
    rng.shuffle(indices)
    city_splits = np.array_split(indices, T)
    return [
        DataLoader(
            Subset(dataset, split),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False
        )
        for split in city_splits
    ]

def make_mixed_city_loader(city_loader_in, city_loader_aux_in, city_loader_aux_cor, city_loader_aux_out, batch_size, pi_1, pi_2, seed=42):
    rng = np.random.default_rng(seed)
    all_data = []

    for (in_set, aux_in_set, aux_in_cor_set, aux_out_set) in zip(city_loader_in, city_loader_aux_in, city_loader_aux_cor, city_loader_aux_out):
        # Mix the batches
        clean_data = in_set[0]
        cor_data = aux_in_cor_set[0]
        out_data = aux_out_set[0]

        # Create masks
        n_total = clean_data.shape[0]
        n_clean = int(n_total * (1.0 - pi_1 - pi_2))
        n_cor = int(n_total * pi_1)
        n_out = int(n_total * pi_2)

        idx_clean = rng.choice(clean_data.shape[0], n_clean, replace=False)
        idx_cor = rng.choice(cor_data.shape[0], n_cor, replace=False)
        idx_out = rng.choice(out_data.shape[0], n_out, replace=False)

        # Concatenate
        mixed_batch = torch.cat([
            clean_data[idx_clean],
            cor_data[idx_cor],
            out_data[idx_out]
        ], dim=0)

        all_data.append(mixed_batch)

    mixed_dataset = torch.cat(all_data, dim=0)
    mixed_labels = torch.zeros(len(mixed_dataset))  # dummy labels (not used)

    mixed_loader = DataLoader(
        TensorDataset(mixed_dataset, mixed_labels),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )

    return mixed_loader



def to_np(x): return x.data.cpu().numpy()

torch.manual_seed(args.seed)
rng = np.random.default_rng(args.seed)





if args.dataset in ['cifar10','cinic10']:
    num_classes = 10
elif args.dataset in ['cifar100']:
    num_classes = 100
elif args.dataset in ['MNIST']:
    num_classes = 10  

# WRN architecture with 10 output classes (extra NN is added later for SSND methods)
net = WideResNet(args.layers, num_classes, args.widen_factor, dropRate=args.droprate)
# MLP architecture with num_classes
#net = woods_mlp(num_classes)


# create logistic regression layer for energy_vos and woods
if args.score in ['energy_vos', 'woods', 'scone']:
    logistic_regression = nn.Linear(1, 1)
    logistic_regression.cuda()


# Restore model
model_found = False
print(args.load_pretrained)
if args.load_pretrained == 'snapshots/pretrained':
    print('Restoring trained model...')
    for i in range(200, -1, -1):

        model_name = os.path.join(args.load_pretrained, args.dataset + '_' + args.model +
                                  '_pretrained_epoch_' + str(i) + '.pt')
        #model_name = "snapshots/pretrained/cinic10_wrn_pretrained_epoch_299.pt"
        #model_name = os.path.join(args.load_pretrained, 'in_ratio_50_cifar100_wrn_pretrained_epoch_' + str(i) + '.pt')
        if os.path.isfile(model_name):
            print('found pretrained model: {}'.format(model_name))
            net.load_state_dict(torch.load(model_name))
            print('Model restored! Epoch:', i)
            model_found = True
            break
    if not model_found:
        assert False, "could not find model to restore"



# add extra NN for OOD detection (for SSND methods)
if args.score in ['woods_nn']:
    net = WideResNet_SSND(wrn=net)
    #net = woods_mlp(num_classes)

if args.ngpu > 1:
    print('Available CUDA devices:', torch.cuda.device_count())
    print('CUDA available:', torch.cuda.is_available())
    print('Running in parallel across', args.ngpu, 'GPUs')
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))
    net.cuda()
    torch.cuda.manual_seed(1)
elif args.ngpu > 0:
    print('CUDA available:', torch.cuda.is_available())
    print('Available CUDA devices:', torch.cuda.device_count())
    print('Sending model to device', torch.cuda.current_device(), ':', torch.cuda.get_device_name())
    net.cuda()
    torch.cuda.manual_seed(1)

# cudnn.benchmark = True  # fire on all cylinders
cudnn.benchmark = False  # control reproducibility/stochastic behavior

#energy_vos, woods also use logistic regression in optimization
if args.score in ['energy_vos', 'woods', 'scone']:
    optimizer = torch.optim.SGD(
        list(net.parameters()) + list(logistic_regression.parameters()),
        state['learning_rate'], momentum=state['momentum'],
        weight_decay=state['decay'], nesterov=True)

else:
    optimizer = torch.optim.SGD(
        net.parameters(), state['learning_rate'], momentum=state['momentum'],
        weight_decay=state['decay'], nesterov=True)

#define scheduler for learning rate
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                 milestones=[int(args.epochs*.5), int(args.epochs*.75), int(args.epochs*.9)], gamma=0.5)

# /////////////// Training ///////////////

# Create extra variable needed for training

# make in_constraint a global variable
in_constraint_weight = args.in_constraint_weight

# make loss_ce_constraint a global variable
ce_constraint_weight = args.ce_constraint_weight

# create the lagrangian variable for lagrangian methods
if args.score in ['woods_nn', 'woods', 'scone']:
    lam = torch.tensor(0.0, device=device)
    lam = lam.cuda()

    lam2 = torch.tensor(0.0, device=device)
    lam2 = lam.cuda()


def mix_batches(aux_in_set, aux_in_cor_set, aux_out_set):
    in_data, _ = aux_in_set
    cor_data, _ = aux_in_cor_set
    out_data, _ = aux_out_set

    # Use actual batch size from current in_data
    batch_size = in_data.shape[0]

    # Safe probabilities
    pi_1 = args.pi_1
    pi_2 = args.pi_2
    pi_clean = 1.0 - pi_1 - pi_2

    # Calculate number of samples from each type
    n_clean = int(batch_size * pi_clean)
    n_cor = int(batch_size * pi_1)
    n_out = int(batch_size * pi_2)

    # Clamp to available data size
    n_clean = min(n_clean, in_data.shape[0])
    n_cor = min(n_cor, cor_data.shape[0])
    n_out = min(n_out, out_data.shape[0])

    # Subsample using random indices
    idx_clean = np.random.choice(in_data.shape[0], n_clean, replace=False)
    idx_cor = np.random.choice(cor_data.shape[0], n_cor, replace=False)
    idx_out = np.random.choice(out_data.shape[0], n_out, replace=False)

    aux_in_clean_sub = in_data[idx_clean]
    aux_in_cor_sub = cor_data[idx_cor]
    aux_out_sub = out_data[idx_out]

    aux_set = torch.cat([aux_in_clean_sub, aux_in_cor_sub, aux_out_sub], dim=0)

    return aux_set

def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out


def fpr_and_fdr_at_recall(y_true, y_score, recall_level, pos_label=1.):
    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))   # , fps[cutoff]/(fps[cutoff] + tps[cutoff])

def compute_entropy_atc(model, dataloader, delta=1.5, device='cuda'):
    model.eval()
    all_entropies = []
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=1)
            all_entropies.append(entropy)
    entropies = torch.cat(all_entropies)
    atc = (entropies < delta).float().mean().item()
    return atc

def compute_atc_max_softmax(model, dataloader, delta=0.9, device='cuda'):
    model.eval()
    all_confidences = []
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            max_conf = torch.max(probs, dim=1)[0]  # Take max prob per sample
            all_confidences.append(max_conf)
    confidences = torch.cat(all_confidences)
    atc = (confidences >= delta).float().mean().item()
    return atc

prev_atc_maxsoft = None
prev_fpr = None
temporal_loss = 0

def train(epoch, train_loader_in, train_loader_aux_in, train_loader_aux_in_cor, train_loader_aux_out, t):
        '''
        Train the model using the specified score
        '''

        # make the variables global for optimization purposes
        global in_constraint_weight
        global ce_constraint_weight

        # declare lam global
        if args.score in ['woods_nn',  'woods', 'scone']:
            global lam
            global lam2

        # print the learning rate
        for param_group in optimizer.param_groups:
            print("lr {}".format(param_group['lr']))

        net.train()  # enter train mode

        # track train classification accuracy
        train_accuracies = []

        # # start at a random point of the dataset for; this induces more randomness without obliterating locality
        train_loader_aux_in.dataset.offset = rng.integers(
            len(train_loader_aux_in.dataset))

        train_loader_aux_in_cor.dataset.offset = rng.integers(
            len(train_loader_aux_in_cor.dataset))   

        train_loader_aux_out.dataset.offset = rng.integers(
            len(train_loader_aux_out.dataset))
        batch_num = 1
        #loaders = zip(train_loader_in, train_loader_aux_in, train_loader_aux_out)
        loaders = zip(train_loader_in, train_loader_aux_in, train_loader_aux_in_cor, train_loader_aux_out)

        # for logging in weights & biases
        losses_ce = []
        in_losses = []
        out_losses = []
        out_losses_weighted = []
        losses = []

        for in_set, aux_in_set, aux_in_cor_set, aux_out_set in loaders:    
            #create the mixed batch
            aux_set = mix_batches(aux_in_set, aux_in_cor_set, aux_out_set)

            batch_num += 1
            data = torch.cat((in_set[0], aux_set), 0)
            target = in_set[1]

            if args.ngpu > 0:
                # data, target = data.to('cpu'), target.to('cpu')
                data, target = data.to(device), target.to(device)

            # forward
            x = net(data.to(device))

            # in-distribution classification accuracy
            if args.score in ['woods_nn']:
                x_classification = x[:len(in_set[0]), :num_classes]
            elif args.score in ['energy', 'OE', 'energy_vos', 'woods', 'scone']:
                x_classification = x[:len(in_set[0])]
            pred = x_classification.data.max(1)[1]
            train_accuracies.append(accuracy_score(list(to_np(pred)), list(to_np(target))))

            optimizer.zero_grad()
            
            # cross-entropy loss
            if args.classification:
                loss_ce = F.cross_entropy(x_classification, target)
            else:
                # Create a tensor with gradients enabled
                loss_ce = torch.tensor(0.0, device=device, requires_grad=True)
            losses_ce.append(loss_ce.item())

            if args.score == 'woods_nn':
                '''
                This is the same as woods_nn but it now uses separate
                weight for in distribution scores and classification scores.

                it also updates the weights separately
                '''

                # penalty for the mixture/auxiliary dataset
                out_x_ood_task = x[len(in_set[0]):, num_classes]
                out_loss = torch.mean(F.relu(1 - out_x_ood_task))
                out_loss_weighted = args.out_constraint_weight * out_loss

                in_x_ood_task = x[:len(in_set[0]), num_classes]
                f_term = torch.mean(F.relu(1 + in_x_ood_task)) - args.false_alarm_cutoff
                if in_constraint_weight * f_term + lam >= 0:
                    in_loss = f_term * lam + in_constraint_weight / 2 * torch.pow(f_term, 2)
                else:
                    in_loss = - torch.pow(lam, 2) * 0.5 / in_constraint_weight

                loss_ce_constraint = loss_ce - args.ce_tol * full_train_loss
                if ce_constraint_weight * loss_ce_constraint + lam2 >= 0:
                    loss_ce = loss_ce_constraint * lam2 + ce_constraint_weight / 2 * torch.pow(loss_ce_constraint, 2)
                else:
                    loss_ce = - torch.pow(lam2, 2) * 0.5 / ce_constraint_weight

                # add the losses together
                loss = loss_ce + out_loss_weighted + in_loss

                in_losses.append(in_loss.item())
                out_losses.append(out_loss.item())
                out_losses_weighted.append(out_loss.item())
                losses.append(loss.item())

            elif args.score == 'energy':

                Ec_out = -torch.logsumexp(x[len(in_set[0]):], dim=1)
                Ec_in = -torch.logsumexp(x[:len(in_set[0])], dim=1)
                loss_energy = 0.1 * (torch.pow(F.relu(Ec_in - args.m_in), 2).mean() + torch.pow(F.relu(args.m_out - Ec_out),
                                                                                            2).mean())
                loss = loss_ce + loss_energy

                losses.append(loss.item())

            elif args.score == 'energy_vos':

                Ec_out = torch.logsumexp(x[len(in_set[0]):], dim=1)
                Ec_in = torch.logsumexp(x[:len(in_set[0])], dim=1)
                binary_labels = torch.ones(len(x)).to('cpu')
                binary_labels[len(in_set[0]):] = 0
                loss_energy = F.binary_cross_entropy_with_logits(logistic_regression(
                    torch.cat([Ec_in, Ec_out], -1).unsqueeze(1)).squeeze(),
                                                                 binary_labels)

                loss = loss_ce + args.energy_vos_lambda * loss_energy

                losses.append(loss.item())

            elif args.score == 'scone':
                #apply the sigmoid loss
                loss_energy_in =  torch.mean(torch.sigmoid(logistic_regression(
                    (torch.logsumexp(x[:len(in_set[0])], dim=1)).unsqueeze(1)).squeeze()))
                loss_energy_out = torch.mean(torch.sigmoid(-logistic_regression(
                    (torch.logsumexp(x[len(in_set[0]):], dim=1) - args.eta).unsqueeze(1)).squeeze()))
    
                #alm function for the in distribution constraint
                in_constraint_term = loss_energy_in - args.false_alarm_cutoff
                if in_constraint_weight * in_constraint_term + lam >= 0:
                    in_loss = in_constraint_term * lam + in_constraint_weight / 2 * torch.pow(in_constraint_term, 2)
                else:
                    in_loss = - torch.pow(lam, 2) * 0.5 / in_constraint_weight
    
                #alm function for the cross entropy constraint
                loss_ce_constraint = loss_ce - args.ce_tol * full_train_loss
                if ce_constraint_weight * loss_ce_constraint + lam2 >= 0:
                    loss_ce = loss_ce_constraint * lam2 + ce_constraint_weight / 2 * torch.pow(loss_ce_constraint, 2)
                else:
                    loss_ce = - torch.pow(lam2, 2) * 0.5 / ce_constraint_weight
    
                loss = loss_ce + args.out_constraint_weight*loss_energy_out + in_loss 

            elif args.score == 'OE':

                loss_oe = args.oe_lambda * -(x[len(in_set[0]):].mean(1) - torch.logsumexp(x[len(in_set[0]):], dim=1)).mean()
                loss = loss_ce + loss_oe
                losses.append(loss.item())

            loss.backward()
            optimizer.step()

        loss_ce_avg = np.mean(losses_ce)
        in_loss_avg = np.mean(in_losses)
        out_loss_avg = np.mean(out_losses)
        out_loss_weighted_avg = np.mean(out_losses_weighted)
        loss_avg = np.mean(losses)
        train_acc_avg = np.mean(train_accuracies)

        wandb.log({
            'epoch':epoch,
            "learning rate": optimizer.param_groups[0]['lr'],
           f'CE loss {t}':loss_ce_avg,
            'in loss':in_loss_avg,
            'out loss':out_loss_avg,
            'out loss (weighted)':out_loss_weighted_avg,
            f'loss':loss_avg,
            'train accuracy':train_acc_avg
        }, step=epoch)

        # store train accuracy
        state['train_accuracy'].append(train_acc_avg)

        # updates for alm methods
        if args.score in ["woods_nn"]:
            print("making updates for SSND alm methods...")

            # compute terms for constraints
            in_term, ce_loss = compute_constraint_terms()

            # update lam for in-distribution term
            if args.score in ["woods_nn"]:
                print("updating lam...")

                in_term_constraint = in_term - args.false_alarm_cutoff
                print("in_distribution constraint value {}".format(in_term_constraint))
                state['in_dist_constraint'].append(in_term_constraint.item())

                # wandb
                wandb.log({
                    "in_term_constraint": in_term_constraint.item(),
                    'in_constraint_weight': in_constraint_weight,
                    'epoch': epoch
                }, step=epoch)

                # update lambda
                if in_term_constraint * in_constraint_weight + lam >= 0:
                    lam += args.lr_lam * in_term_constraint
                else:
                    lam += -args.lr_lam * lam / in_constraint_weight

            # update lam2
            if args.score in ["woods_nn"]:
                print("updating lam2...")

                ce_constraint = ce_loss - args.ce_tol * full_train_loss
                print("cross entropy constraint {}".format(ce_constraint))
                state['train_loss_constraint'].append(ce_constraint.item())

                # wandb
                wandb.log({
                    "ce_term_constraint": ce_constraint.item(),
                    'ce_constraint_weight': ce_constraint_weight,
                    'epoch': epoch
                }, step=epoch)

                # update lambda2
                if ce_constraint * ce_constraint_weight + lam2 >= 0:
                    lam2 += args.lr_lam * ce_constraint
                else:
                    lam2 += -args.lr_lam * lam2 / ce_constraint_weight

            # update weight for alm_full_2
            if args.score == 'woods_nn' and in_term_constraint > args.constraint_tol:
                print('increasing in_constraint_weight weight....\n')
                in_constraint_weight *= args.penalty_mult

            if args.score == 'woods_nn' and ce_constraint > args.constraint_tol:
                print('increasing ce_constraint_weight weight....\n')
                ce_constraint_weight *= args.penalty_mult

        #alm update for energy_vos alm methods
        if args.score in ['scone', 'woods']:
            print("making updates for energy alm methods...")
            avg_sigmoid_energy_losses, _, avg_ce_loss = evaluate_energy_logistic_loss(train_loader_in)

            in_term_constraint = avg_sigmoid_energy_losses -  args.false_alarm_cutoff
            print("in_distribution constraint value {}".format(in_term_constraint))
            state['in_dist_constraint'].append(in_term_constraint.item())

            # update lambda
            print("updating lam...")
            if in_term_constraint * in_constraint_weight + lam >= 0:
                lam += args.lr_lam * in_term_constraint
            else:
                lam += -args.lr_lam * lam / in_constraint_weight

            # wandb
            wandb.log({
                "in_term_constraint": in_term_constraint.item(),
                'in_constraint_weight': in_constraint_weight,
                "avg_sigmoid_energy_losses": avg_sigmoid_energy_losses.item(),
                'lam': lam,
                'epoch': epoch
            }, step=epoch)

            # update lam2
            if args.score in ['scone', 'woods']:
                print("updating lam2...")

                ce_constraint = avg_ce_loss - args.ce_tol * full_train_loss
                print("cross entropy constraint {}".format(ce_constraint))
                state['train_loss_constraint'].append(ce_constraint.item())

                # wandb
                wandb.log({
                    "ce_term_constraint": ce_constraint.item(),
                    'ce_constraint_weight': ce_constraint_weight,
                    'epoch': epoch
                }, step=epoch)

                # update lambda2
                if ce_constraint * ce_constraint_weight + lam2 >= 0:
                    lam2 += args.lr_lam * ce_constraint
                else:
                    lam2 += -args.lr_lam * lam2 / ce_constraint_weight

            # update in-distribution weight for alm
            if args.score in ['scone', 'woods'] and in_term_constraint > args.constraint_tol:
                print("energy in distribution constraint violated, so updating in_constraint_weight...")
                in_constraint_weight *= args.penalty_mult

            # update ce_loss weight for alm
            if args.score in ['scone', 'woods'] and ce_constraint > args.constraint_tol:
                print('increasing ce_constraint_weight weight....\n')
                ce_constraint_weight *= args.penalty_mult


def compute_constraint_terms(city_loader_in):
    '''

    Compute the in-distribution term and the cross-entropy loss over the whole training set
    '''

    net.eval()

    # create list for the in-distribution term and the ce_loss
    in_terms = []
    ce_losses = []
    num_batches = 0
    for in_set in city_loader_in:
        num_batches += 1
        data = in_set[0]
        target = in_set[1]

        if args.ngpu > 0:
            data, target = data.cuda(), target.cuda()

        # forward
        net(data)
        z = net(data)

        # compute in-distribution term
        in_x_ood_task = z[:, num_classes]
        in_terms.extend(list(to_np(F.relu(1 + in_x_ood_task))))

        # compute cross entropy term
        z_classification = z[:, :num_classes]
        loss_ce = F.cross_entropy(z_classification, target, reduction='none')
        ce_losses.extend(list(to_np(loss_ce)))

    return np.mean(np.array(in_terms)), np.mean(np.array(ce_losses))


def compute_fnr(out_scores, in_scores, fpr_cutoff=.05):
    '''
    compute fnr at 05
    '''

    in_labels = np.zeros(len(in_scores))
    out_labels = np.ones(len(out_scores))
    y_true = np.concatenate([in_labels, out_labels])
    y_score = np.concatenate([in_scores, out_scores])
    fpr, fnr, thresholds = det_curve(y_true=y_true, y_score=y_score)

    idx = np.argmin(np.abs(fpr - fpr_cutoff))

    fpr_at_fpr_cutoff = fpr[idx]
    fnr_at_fpr_cutoff = fnr[idx]

    if fpr_at_fpr_cutoff > 0.1:
        fnr_at_fpr_cutoff = 1.0

    return fnr_at_fpr_cutoff


def compute_auroc(out_scores, in_scores):
    in_labels = np.zeros(len(in_scores))
    out_labels = np.ones(len(out_scores))
    y_true = np.concatenate([in_labels, out_labels])
    y_score = np.concatenate([in_scores, out_scores])
    auroc = roc_auc_score(y_true=y_true, y_score=y_score)

    return auroc
def compute_average_confidence(model, dataloader, device='cuda'):
    """
    Calculate the Average Confidence (AC) score across a dataset.
    
    Args:
        model: The neural network model
        dataloader: DataLoader containing the samples
        device: Device to run computation on ('cuda' or 'cpu')
        
    Returns:
        float: Average confidence score across all samples
    """
    model.eval()
    all_confidences = []
    
    with torch.no_grad():
        for data, _ in dataloader:
            data = data.to(device)
            logits = model(data)
            probs = F.softmax(logits, dim=1)
            max_conf = torch.max(probs, dim=1)[0]  # Take max probability per sample
            all_confidences.append(max_conf)
    
    # Concatenate all confidences and compute mean
    confidences = torch.cat(all_confidences)
    ac = confidences.mean().item()
    
    return ac

# test function
def test(epoch,test_loader_in,
         test_loader_cor,
         test_loader_ood,
         valid_loader_in,
         valid_loader_aux):
    """
    tests current model
    """
    print('validation and testing...')

    net.eval()

    #Calculate AC for regular test data
    test_ac = compute_average_confidence(net, test_loader_in)
    print(f"\nTest Average Confidence: {test_ac:.3f}")
    
    # Calculate AC for corrupted test data
    test_ac_cor = compute_average_confidence(net, test_loader_cor)
    print(f"Test Average Confidence (Corrupted): {test_ac_cor:.3f}")

    # Log AC scores to wandb
    wandb.log({
        "test_ac": test_ac,
        "test_ac_cor": test_ac_cor,
        'epoch': epoch
    }, step=epoch)

    # in-distribution performance
    print("computing over test in-distribution data...\n")
    with torch.no_grad():
        accuracies = []
        OOD_scores_P0 = []
        for data, target in test_loader_in:
            if args.ngpu > 0:
                data, target = data.cuda(), target.cuda()
            # forward
            output = net(data)
            if args.score in ["woods_nn"]:
                # classification accuracy
                output_classification = output[:len(data), :num_classes]
                pred = output_classification.data.max(1)[1]
                accuracies.append(accuracy_score(list(to_np(pred)), list(to_np(target))))
                # OOD scores
                np_in = to_np(output[:, num_classes])
                np_in_list = list(np_in)
                OOD_scores_P0.extend(np_in_list)

            elif args.score in ['energy', 'OE', 'energy_vos', 'woods', 'scone']:
                # classification accuracy
                pred = output.data.max(1)[1]
                accuracies.append(accuracy_score(list(to_np(pred)), list(to_np(target))))

                if args.score in ['energy', 'energy_vos', 'woods', 'scone']:
                    # OOD scores
                    OOD_scores_P0.extend(list(-to_np((args.T * torch.logsumexp(output / args.T, dim=1)))))

                elif args.score == 'OE':
                    # OOD scores
                    smax = to_np(F.softmax(output, dim=1))
                    OOD_scores_P0.extend(list(-np.max(smax, axis=1)))

    # test covariate shift OOD distribution performance
    print("computing over test cor-distribution data...\n")
    with torch.no_grad():
        accuracies_cor = []
        OOD_scores_P_cor = []
        #for data, target in in_loader:
        for data, target in test_loader_cor:
            if args.ngpu > 0:
                data, target = data.cuda(), target.cuda()
            # forward
            output = net(data)
            if args.score in ["woods_nn"]:
                # classification accuracy
                output_classification = output[:len(data), :num_classes]
                pred = output_classification.data.max(1)[1]
                accuracies_cor.append(accuracy_score(list(to_np(pred)), list(to_np(target))))
                # OOD scores
                np_in = to_np(output[:, num_classes])
                np_in_list = list(np_in)
                OOD_scores_P_cor.extend(np_in_list)

            elif args.score in ['energy', 'OE', 'energy_vos', 'woods', 'scone']:
                # classification accuracy
                pred = output.data.max(1)[1]
                accuracies_cor.append(accuracy_score(list(to_np(pred)), list(to_np(target))))

                if args.score in ['energy', 'energy_vos', 'woods', 'scone']:
                    # OOD scores
                    OOD_scores_P_cor.extend(list(-to_np((args.T * torch.logsumexp(output / args.T, dim=1)))))

                elif args.score == 'OE':
                    # OOD scores
                    smax = to_np(F.softmax(output, dim=1))
                    OOD_scores_P_cor.extend(list(-np.max(smax, axis=1)))


    # semantic shift OOD distribution performance
    print("computing over test OOD-distribution data...\n")
    with torch.no_grad():
        OOD_scores_P_out = []
        for data, target in test_loader_ood:
            if args.ngpu > 0:
                data, target = data.cuda(), target.cuda()
            # forward
            output = net(data)
            if args.score in ["woods_nn"]:
                # classification accuracy
                output_classification = output[:len(data), :num_classes]
                pred = output_classification.data.max(1)[1]
                # OOD scores
                np_in = to_np(output[:, num_classes])
                np_in_list = list(np_in)
                OOD_scores_P_out.extend(np_in_list)

            elif args.score in ['energy', 'OE', 'energy_vos', 'woods', 'scone']:
                # classification accuracy
                pred = output.data.max(1)[1]

                if args.score in ['energy', 'energy_vos', 'woods', 'scone']:
                    # OOD scores
                    OOD_scores_P_out.extend(list(-to_np((args.T * torch.logsumexp(output / args.T, dim=1)))))

                elif args.score == 'OE':
                    # OOD scores
                    smax = to_np(F.softmax(output, dim=1))
                    OOD_scores_P_out.extend(list(-np.max(smax, axis=1)))

    # valid in-distribution performance
    print("computing over valid in-distribution data...\n")
    with torch.no_grad():
        accuracies_val = []
        OOD_scores_val_P0 = []
        for data, target in valid_loader_in:
            if args.ngpu > 0:
                data, target = data.cuda(), target.cuda()
            # forward
            output = net(data)
            if args.score in ["woods_nn"]:
                # classification accuracy
                output_classification = output[:len(data), :num_classes]
                pred = output_classification.data.max(1)[1]
                accuracies_val.append(accuracy_score(list(to_np(pred)), list(to_np(target))))
                # OOD scores
                np_in = to_np(output[:, num_classes])
                np_in_list = list(np_in)
                OOD_scores_val_P0.extend(np_in_list)

            elif args.score in ['energy', 'OE', 'energy_vos', 'woods', 'scone']:
                # classification accuracy
                pred = output.data.max(1)[1]
                accuracies_val.append(accuracy_score(list(to_np(pred)), list(to_np(target))))

                if args.score in ['energy', 'energy_vos', 'woods', 'scone']:
                    # OOD scores
                    OOD_scores_val_P0.extend(list(-to_np((args.T * torch.logsumexp(output / args.T, dim=1)))))

                elif args.score == 'OE':
                    # OOD scores
                    smax = to_np(F.softmax(output, dim=1))
                    OOD_scores_val_P0.extend(list(-np.max(smax, axis=1)))

    # valid wild-distribution performance
    print("computing over valid wild-distribution data...\n")
    with torch.no_grad():
        OOD_scores_val_P_wild = []
        for data, target in valid_loader_aux:
            if args.ngpu > 0:
                data, target = data.cuda(), target.cuda()
            # forward
            output = net(data)
            if args.score in ["woods_nn"]:
                # classification accuracy
                output_classification = output[:len(data), :num_classes]
                pred = output_classification.data.max(1)[1]
                # OOD scores
                np_in = to_np(output[:, num_classes])
                np_in_list = list(np_in)
                OOD_scores_val_P_wild.extend(np_in_list)

            elif args.score in ['energy', 'OE', 'energy_vos', 'woods', 'scone']:
                # classification accuracy
                pred = output.data.max(1)[1]

                if args.score in ['energy', 'energy_vos', 'woods', 'scone']:
                    # OOD scores
                    OOD_scores_val_P_wild.extend(list(-to_np((args.T * torch.logsumexp(output / args.T, dim=1)))))

                elif args.score == 'OE':
                    # OOD scores
                    smax = to_np(F.softmax(output, dim=1))
                    OOD_scores_val_P_wild.extend(list(-np.max(smax, axis=1)))

    in_scores = np.array(OOD_scores_P0)

    in_scores.sort()
    threshold_idx = int(len(in_scores)*0.95)
    threshold = in_scores[threshold_idx]

    val_in = np.array(OOD_scores_val_P0)
    val_wild = np.array(OOD_scores_val_P_wild)

    val_wild_total = len(val_wild)
    val_wild_class_as_in = np.sum(val_wild < threshold)

    print("\n validation wild total {}".format(val_wild_total))
    print("\n validation wild classify as in {}".format(val_wild_class_as_in))

    # compute FPR95 and accuracy
    fpr95 = compute_fnr(np.array(OOD_scores_P_out), np.array(OOD_scores_P0))
    auroc = compute_auroc(np.array(OOD_scores_P_out), np.array(OOD_scores_P0))
    
    acc = sum(accuracies) / len(accuracies)

    acc_cor = sum(accuracies_cor) / len(accuracies_cor)

    # store and print result
    state['fpr95_test'].append(fpr95)
    state['auroc_test'].append(auroc)
    state['test_accuracy'].append(acc)
    state['test_accuracy_cor'].append(acc_cor)
    state['OOD_scores_P0_test'].append(OOD_scores_P0)
    state['OOD_scores_Ptest'].append(OOD_scores_P_out)
    state['val_wild_total'].append(val_wild_total)
    state['val_wild_class_as_in'].append(val_wild_class_as_in)

    wandb.log({"fpr95_test": fpr95,
            "auroc_test": auroc,
            "test_accuracy": acc,
            "test_accuracy_cor":acc_cor,
            "val_wild_total": val_wild_total,
            "val_wild_class_as_in": val_wild_class_as_in,
            'epoch':epoch}, step=epoch)

    print("\n fpr95_test {}".format(state['fpr95_test']))
    print("\n auroc_test {}".format(state['auroc_test']))
    print("test_accuracy {} \n".format(state['test_accuracy']))
    print("test_accuracy_cor {} \n".format(state['test_accuracy_cor']))
    print("val_wild_total {} \n".format(state['val_wild_total']))
    print("val_wild_class_as_in {} \n".format(state['val_wild_class_as_in']))


def evaluate_classification_loss_training(city_loader_in):
    '''
    evaluate classification loss on training dataset
    '''

    net.eval()
    losses = []
    for in_set in city_loader_in:
        data = in_set[0]
        target = in_set[1]

        if args.ngpu > 0:
            data, target = data.cuda(), target.cuda()
        # forward
        x = net(data)

        # in-distribution classification accuracy
        x_classification = x[:, :num_classes]
        loss_ce = F.cross_entropy(x_classification, target, reduction='none')

        losses.extend(list(to_np(loss_ce)))

    avg_loss = np.mean(np.array(losses))
    print("average loss fr classification {}".format(avg_loss))

    return avg_loss


def evaluate_energy_logistic_loss(city_loader_in):
    '''
    evaluate energy logistic loss on training dataset
    '''

    net.eval()
    sigmoid_energy_losses = []
    logistic_energy_losses = []
    ce_losses = []
    for in_set in city_loader_in:
        data = in_set[0]
        target = in_set[1]

        if args.ngpu > 0:
            data, target = data.cuda(), target.cuda()

        # forward
        x = net(data)

        # compute energies
        Ec_in = torch.logsumexp(x, dim=1)

        # compute labels
        binary_labels_1 = torch.ones(len(data)).cuda()

        # compute in distribution logistic losses
        logistic_loss_energy_in = F.binary_cross_entropy_with_logits(logistic_regression(
            Ec_in.unsqueeze(1)).squeeze(), binary_labels_1, reduction='none')

        logistic_energy_losses.extend(list(to_np(logistic_loss_energy_in)))

        # compute in distribution sigmoid losses
        sigmoid_loss_energy_in = torch.sigmoid(logistic_regression(
            Ec_in.unsqueeze(1)).squeeze())

        sigmoid_energy_losses.extend(list(to_np(sigmoid_loss_energy_in)))

        # in-distribution classification losses
        x_classification = x[:, :num_classes]
        loss_ce = F.cross_entropy(x_classification, target, reduction='none')

        ce_losses.extend(list(to_np(loss_ce)))

    avg_sigmoid_energy_losses = np.mean(np.array(sigmoid_energy_losses))
    print("average sigmoid in distribution energy loss {}".format(avg_sigmoid_energy_losses))

    avg_logistic_energy_losses = np.mean(np.array(logistic_energy_losses))
    print("average in distribution energy loss {}".format(avg_logistic_energy_losses))

    avg_ce_loss = np.mean(np.array(ce_losses))
    print("average loss fr classification {}".format(avg_ce_loss))

    return avg_sigmoid_energy_losses, avg_logistic_energy_losses, avg_ce_loss

print('Beginning Training\n')

#compute training loss for scone/woods methods

###################################################################
# Main loop
###################################################################

from torch.utils.data import Subset, DataLoader
import numpy as np


from load_any_dataset import load_cifar, load_Imagenette, load_cinic10
# city_0_loaders =  make_datasets(in_dset='cifar10', aux_out_dset='lsun_c', test_out_dset='lsun_c', state ={'batch_size': 128, 'prefetch': 4, 'seed': 42}, alpha=0.5, pi_1=0.5, pi_2=0.1, cortype='gaussian_noise')
city_0_loaders = load_cifar()
city_1_loaders = load_Imagenette()  # CINIC
city_2_loaders =load_cinic10()


T = 3  # number of city splits



# def compute_temporal_loss_and_log(global_epoch, city_mixed_loader, state, prev_atc, prev_fpr):
#     delta = 0.05
#     gamma = 0.05
#     lambda_temp = 1.0

#     current_atc = compute_atc_max_softmax(net, city_mixed_loader, delta=0.9)

#     if len(state['OOD_scores_Ptest']) > 0 and len(state['OOD_scores_P0_test']) > 0:
#         current_fpr = compute_fnr(np.array(state['OOD_scores_Ptest'][-1]), np.array(state['OOD_scores_P0_test'][-1]))
#     else:
#         current_fpr = 0.0

#     if prev_atc is not None and prev_fpr is not None:
#         atc_diff = abs(current_atc - prev_atc)
#         fpr_diff = abs(current_fpr - prev_fpr)
#         temporal_penalty = max(0.0, atc_diff - delta) + max(0.0, fpr_diff - gamma)
#         temporal_loss = lambda_temp * temporal_penalty
#     else:
#         atc_diff = 0.0
#         fpr_diff = 0.0
#         temporal_loss = 0.0

#     wandb.log({
#         "temporal_loss": temporal_loss,
#         "atc_maxsoft_change": atc_diff,
#         "fpr_change": fpr_diff,
#         "epoch": global_epoch
#     })

#     return current_atc, current_fpr, temporal_loss


import time
T = 3  # number of cities
total_epochs_per_city = args.epochs


city_loaders = [
    city_0_loaders,
    city_1_loaders,
    city_2_loaders,
]
from torch.utils.data import Subset
import random

def subsample_loader(loader, target_size, seed=42):
    random.seed(seed)
    dataset = loader.dataset
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    selected_indices = indices[:target_size]
    subset = Subset(dataset, selected_indices)
    new_loader = torch.utils.data.DataLoader(
        subset,
        batch_size=loader.batch_size,
        shuffle=loader.shuffle if hasattr(loader, 'shuffle') else False,
        num_workers=loader.num_workers,
        pin_memory=True
    )
    return new_loader

def get_min_dataset_size(city_loaders):
    min_size = float('inf')
    for loaders in city_loaders:
        train_loader_in = loaders[0]  # First loader is train_loader_in
        dataset_size = len(train_loader_in.dataset)
        min_size = min(min_size, dataset_size)
    return min_size

# Get the minimum size across all cities
target_size = get_min_dataset_size(city_loaders)
print(f"Target dataset size for all cities: {target_size}")
for t in range(T):
    print(f"\n=======================\nTraining on City {t}\n=======================")
    train_loader_in, train_loader_aux_in, train_loader_aux_in_cor, \
    train_loader_aux_out, test_loader_in, test_loader_cor, \
    test_loader_out, valid_loader_in, valid_loader_aux = city_loaders[t]
    print(city_loaders[t])
    train_loader_in = subsample_loader(train_loader_in, target_size)
    train_loader_aux_in = subsample_loader(train_loader_aux_in, target_size)
    train_loader_aux_in_cor = subsample_loader(train_loader_aux_in_cor, target_size)
    train_loader_aux_out = subsample_loader(train_loader_aux_out, target_size)
    test_loader_in = subsample_loader(test_loader_in, target_size)
    test_loader_cor = subsample_loader(test_loader_cor, target_size)
    test_loader_out = subsample_loader(test_loader_out, target_size)
    valid_loader_in = subsample_loader(valid_loader_in, target_size)
    valid_loader_aux = subsample_loader(valid_loader_aux, target_size)
    # Adjust learning rate based on city
    if t == 1 or t == 2:  # City1 (Imagenette)
        new_lr = state['learning_rate'] * 5.0  # 5x higher learning rate for Imagenette
    else:
        new_lr = state['learning_rate']  # Default learning rate for other cities

    # Update learning rate for all parameter groups
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

    print(f"Updated learning rate to {new_lr} for City {t}")

    if args.score in [ 'woods_nn', 'woods', 'scone']:
        full_train_loss = evaluate_classification_loss_training(train_loader_in)

    
    for epoch in range(total_epochs_per_city):
        global_epoch = t * total_epochs_per_city + epoch
        print('epoch', global_epoch + 1, '/', total_epochs_per_city * T)
        state['epoch'] = global_epoch
#         prev_atc_maxsoft, prev_fpr, temporal_loss = compute_temporal_loss_and_log(
#      global_epoch, city_mixed_loader, state, prev_atc_maxsoft, prev_fpr
# )
        train(global_epoch,
              train_loader_in,
              train_loader_aux_in,
              train_loader_aux_in_cor,
              train_loader_aux_out,t)

        test(global_epoch,test_loader_in,
         test_loader_cor,
         test_loader_out,
         valid_loader_in,
         valid_loader_aux)
       

        scheduler.step()
        # prev_atc_maxsoft = compute_atc_max_softmax(net, city_mixed_loader, delta=0.9)
        # if len(state['OOD_scores_Ptest']) > 0 and len(state['OOD_scores_P0_test']) > 0:
        #     prev_fpr = compute_fnr(
        #         np.array(state['OOD_scores_Ptest'][-1]),
        #         np.array(state['OOD_scores_P0_test'][-1])
        #     )
        # else:
        #     prev_fpr = 0.0
        
        

state['best_epoch_valid'] = epoch

state['test_fpr95_at_best'] = state['fpr95_test'][-1]
state['test_auroc_at_best'] = state['auroc_test'][-1]
state['test_accuracy_at_best'] = state['test_accuracy'][-1]
state['test_accuracy_cor_at_best'] = state['test_accuracy_cor'][-1]
state['val_wild_total_at_best'] = state['val_wild_total'][-1]
state['val_wild_class_as_in_at_best'] = state['val_wild_class_as_in'][-1]

print('best epoch = {}'.format(state['best_epoch_valid']))

wandb.log({"best_epoch_valid": state['best_epoch_valid'],
            "test_fpr95_at_best": state['test_fpr95_at_best'],
            "test_auroc_at_best": state['test_auroc_at_best'],
            "test_accuracy_at_best": state['test_accuracy_at_best'],
            "test_accuracy_cor_at_best": state['test_accuracy_cor_at_best'],
            "val_wild_total_at_best": state['val_wild_total_at_best'],
            "val_wild_class_as_in_at_best": state['val_wild_class_as_in_at_best']
            })
print("\n=== Comparing ATC across cities ===")

delta = 1.5  # same delta threshold

# # Recompute final ATC for each city
# atc_city_0 = compute_entropy_atc(net, city_mixed_loader_0, delta)
# atc_city_1 = compute_entropy_atc(net, city_mixed_loader_1, delta)
# atc_city_2 = compute_entropy_atc(net, city_mixed_loader_2, delta)

# print(f"ATC City 0: {atc_city_0:.3f}")
# print(f"ATC City 1: {atc_city_1:.3f}")
# print(f"ATC City 2: {atc_city_2:.3f}")

# # Compute differences
# atc_diff_01 = abs(atc_city_0 - atc_city_1)
# atc_diff_12 = abs(atc_city_1 - atc_city_2)
# atc_diff_02 = abs(atc_city_0 - atc_city_2)

# print(f"ATC Diff (City 0 vs 1): {atc_diff_01:.3f}")
# print(f"ATC Diff (City 1 vs 2): {atc_diff_12:.3f}")
# print(f"ATC Diff (City 0 vs 2): {atc_diff_02:.3f}")

# # Save ATC results into wandb or state
# wandb.log({
#     "final_atc_city_0": atc_city_0,
#     "final_atc_city_1": atc_city_1,
#     "final_atc_city_2": atc_city_2,
#     "final_atc_diff_01": atc_diff_01,
#     "final_atc_diff_12": atc_diff_12,
#     "final_atc_diff_02": atc_diff_02,
# })

# save model checkpoint
#args.checkpoints_dir = './checkpoints/save/'
if args.checkpoints_dir != '':
    model_checkpoint_dir = os.path.join(args.checkpoints_dir,
                                        args.dataset,
                                        args.aux_out_dataset,
                                        args.score)
    if not os.path.exists(model_checkpoint_dir):
        os.makedirs(model_checkpoint_dir, exist_ok=True)
    model_filename = '{}_epoch_{}.pt'.format(method_data_name, epoch)
    model_path = os.path.join(model_checkpoint_dir, model_filename)
    print('saving model to {}'.format(model_path))
    torch.save(net.state_dict(), model_path)

# save results to .pkl file
results_dir = os.path.join(args.results_dir,
                            args.dataset,
                            args.aux_out_dataset,
                            args.score)
if not os.path.exists(results_dir):
    os.makedirs(results_dir, exist_ok=True)
results_filename = '{}.pkl'.format(method_data_name)
results_path = os.path.join(results_dir, results_filename)
with open(results_path, 'wb') as f:
    print('saving results to', results_path)
    pickle.dump(state, f)


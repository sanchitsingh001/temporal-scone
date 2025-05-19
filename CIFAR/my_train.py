# -*- coding: utf-8 -*-
from sklearn.metrics import det_curve, accuracy_score, roc_auc_score
from make_datasets import *
from models.wrn_ssnd import *

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




def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

parser = argparse.ArgumentParser(description='Tunes a CIFAR Classifier with OE',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('dataset', type=str, choices=['cifar10', 'cifar100', 'MNIST'],
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
                    default=2, help='tolerance for the loss constraint')

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

def to_np(x): return x.data.cpu().numpy()

torch.manual_seed(args.seed)
rng = np.random.default_rng(args.seed)

#make the data_loaders
train_loader_in, train_loader_aux_in, train_loader_aux_in_cor, train_loader_aux_out, test_loader_in, test_loader_cor, test_loader_ood, valid_loader_in, valid_loader_aux = make_datasets(args.dataset, args.aux_out_dataset, args.test_out_dataset, state, args.alpha, args.pi_1, args.pi_2, args.cortype)




print("\n len(train_loader_in.dataset) {} \n" \
      "len(train_loader_aux_in.dataset) {}, \n" \
      "len(train_loader_aux_in_cor.dataset) {},\n"\
      "len(train_loader_aux_out.dataset) {}, \n" \
      "len(test_loader_mnist.dataset) {}, \n" \
      "len(test_loader_cor.dataset) {}, \n" \
      "len(test_loader_ood.dataset) {}, \n" \
      "len(valid_loader_in.dataset) {}, \n" \
      "len(valid_loader_aux.dataset) {}\n".format(
    len(train_loader_in.dataset),
    len(train_loader_aux_in.dataset),
    len(train_loader_aux_in_cor.dataset),
    len(train_loader_aux_out.dataset),
    len(test_loader_in.dataset),
    len(test_loader_cor.dataset),
    len(test_loader_ood.dataset),
    len(valid_loader_in.dataset),
    len(valid_loader_aux.dataset)))



state['train_in_size'] = len(train_loader_in.dataset)
state['train_aux_in_size'] = len(train_loader_aux_in.dataset)
state['train_aux_out_size'] = len(train_loader_aux_out.dataset)
state['valid_in_size'] = len(valid_loader_in.dataset)
state['valid_aux_size'] = len(valid_loader_aux.dataset)
state['test_in_size'] = len(test_loader_in.dataset)
state['test_in_cor_size'] = len(test_loader_cor.dataset)
state['test_out_size'] = len(test_loader_ood.dataset)

if args.dataset in ['cifar10']:
    num_classes = 10
elif args.dataset in ['cifar100']:
    num_classes = 100
elif args.dataset in ['MNIST']:
    num_classes = 10  



# WRN architecture with 10 output classes (extra NN is added later for SSND methods)
net =  WideResNet(depth=40, num_classes=10, widen_factor=2, dropRate=0.3).cuda()
# MLP architecture with num_classes
#net = woods_mlp(num_classes)


# create logistic regression layer for energy_vos and woods
if args.score in ['energy_vos', 'woods', 'scone']:
    logistic_regression = nn.Linear(1, 1)
    logistic_regression.cuda()

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
    lam = torch.tensor(0).float()
    lam = lam.cuda()

    lam2 = torch.tensor(0).float()
    lam2 = lam.cuda()

import torch.nn.functional as F


def mix_batches(aux_in_set, aux_in_cor_set, aux_out_set):
    '''
    Args:
        aux_in_set: minibatch from in_distribution
        aux_in_cor_set: minibatch from covariate shift OOD distribution
        aux_out_set: minibatch from semantic shift OOD distribution

    Returns:
        mixture of minibatches with mixture proportion pi_1 of aux_in_cor_set and pi_2 of aux_out_set
    '''
    
    # create a mask to decide which sample is in the batch
    mask_1 = rng.choice(a=[False, True], size=(args.batch_size,), p=[1 - args.pi_1, args.pi_1])
    aux_in_cor_set_subsampled = aux_in_cor_set[0][mask_1]

    mask_2 = rng.choice(a=[False, True], size=(args.batch_size,), p=[1 - args.pi_2, args.pi_2])
    aux_out_set_subsampled = aux_out_set[0][mask_2]

    mask_12 = rng.choice(a=[False, True], size=(args.batch_size,), p=[1 - (args.pi_1 + args.pi_2), (args.pi_1 + args.pi_2)])
    #mask = rng.choice(a=[False, True], size=(args.batch_size,), p=[1 - 0.05, 0.05])
    aux_in_set_subsampled = aux_in_set[0][np.invert(mask_12)]

    # note: ordering of aux_out_set_subsampled, aux_in_set_subsampled does not matter because you always take the sum
    aux_set = torch.cat((aux_out_set_subsampled, aux_in_set_subsampled, aux_in_cor_set_subsampled), 0)

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


def train(epoch):
    print(f"\n==> Training Epoch {epoch + 1}/{args.epochs}...")

    net.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader_in):
        inputs, targets = inputs.cuda(), targets.cuda()

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss / len(train_loader_in)
    acc = 100. * correct / total
    print(f"Epoch [{epoch+1}] | Loss: {avg_loss:.4f} | Accuracy: {acc:.2f}%")

    # Save stats
    state['train_accuracy'].append(acc)

    # Step the scheduler
    scheduler.step()

    # Log to wandb
    wandb.log({
        'epoch': epoch + 1,
        'train_loss': avg_loss,
        'train_accuracy': acc,
        'learning_rate': scheduler.get_last_lr()[0]
    })

def train_scone(epoch):
    global lam, lam2, in_constraint_weight, ce_constraint_weight

    print(f"\n==> [SCONE] Training Epoch {epoch + 1}/{args.epochs}...")
    net.train()

    train_accuracies, in_losses, out_losses, losses_ce = [], [], [], []

    # induce randomness
    train_loader_aux_in.dataset.offset = rng.integers(len(train_loader_aux_in.dataset))
    train_loader_aux_in_cor.dataset.offset = rng.integers(len(train_loader_aux_in_cor.dataset))
    train_loader_aux_out.dataset.offset = rng.integers(len(train_loader_aux_out.dataset))

    batch_num = 1
    loaders = zip(train_loader_in, train_loader_aux_in, train_loader_aux_in_cor, train_loader_aux_out)

    for in_set, aux_in_set, aux_in_cor_set, aux_out_set in loaders:
        aux_set = mix_batches(aux_in_set, aux_in_cor_set, aux_out_set)
        data = torch.cat((in_set[0], aux_set), 0)
        target = in_set[1]

        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        x = net(data)

        x_classification = x[:len(in_set[0])]
        pred = x_classification.data.max(1)[1]
        train_accuracies.append(accuracy_score(list(to_np(pred)), list(to_np(target))))

        loss_ce = F.cross_entropy(x_classification, target)

        # === SCONE losses ===
        energy_in = torch.logsumexp(x[:len(in_set[0])], dim=1)
        energy_out = torch.logsumexp(x[len(in_set[0]):], dim=1)

        loss_energy_in = torch.mean(torch.sigmoid(logistic_regression(energy_in.unsqueeze(1)).squeeze()))
        loss_energy_out = torch.mean(torch.sigmoid(-logistic_regression((energy_out - args.eta).unsqueeze(1)).squeeze()))

        in_constraint_term = loss_energy_in - args.false_alarm_cutoff
        ce_constraint_term = loss_ce - args.ce_tol * full_train_loss

        # augmented Lagrangian
        in_loss = (
            in_constraint_term * lam + 
            in_constraint_weight / 2 * torch.pow(in_constraint_term, 2)
            if in_constraint_term * in_constraint_weight + lam >= 0
            else -torch.pow(lam, 2) * 0.5 / in_constraint_weight
        )

        ce_loss_aug = (
            ce_constraint_term * lam2 + 
            ce_constraint_weight / 2 * torch.pow(ce_constraint_term, 2)
            if ce_constraint_term * ce_constraint_weight + lam2 >= 0
            else -torch.pow(lam2, 2) * 0.5 / ce_constraint_weight
        )

        loss = ce_loss_aug + args.out_constraint_weight * loss_energy_out + in_loss
        loss.backward()
        optimizer.step()

        # logging
        losses_ce.append(loss_ce.item())
        in_losses.append(in_loss.item())
        out_losses.append(loss_energy_out.item())

    avg_train_acc = np.mean(train_accuracies)
    avg_loss_ce = np.mean(losses_ce)
    avg_in_loss = np.mean(in_losses)
    avg_out_loss = np.mean(out_losses)

    # scheduler
    scheduler.step()

    # wandb
    wandb.log({
        'epoch': epoch,
        'train_accuracy': avg_train_acc,
        'CE loss': avg_loss_ce,
        'in loss': avg_in_loss,
        'out loss': avg_out_loss,
        'learning_rate': scheduler.get_last_lr()[0]
    })

    state['train_accuracy'].append(avg_train_acc)

    # === Update lambdas ===
    avg_sigmoid_energy_in, _, avg_ce_loss = evaluate_energy_logistic_loss()
    in_term_constraint = avg_sigmoid_energy_in - args.false_alarm_cutoff
    ce_term_constraint = avg_ce_loss - args.ce_tol * full_train_loss

    if in_term_constraint * in_constraint_weight + lam >= 0:
        lam += args.lr_lam * in_term_constraint
    else:
        lam += -args.lr_lam * lam / in_constraint_weight

    if ce_term_constraint * ce_constraint_weight + lam2 >= 0:
        lam2 += args.lr_lam * ce_term_constraint
    else:
        lam2 += -args.lr_lam * lam2 / ce_constraint_weight

    # escalate penalties
    if in_term_constraint > args.constraint_tol:
        in_constraint_weight *= args.penalty_mult
    if ce_term_constraint > args.constraint_tol:
        ce_constraint_weight *= args.penalty_mult

    # log lambda values
    wandb.log({
        'in_term_constraint': in_term_constraint.item(),
        'ce_term_constraint': ce_term_constraint.item(),
        'lam': lam.item(),
        'lam2': lam2.item()
    })


def test(epoch):
    print(f"\n==> Evaluating on test set at Epoch {epoch + 1}...")

    net.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader_in):
            inputs, targets = inputs.cuda(), targets.cuda()

            outputs = net(inputs)
            loss = F.cross_entropy(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss / len(test_loader_in)
    acc = 100. * correct / total
    print(f"Test Epoch [{epoch+1}] | Loss: {avg_loss:.4f} | Accuracy: {acc:.2f}%")

    # Save stats
    state['test_accuracy'].append(acc)

    # Log to wandb
    wandb.log({
        'epoch': epoch + 1,
        'test_loss': avg_loss,
        'test_accuracy': acc
    })


def test_scone(epoch):
    print(f"\n==> [SCONE] Evaluating on test set at Epoch {epoch + 1}...")

    net.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for inputs, targets in test_loader_in:
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = F.cross_entropy(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

    avg_loss = total_loss / len(test_loader_in)
    acc = 100. * correct / total
    print(f"[SCONE] Test Epoch {epoch+1} | Loss: {avg_loss:.4f} | Accuracy: {acc:.2f}%")

    state['test_accuracy'].append(acc)
    wandb.log({
        'epoch': epoch + 1,
        'test_loss': avg_loss,
        'test_accuracy': acc
    })
def evaluate_classification_loss_training():
    '''
    evaluate classification loss on training dataset
    '''

    net.eval()
    losses = []
    for in_set in train_loader_in:
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



def evaluate_energy_logistic_loss():
    '''
    evaluate energy logistic loss on training dataset
    '''

    net.eval()
    sigmoid_energy_losses = []
    logistic_energy_losses = []
    ce_losses = []
    for in_set in train_loader_in:
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

if args.score in [ 'woods_nn', 'woods', 'scone']:
    full_train_loss = evaluate_classification_loss_training()
    
for epoch in range(args.epochs):
    print(f"epoch {epoch + 1}/{args.epochs}")
    state['epoch'] = epoch
    begin_epoch = time()

    if args.score == 'scone':
        train_scone(epoch)
        test_scone(epoch)
    else:
        train(epoch)
        test(epoch)

    print(f"Epoch {epoch + 1} finished in {time() - begin_epoch:.2f} seconds\n")

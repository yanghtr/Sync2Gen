import os
import time
import sys
import shutil
import random
import argparse
import numpy as np
import importlib
import torch
from torch import nn
import torch.utils.data
from dataloader.loader_discrete import DatasetDiscrete
import vis_utils
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

from scene_loss_utils import get_loss
sys.path.append('models')

from tf_visualizer import Visualizer as TfVisualizer

parser = argparse.ArgumentParser()

# data parameters
parser.add_argument('--data_path', type=str, default='/mnt/yanghaitao/Dataset/Scene_Dataset/3D-FRONT/src/dataset/data/')
parser.add_argument('--type', type=str, help='bedroom or living')
parser.add_argument('--checkpoint_path', default=None, help='Model checkpoint path [default: None]')
parser.add_argument('--log_dir', default='log', help='Dump dir to save model checkpoint [default: log]')
parser.add_argument('--dump_dir', default='dump', help='Dump dir to save results')
parser.add_argument('--not_load_model', action='store_false', help='whether load checkpoint')

# training paramters
parser.add_argument('--max_epoch', type=int, default=1000, help='Epoch to run [default: 180]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 8]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--weight_decay', type=float, default=0, help='Optimization L2 weight decay [default: 0]')
parser.add_argument('--bn_decay_step', type=int, default=20, help='Period of BN decay (in epochs) [default: 20]')
parser.add_argument('--bn_decay_rate', type=float, default=0.5, help='Decay rate for BN decay [default: 0.5]')
parser.add_argument('--lr_decay_steps', default='400,700,1000', help='When to decay the learning rate (in epochs) [default: 80,120,160]')
parser.add_argument('--lr_decay_rates', default='0.1,0.1,0.1', help='Decay rates for lr decay [default: 0.1,0.1,0.1]')

# model parameters
parser.add_argument('--model_dict', type=str, default='model_scene_box_conv_sparse', help='model file name')
parser.add_argument('--variational', action='store_true', help='whether use VAE model')
parser.add_argument('--max_parts', type=int, default=120, help='maximum number of parts in one object')
parser.add_argument('--num_class', type=int, default=30, help='number of semantic class')
parser.add_argument('--num_each_class', type=int, default=4, help='number of each semantic class')
parser.add_argument('--adjust_kld', action='store_true', help='whether adjust kld for different epochs')
parser.add_argument('--weight_kld', type=float, default=0.001, help='weight of kldiv loss')
parser.add_argument('--weight_rep', type=float, default=1.0, help='weight of representation loss')
parser.add_argument('--kld_interval', type=int, default=50, help='interval of kldiv loss cycle')
parser.add_argument('--latent_dim', type=int, default=256, help='latent vector dimension')
parser.add_argument('--abs_dim', type=int, default=16, help='abs dimension')

parser.add_argument('--sparse_num', type=int, default=4, help='sparse linear layer parameter, switch to fully linear if 0')
parser.add_argument('--valid_threshold', type=float, default=0.5, help='output valid mask threshold')
parser.add_argument('--use_dumped_pairs', action='store_true', help='use the dumped pairs')

# general parameters
parser.add_argument('--vis', action='store_true', help='whether do the visualization')
parser.add_argument('--eval', action='store_true', help='whether switch to eval module')
parser.add_argument('--debug', action='store_true', help='whether switch to debug module')
parser.add_argument('--dump_results', action='store_true', help='whether dump predicted result')
parser.add_argument('--eval_epoch', type=int, default=799, help='Epoch to eval')
parser.add_argument('--gen_from_noise', action='store_true', help='whether generate from noise')
parser.add_argument('--num_gen_from_noise', type=int, default=100, help='num of generation samples')

args = parser.parse_args()


MAX_EPOCH = args.max_epoch
BATCH_SIZE = args.batch_size
BASE_LEARNING_RATE = args.learning_rate
BN_DECAY_STEP = args.bn_decay_step
BN_DECAY_RATE = args.bn_decay_rate
LR_DECAY_STEPS = [int(x) for x in args.lr_decay_steps.split(',')]
LR_DECAY_RATES = [float(x) for x in args.lr_decay_rates.split(',')]
assert(len(LR_DECAY_STEPS)==len(LR_DECAY_RATES))
if args.adjust_kld:
    assert(args.variational)

assert(args.type == 'bedroom' or args.type == 'living')
DATA_PATH = os.path.join(args.data_path, args.type)
TRAIN_DATASET = f'{args.data_path}/train_{args.type}.txt'
VAL_DATASET = f'{args.data_path}/val_{args.type}.txt'

# Prepare LOG_DIR
LOG_DIR = args.log_dir
MODEL_NAME = LOG_DIR.split('/')[-1]
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

DEFAULT_CHECKPOINT_PATH = os.path.join(LOG_DIR, f'checkpoint_eval{args.eval_epoch}.tar')
CHECKPOINT_PATH = args.checkpoint_path if args.checkpoint_path is not None \
    else DEFAULT_CHECKPOINT_PATH

# Init datasets and dataloaders 
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

# Create dataset
train_dataset = DatasetDiscrete(DATA_PATH, TRAIN_DATASET)
val_dataset = DatasetDiscrete(DATA_PATH, VAL_DATASET)

num_workers = 0 if args.debug else 4

val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, \
        shuffle=False, num_workers=num_workers, worker_init_fn=my_worker_init_fn)

# Initialize the model and optimizer
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


importlib.invalidate_caches()
model_dict = importlib.import_module(args.model_dict)

if args.variational:
    net = model_dict.VAE(abs_dim=args.abs_dim, num_class=args.num_class, num_each_class=args.num_each_class, latent_dim=args.latent_dim, 
                         use_dumped_pairs=args.use_dumped_pairs, variational=args.variational, log_dir=LOG_DIR)
else:
    raise AssertionError('AE is currently not checked. Need to check model')

print(net)

if torch.cuda.device_count() > 1:
    print("Let's use %d GPUs!" % (torch.cuda.device_count()))
    net = nn.DataParallel(net)

net.to(device)

# Load the Adam optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=BASE_LEARNING_RATE, weight_decay=args.weight_decay)

# Load checkpoint if any
start_epoch = 0
if CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH) and args.not_load_model:
    print('load checkpoint path: %s' % CHECKPOINT_PATH)
    checkpoint = torch.load(CHECKPOINT_PATH)
    model_state_dict_new = {}
    for key, val in checkpoint['model_state_dict'].items():
        if 'pairs' not in key:
            model_state_dict_new[key] = val
    net.load_state_dict(model_state_dict_new)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    print("Successfully Load Model...")

# Helper for learning rate adjustment
def get_current_lr(epoch):
    lr = BASE_LEARNING_RATE
    for i,lr_decay_epoch in enumerate(LR_DECAY_STEPS):
        if epoch >= lr_decay_epoch:
            lr *= LR_DECAY_RATES[i]
    return lr

def adjust_learning_rate(optimizer, epoch):
    lr = get_current_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_weight_kld(epoch):
    if epoch < args.kld_interval:
        return (epoch / args.kld_interval) * args.weight_kld
    else:
        return args.weight_kld


# TFBoard visualizer
TRAIN_VISUALIZER = TfVisualizer(LOG_DIR, 'train')
VAL_VISUALIZER = TfVisualizer(LOG_DIR, 'val')


def eval_one_epoch(epoch, eval_mode):
    assert eval_mode in ['val']
    print('================ In ' + eval_mode + ' mode ================')

    if args.adjust_kld:
        weight_kld = adjust_weight_kld(epoch)
    else:
        weight_kld = args.weight_kld

    stat_dict = {}
    
    net.eval()

    eval_dataloader = val_dataloader

    if args.dump_results:
        DUMP_DIR = f'{args.dump_dir}/{args.type}/{MODEL_NAME}'
        if not os.path.exists(DUMP_DIR):
            os.makedirs(DUMP_DIR)

    '''
        Note that we set torch.manual_seed(0)
    '''
    if args.gen_from_noise:
        assert(args.variational)
        num_epochs = args.num_gen_from_noise // BATCH_SIZE
        num_rest = args.num_gen_from_noise % BATCH_SIZE
        count = 0
        for i in range(num_epochs + 1):
            print(i)
            with torch.no_grad():
                if i < num_epochs:
                    num_noise = BATCH_SIZE
                else:
                    num_noise = num_rest

                latent_code = torch.randn(num_noise, args.latent_dim).to(device)

                pred_abs, pred_rel = net(None, None, latent_code=latent_code)

            if args.dump_results:
                for idx in range(num_noise):
                    np.save(os.path.join(DUMP_DIR, str(count).zfill(4) + '_abs_pred.npy'), pred_abs[idx].detach().cpu().numpy())
                    np.save(os.path.join(DUMP_DIR, str(count).zfill(4) + '_rel_pred.npy'), pred_rel[idx].detach().cpu().numpy())
                    count += 1

        assert(count == args.num_gen_from_noise)

    else:
        for batch_idx, batch_data_label in enumerate(eval_dataloader):
            if batch_idx % 10 == 0:
                print('Eval batch: %d'%(batch_idx))

            for key in batch_data_label:
                batch_data_label[key] = batch_data_label[key].to(device)

            # Forward pass
            inputs_abs = batch_data_label['X_abs']
            inputs_rel = batch_data_label['X_rel']

            labels_abs = batch_data_label['X_abs']
            labels_rel = batch_data_label['X_rel']

            with torch.no_grad():
                pred_abs, pred_rel, kldiv_loss = net(inputs_abs, inputs_rel)

                total_loss, loss_dict, _, _, _ = get_loss(args.type, labels_abs, pred_abs, labels_rel, pred_rel,
                                                    num_class=args.num_class, 
                                                    num_each_class=args.num_each_class,
                                                    )
            if args.variational:
                total_loss += weight_kld * kldiv_loss
                loss_dict['kldiv_loss'] = weight_kld * kldiv_loss
                loss_dict['loss'] += weight_kld * kldiv_loss

            # Accumulate statistics and print out
            for key in loss_dict:
                if key not in stat_dict: stat_dict[key] = 0
                stat_dict[key] += loss_dict[key].item()

            if args.dump_results:
                if not os.path.exists(f"joint_hyper_opt/{DUMP_DIR}"):
                    os.makedirs(f"joint_hyper_opt/{DUMP_DIR}")
                for b in range(batch_data_label['X_abs'].shape[0]):
                    fid = str(batch_data_label['index'][b].cpu().numpy()).zfill(4)
                    print(fid)
                    np.save(f'joint_hyper_opt/{DUMP_DIR}/{fid}_abs_gt.npy', labels_abs[b].detach().cpu().numpy())
                    np.save(f'joint_hyper_opt/{DUMP_DIR}/{fid}_abs_pred.npy', pred_abs[b].detach().cpu().numpy())
                    np.save(f'joint_hyper_opt/{DUMP_DIR}/{fid}_rel_pred.npy', pred_rel[b].detach().cpu().numpy())

        for key in sorted(stat_dict.keys()):
            print('%s %s: %f' % (eval_mode, key, stat_dict[key] / (float(batch_idx+1))), end=' ')

    return 0


def test(eval_epoch):
    np.random.seed(0)
    torch.manual_seed(0)
    eval_one_epoch(eval_epoch, 'val')


if __name__ == '__main__':
    test(args.eval_epoch)


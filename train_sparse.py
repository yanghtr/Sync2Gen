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

DEFAULT_CHECKPOINT_PATH = os.path.join(LOG_DIR, 'checkpoint.tar')
CHECKPOINT_PATH = args.checkpoint_path if args.checkpoint_path is not None \
    else DEFAULT_CHECKPOINT_PATH

# Init datasets and dataloaders 
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

# Create dataset
train_dataset = DatasetDiscrete(DATA_PATH, TRAIN_DATASET)
val_dataset = DatasetDiscrete(DATA_PATH, VAL_DATASET)

num_workers = 0 if args.debug else 4

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, \
        shuffle=True, num_workers=num_workers, worker_init_fn=my_worker_init_fn)
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
    net.load_state_dict(checkpoint['model_state_dict'])
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


def train_one_epoch(epoch):
    stat_dict = {}
    adjust_learning_rate(optimizer, epoch)

    if args.adjust_kld:
        weight_kld = adjust_weight_kld(epoch)
    else:
        weight_kld = args.weight_kld
    print('Current weight of kld loss: %f'% weight_kld)

    net.train()

    for batch_idx, batch_data_label in enumerate(train_dataloader):

        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].to(device)

        # Forward pass
        optimizer.zero_grad()
 
        inputs_abs = batch_data_label['X_abs']
        inputs_rel = batch_data_label['X_rel']

        labels_abs = batch_data_label['X_abs']
        labels_rel = batch_data_label['X_rel']

        pred_abs, pred_rel, kldiv_loss = net(inputs_abs, inputs_rel)

        total_loss, loss_dict, batch_index, matched_gt_idx, matched_pred_idx = get_loss(args.type, labels_abs, pred_abs, labels_rel, pred_rel,
                                            num_class=args.num_class, 
                                            num_each_class=args.num_each_class,
                                            )

        if args.variational:
            total_loss += weight_kld * kldiv_loss
            loss_dict['kldiv_loss'] = weight_kld * kldiv_loss
            loss_dict['loss'] += weight_kld * kldiv_loss

        total_loss.backward()

        optimizer.step()

        # Accumulate statistics and print out
        for key in loss_dict:
            if key not in stat_dict: stat_dict[key] = 0
            stat_dict[key] += loss_dict[key].item()

        batch_interval = 50
        if (batch_idx+1) % batch_interval == 0:
            print('batch: %03d:' % (batch_idx+1), end=' ')
            TRAIN_VISUALIZER.log_scalars({key: stat_dict[key] / batch_interval for key in stat_dict},
                (epoch*len(train_dataloader)+batch_idx)*BATCH_SIZE)
            for key in sorted(stat_dict.keys()):
                print('%s: %f |' % (key, stat_dict[key] / batch_interval), end=' ')
                stat_dict[key] = 0
            print()



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
            if args.variational:
                pred_abs, pred_rel, kldiv_loss = net(inputs_abs, inputs_rel)

            total_loss, loss_dict, batch_index, matched_gt_idx, matched_pred_idx = get_loss(args.type, labels_abs, pred_abs, labels_rel, pred_rel,
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


    for key in sorted(stat_dict.keys()):
        print('%s %s: %f' % (eval_mode, key, stat_dict[key] / (float(batch_idx+1))), end=' ')

    mean_loss = stat_dict['loss']/float(batch_idx+1)

    if args.debug:
        import ipdb; ipdb.set_trace()
        return mean_loss
    
    if args.vis or args.eval or args.dump_results:
        return mean_loss

    # Log statistics
    VAL_VISUALIZER.log_scalars({key: stat_dict[key] / float(batch_idx+1) for key in stat_dict},
        (epoch+1) * len(train_dataloader) * BATCH_SIZE)
    print()

    return mean_loss

        

def train(start_epoch):
    for epoch in range(start_epoch, MAX_EPOCH):
        print('**** EPOCH %03d ****' % (epoch))
        print('Current learning rate: %f'%(get_current_lr(epoch)))
        if args.vis or args.eval:
            val_loss = eval_one_epoch(epoch, 'val')
            break
 
        # Reset numpy seed.
        # REF: https://github.com/pytorch/pytorch/issues/5059
        '''  here: problem: whether need to use seed  '''
        np.random.seed()
        train_one_epoch(epoch)

        # Eval every 10 epochs
        if epoch == 0 or epoch % 10 == 9: 
            val_loss = eval_one_epoch(epoch, 'val')
            # Save checkpoint
            save_dict = {'epoch': epoch+1, # after training one epoch, the start_epoch should be epoch+1
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': val_loss,
                        }
            try: # with nn.DataParallel() the net is added as a submodule of DataParallel
                save_dict['model_state_dict'] = net.module.state_dict()
            except:
                save_dict['model_state_dict'] = net.state_dict()
            torch.save(save_dict, os.path.join(LOG_DIR, 'checkpoint.tar'))
            if epoch > 0 and epoch % 100 == 99:
                torch.save(save_dict, os.path.join(LOG_DIR, 'checkpoint_eval%d.tar' % epoch))


if __name__ == '__main__':
    train(start_epoch)

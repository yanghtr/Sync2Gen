import os
import sys
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
import utils

from models.sparse_linear import SparseLinear, generate_pairs, generate_pairs_reverse

sparse_num = 4

''' abs network param '''
ldim1_abs = 30
ldim2_abs = 40
ldim3_abs = 50

layer_sc1_abs = 200
layer_sc2_abs = 80
layer_sc3_abs = 20

# fc here serves as pooling, which reduces layer_sci
layer_fc1_abs = 20
layer_fc2_abs = 10
layer_fc3_abs = 4


''' rel network param '''
ldim1_rel = 32
ldim2_rel = 64
ldim3_rel = 256

layer_sc1_rel = 256
layer_sc2_rel = 128
layer_sc3_rel = 64

layer_fc1_rel = 64
layer_fc2_rel = 32
layer_fc3_rel = 16


def dump_pairs_abs(log_dir, num_class, num_each_class):
    dump_dir = os.path.join(log_dir, 'pairs')
    if not os.path.exists(dump_dir):
        os.mkdir(dump_dir)
    if os.path.exists(os.path.join(dump_dir, 'pairs_abs.pkl')):
        raise AssertionError('overwrite pairs_abs.pkl')

    print('--- start gen abs pairs ---')
    pairs_dict = {}
    pairs_dict['pairs1'] = generate_pairs(num_class, sparse_num, layer_sc1_abs)
    pairs_dict['pairs2'] = generate_pairs(layer_fc1_abs, sparse_num, layer_sc2_abs)
    pairs_dict['pairs3'] = generate_pairs(layer_fc2_abs, sparse_num, layer_sc3_abs)

    pairs_dict['pairs_reverse1'] = generate_pairs_reverse(layer_fc2_abs, 1, pairs_dict['pairs3'])
    pairs_dict['pairs_reverse2'] = generate_pairs_reverse(layer_fc1_abs, 1, pairs_dict['pairs2'])
    pairs_dict['pairs_reverse3'] = generate_pairs_reverse(num_class, num_each_class, pairs_dict['pairs1'])

    with open(os.path.join(dump_dir, 'pairs_abs.pkl'), 'wb') as f:
        pickle.dump(pairs_dict, f)
    print('--- dump abs pairs success ---')



class Encoder_Abs(nn.Module):

    def __init__(self, input_dim=16, num_class=30, num_each_class=4, bn_momentum=0.01, 
                 variational=False, latent_dim=256, input_pairs_list=None): # (qi, ti, si, ci)
        super(Encoder_Abs, self).__init__()
        
        self.input_dim = input_dim
        self.num_class = num_class
        self.num_each_class = num_each_class
        self.max_parts = num_class * num_each_class 
        self.variational = variational

        self.sc1 = SparseLinear(num_class, input_dim * num_each_class, layer_sc1_abs, ldim1_abs, sparse_num, input_pairs_list[0])
        self.bn_sc1 = nn.BatchNorm1d(layer_sc1_abs * ldim1_abs, momentum=bn_momentum)
        self.fc1 = nn.Linear(layer_sc1_abs, layer_fc1_abs)
        self.bn_fc1 = nn.BatchNorm1d(layer_fc1_abs * ldim1_abs, momentum=bn_momentum)
        
        self.sc2 = SparseLinear(layer_fc1_abs, ldim1_abs, layer_sc2_abs, ldim2_abs, sparse_num, input_pairs_list[1])
        self.bn_sc2 = nn.BatchNorm1d(layer_sc2_abs * ldim2_abs, momentum=bn_momentum)
        self.fc2 = nn.Linear(layer_sc2_abs, layer_fc2_abs)
        self.bn_fc2 = nn.BatchNorm1d(layer_fc2_abs * ldim2_abs, momentum=bn_momentum)

        self.sc3 = SparseLinear(layer_fc2_abs, ldim2_abs, layer_sc3_abs, ldim3_abs, sparse_num, input_pairs_list[2])
        self.bn_sc3 = nn.BatchNorm1d(layer_sc3_abs * ldim3_abs, momentum=bn_momentum)
        self.fc3 = nn.Linear(layer_sc3_abs, layer_fc3_abs)
        self.bn_fc3 = nn.BatchNorm1d(layer_fc3_abs * ldim3_abs, momentum=bn_momentum)

        self.mlp2mu = nn.Linear(layer_fc3_abs * ldim3_abs, latent_dim)
        self.mlp2var = nn.Linear(layer_fc3_abs * ldim3_abs, latent_dim)

        self.bn_mlp2mu = nn.BatchNorm1d(latent_dim, momentum=bn_momentum)

    def forward(self, x):
        '''
        @Args:
            x: (B, max_parts, input_dim)
        @Returns:
            x: (B, latent_dim)
        '''
        assert(x.shape[1] == self.num_class * self.num_each_class)
        B = x.shape[0]
        x = x.reshape(-1, self.num_class, self.input_dim * self.num_each_class)

        x = self.sc1(x) # (B, 200, 30)
        x = x.reshape(B, -1)
        x = F.relu(self.bn_sc1(x)).reshape(B, layer_sc1_abs, ldim1_abs) # (B, 200, 30)

        x = self.fc1(x.permute(0, 2, 1).reshape(B * ldim1_abs, layer_sc1_abs)).reshape(B, ldim1_abs, layer_fc1_abs).permute(0, 2, 1) # (B, 20, 30)
        x = x.reshape(B, -1)
        x = F.relu(self.bn_fc1(x)).reshape(B, layer_fc1_abs, ldim1_abs) # (B, 20, 30)

        x = self.sc2(x) # (B, 80, 40)
        x = x.reshape(B, -1)
        x = F.relu(self.bn_sc2(x)).reshape(B, layer_sc2_abs, ldim2_abs) # (B, 80, 40)

        x = self.fc2(x.permute(0, 2, 1).reshape(B * ldim2_abs, layer_sc2_abs)).reshape(B, ldim2_abs, layer_fc2_abs).permute(0, 2, 1) # (B, 10, 40)
        x = x.reshape(B, -1)
        x = F.relu(self.bn_fc2(x)).reshape(B, layer_fc2_abs, ldim2_abs) # (B, 10, 40)

        x = self.sc3(x) # (B, 20, 50)
        x = x.reshape(B, -1)
        x = F.relu(self.bn_sc3(x)).reshape(B, layer_sc3_abs, ldim3_abs) # (B, 20, 50)

        x = self.fc3(x.permute(0, 2, 1).reshape(B * ldim3_abs, layer_sc3_abs)).reshape(B, ldim3_abs, layer_fc3_abs).permute(0, 2, 1) # (B, 4, 50)
        x = x.reshape(B, -1)
        x = F.relu(self.bn_fc3(x)) # (B, 4 * 50)

        if self.variational:
            mu = self.mlp2mu(x)
            logvar = self.mlp2var(x)
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std)

            kld = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)

            z = eps.mul(std).add_(mu)
            return z, kld

        else:
            ''' If variational == False, the returned is feature, not latent code (which does not contain relu * bn)
            '''
            z = F.relu(self.bn_mlp2mu(self.mlp2mu(x))) # (B, latent_dim)
            return z


class Decoder_Abs(nn.Module):

    def __init__(self, output_dim=16, num_class=30, num_each_class=4, bn_momentum=0.01, latent_dim=256, input_pairs_list=None):
        super(Decoder_Abs, self).__init__()

        self.output_dim = output_dim
        self.num_class = num_class
        self.num_each_class = num_each_class
        self.max_parts = num_class * num_each_class 

        assert(len(input_pairs_list) == 3)
        # input_pairs1, input_pairs2, input_pairs3 = input_pairs_list

        self.fc0 = nn.Linear(latent_dim, layer_sc3_abs * ldim3_abs)
        self.bn_fc0 = nn.BatchNorm1d(layer_sc3_abs * ldim3_abs, momentum=bn_momentum)

        self.sc1 = SparseLinear(layer_sc3_abs, ldim3_abs, layer_fc2_abs, ldim2_abs, sparse_num, input_pairs_list[0], is_decoder=True)
        self.bn_sc1 = nn.BatchNorm1d(layer_fc2_abs * ldim2_abs, momentum=bn_momentum)
        self.fc1 = nn.Linear(layer_fc2_abs * ldim2_abs, layer_sc2_abs * ldim2_abs)  # unpooling
        self.bn_fc1 = nn.BatchNorm1d(layer_sc2_abs * ldim2_abs, momentum=bn_momentum)

        self.sc2 = SparseLinear(layer_sc2_abs, ldim2_abs, layer_fc1_abs, ldim1_abs, sparse_num, input_pairs_list[1], is_decoder=True)
        self.bn_sc2 = nn.BatchNorm1d(layer_fc1_abs * ldim1_abs, momentum=bn_momentum)
        self.fc2 = nn.Linear(layer_fc1_abs * ldim1_abs, layer_sc1_abs * ldim1_abs)  # unpooling
        self.bn_fc2 = nn.BatchNorm1d(layer_sc1_abs * ldim1_abs, momentum=bn_momentum)

        self.sc3 = SparseLinear(layer_sc1_abs, ldim1_abs, num_class * num_each_class, output_dim, sparse_num, input_pairs_list[2], is_decoder=True)
        self.bn_sc3 = nn.BatchNorm1d(num_class * num_each_class * output_dim)
        self.fc3 = nn.Linear(output_dim, output_dim)

        
    def forward(self, x):
        '''
        @Args:
            x: (B, 256)
        @Returns:
            output: (B, N, output_dim),  (qi, ti, si, ci)
        '''
        B = x.shape[0]
        x = F.relu(self.bn_fc0(self.fc0(x)))
        x = x.reshape(B, layer_sc3_abs, ldim3_abs) # (B, 20, 50)

        x = self.sc1(x) # (B, 10, 40)
        x = F.relu(self.bn_sc1(x.reshape(B, layer_fc2_abs * ldim2_abs)))
        x = F.relu(self.bn_fc1(self.fc1(x))).reshape(B, layer_sc2_abs, ldim2_abs) # (B, 80, 40)

        x = self.sc2(x) # (B, 20, 30)
        x = F.relu(self.bn_sc2(x.reshape(B, layer_fc1_abs * ldim1_abs)))
        x = F.relu(self.bn_fc2(self.fc2(x))).reshape(B, layer_sc1_abs, ldim1_abs) # (B, 200, 30)

        x = self.sc3(x) # (B, num_class * num_each_class, output_dim)

        if self.output_dim == 10:
            x_norm = torch.norm(x[:, :, :3], dim=2, keepdim=True).repeat(1, 1, 3) + 1e-12
            x_param = x[:, :, :3] / x_norm

            ''' TODO: check whether need sigmoid '''
            center = x[:, :, 3:6]
            size = x[:, :, 6:9]
            indicator_logits = x[:, :, -1:] # Need sigmoid to get probability
     
            output = torch.cat((x_param, center, size, indicator_logits), dim=2)
            return output

        elif self.output_dim == 16:
            # angle_class_id(8), angle_residual(1), translation(3), size(3), indicator(1)
            return x

        else:
            raise AssertionError('not implement')



class Decoder_Rel(nn.Module):
    def __init__(self, num_class=30, num_each_class=4, bn_momentum=0.01, halfRange=6, interval=0.3):
        super(Decoder_Rel, self).__init__()

        self.num_class = num_class
        self.num_each_class = num_each_class
        self.max_parts = num_class * num_each_class 
        self.halfRange = halfRange
        self.interval = interval
        self.num_bins = int(halfRange / interval + 1)
        self.output_dim = 2 + 2 * self.num_bins + 2 + 1 + 3 + 1 + 1 + 1

        # encoder
        self.fc11 = nn.Conv2d(2 * (9 + num_class), 64, 1, 1)
        self.bn_fc11 = nn.BatchNorm2d(64, momentum=bn_momentum)
        self.fc12 = nn.Conv2d(64, 128, 1, 1)
        self.bn_fc12 = nn.BatchNorm2d(128, momentum=bn_momentum)
        self.fc13 = nn.Conv2d(128, 256, 1, 1)
        self.bn_fc13 = nn.BatchNorm2d(256, momentum=bn_momentum)

        self.pool1 = nn.MaxPool2d(num_each_class, stride=num_each_class)

        self.fc21 = nn.Conv2d(256, 256, 1, 1)
        self.bn_fc21 = nn.BatchNorm2d(256, momentum=bn_momentum)
        self.fc22 = nn.Conv2d(256, 256, 1, 1)
        self.bn_fc22 = nn.BatchNorm2d(256, momentum=bn_momentum)

        self.pool2 = nn.MaxPool2d(num_class, stride=num_class)

        self.fc31 = nn.Conv2d(256, 512, 1, 1)
        self.bn_fc31 = nn.BatchNorm2d(512, momentum=bn_momentum)
        self.fc32 = nn.Conv2d(512, 256, 1, 1)
        self.bn_fc32 = nn.BatchNorm2d(256, momentum=bn_momentum)

        # decoder
        self.deconv1 = nn.ConvTranspose2d(256, 256, num_class, stride=num_class) 
        self.bn_deconv1 = nn.BatchNorm2d(256, momentum=bn_momentum)

        self.fc41 = nn.Conv2d(256 + 256, 256, 1, 1)
        self.bn_fc41 = nn.BatchNorm2d(256, momentum=bn_momentum)
        self.fc42 = nn.Conv2d(256, 256, 1, 1)
        self.bn_fc42 = nn.BatchNorm2d(256, momentum=bn_momentum)

        self.deconv2 = nn.ConvTranspose2d(256, 256, num_each_class, stride=num_each_class) 
        self.bn_deconv2 = nn.BatchNorm2d(256, momentum=bn_momentum)

        self.fc51 = nn.Conv2d(256 + 256, 256, 1, 1)
        self.bn_fc51 = nn.BatchNorm2d(256, momentum=bn_momentum)
        self.fc52 = nn.Conv2d(256, 128, 1, 1)
        self.bn_fc52 = nn.BatchNorm2d(128, momentum=bn_momentum)
        self.fc53 = nn.Conv2d(128, self.output_dim, 1, 1)


    def forward(self, pred_abs):
        '''
        @Args:
            pred_abs: (B, N, abs_dim)
        @Returns:
            output: (B, N, N, output_dim)
        '''
        B, N = pred_abs.shape[0], pred_abs.shape[1]
        assert(pred_abs.shape[2] == 16)
        x = utils.parse_abs_batch(pred_abs)
        assert(x.shape[2] == 9)

        sem = torch.zeros((B, N, self.num_class)).to(pred_abs.device)
        for i in range(self.num_class):
            sem[:, i * self.num_each_class : (i+1) * self.num_each_class, i] = 1
        x = torch.cat((x, sem), dim=2)

        x = torch.cat((x[:, :, None, :].repeat(1, 1, self.max_parts, 1), x[:, None, :, :].repeat(1, self.max_parts, 1, 1)), dim=3) # (B, N, N, 2 * abs_dim)
        x = x.permute(0, 3, 1, 2).contiguous()

        x  = F.relu(self.bn_fc11(self.fc11(x))) # (B, 64, N, N)
        x  = F.relu(self.bn_fc12(self.fc12(x)))
        x1 = F.relu(self.bn_fc13(self.fc13(x)))
        x  = self.pool1(x1) # (B, 256, C, C)

        x  = F.relu(self.bn_fc21(self.fc21(x)))
        x2 = F.relu(self.bn_fc22(self.fc22(x)))
        x  = self.pool2(x2) # (B, 256, 1, 1)

        x  = F.relu(self.bn_fc31(self.fc31(x)))
        x  = F.relu(self.bn_fc32(self.fc32(x)))

        x  = F.relu(self.bn_deconv1(self.deconv1(x))) # (B, 256, C, C)
        x  = F.relu(self.bn_fc41(self.fc41(torch.cat((x, x2), dim=1)))) # (B, 256, C, C)
        x  = F.relu(self.bn_fc42(self.fc42(x))) # (B, 256, C, C)

        x  = F.relu(self.bn_deconv2(self.deconv2(x))) # (B, 256, N, N)
        x  = F.relu(self.bn_fc51(self.fc51(torch.cat((x, x1), dim=1)))) # (B, 256, N, N)
        x  = F.relu(self.bn_fc52(self.fc52(x))) # (B, 256, N, N)
        x  = self.fc53(x).permute(0, 2, 3, 1) # (B, N, N, output_dim)

        if True:
            ''' x:
            output_dim = 2 + 2 * self.num_bins + 2 + 1 + 3 + 1 + 1 + 1
                0 -- Ix
                1 -- Iy
                2:2+num_bins -- cx
                2+num_bins:2+2*num_bins -- cy
                -9 -- rx
                -8 -- ry
                -7 -- z
                -6:-3 -- rotation_class 
                -3 -- same_size 
                -2 -- rel_size 
                -1 -- rel_indicator (deprecated)
            '''

            Ix             = x[:, :, :, 0:1]
            Iy             = x[:, :, :, 1:2]
            cx             = x[:, :, :, 2               : 2+self.num_bins]
            cy             = x[:, :, :, 2+self.num_bins : 2+2*self.num_bins]
            rx             = x[:, :, :, -9:-8]
            ry             = x[:, :, :, -8:-7]
            z              = x[:, :, :, -7:-6]
            rotation_class = x[:, :, :, -6:-3]
            same_size      = x[:, :, :, -3:-2]
            rel_size       = x[:, :, :, -2:-1]
            mask           = x[:, :, :, -1:]

            Ix    = (Ix    -    Ix.permute(0, 2, 1, 3)) / 2
            Iy    = (Iy    -    Iy.permute(0, 2, 1, 3)) / 2
            cx    = (cx    +    cx.permute(0, 2, 1, 3)) / 2
            cy    = (cy    +    cy.permute(0, 2, 1, 3)) / 2
            rx    = (rx    +    rx.permute(0, 2, 1, 3)) / 2
            ry    = (ry    +    ry.permute(0, 2, 1, 3)) / 2
            z     = (z     -     z.permute(0, 2, 1, 3)) / 2
            rotation_class = (rotation_class + rotation_class.permute(0, 2, 1, 3)) / 2
            same_size      = (same_size + same_size.permute(0, 2, 1, 3)) / 2

            output = torch.cat((Ix, Iy, cx, cy, rx, ry, z, rotation_class, same_size, rel_size, mask), dim=3)

            return output


class VAE(nn.Module):
    def __init__(self, abs_dim=16, num_class=30, num_each_class=4, bn_momentum=0.01, latent_dim=32, use_dumped_pairs=False, variational=True, log_dir=None, feat_dim=256, halfRange=6, interval=0.3):
        '''
        @Args:
            latent_dim: z~N(0, 1), latent dimension
        '''
        super(VAE, self).__init__()
        self.variational = variational

        if not use_dumped_pairs:
            dump_pairs_abs(log_dir, num_class, num_each_class)
        
        pairs_dict_abs = pickle.load(open(os.path.join(log_dir, 'pairs', 'pairs_abs.pkl'), 'rb'))
        input_pairs_list_abs = [pairs_dict_abs['pairs1'], pairs_dict_abs['pairs2'], pairs_dict_abs['pairs3']]
        input_pairs_reverse_list_abs = [pairs_dict_abs['pairs_reverse1'], pairs_dict_abs['pairs_reverse2'], pairs_dict_abs['pairs_reverse3']]

        # set variational = False so that return latent feature, which are combined together with rel
        self.encoder_abs = Encoder_Abs(abs_dim, num_class, num_each_class, bn_momentum, latent_dim=latent_dim, variational=True, input_pairs_list=input_pairs_list_abs)
        self.decoder_abs = Decoder_Abs(abs_dim, num_class, num_each_class, bn_momentum, latent_dim=latent_dim, input_pairs_list=input_pairs_reverse_list_abs)

        num_bins = int(halfRange / interval + 1)
        self.decoder_rel = Decoder_Rel(num_class, num_each_class, bn_momentum, halfRange=halfRange, interval=interval)

    def forward(self, x_abs, x_rel=None, latent_code=None):
        if latent_code is not None:
            pred_abs = self.decoder_abs(latent_code)
            pred_rel = self.decoder_rel(pred_abs)
            return pred_abs, pred_rel

        z, kld = self.encoder_abs(x_abs) # (B, latent_dim)
        kldiv_loss = -kld.sum() / x_abs.shape[0]

        pred_abs = self.decoder_abs(z)
        pred_rel = self.decoder_rel(pred_abs.detach())
        # pred_rel = self.decoder_rel(pred_abs)

        return pred_abs, pred_rel, kldiv_loss


if __name__ == '__main__':
    x_abs = torch.rand(4, 40, 16)



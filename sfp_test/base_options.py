import argparse
import torch

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--patchSize', type=int, default=32, help='then crop to this size')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
        self.parser.add_argument('--use_wgan', type=float, default=0, help='use wgan-gp')
        self.parser.add_argument('--use_ragan', action='store_true', help='use ragan')
        self.parser.add_argument('--vgg', type=float, default=1, help='use perceptrual loss')
        self.parser.add_argument('--vgg_mean', action='store_true', help='substract mean in vgg loss')
        self.parser.add_argument('--vgg_choose', type=str, default='relu5_1', help='choose layer for vgg')
        self.parser.add_argument('--no_vgg_instance', action='store_true', help='vgg instance normalization')
        self.parser.add_argument('--vgg_maxpooling', action='store_true', help='normalize attention map')
        self.parser.add_argument('--IN_vgg', action='store_true', help='patch vgg individual')
        self.parser.add_argument('--use_avgpool', type=float, default=0, help='use perceptrual loss')
        self.parser.add_argument('--instance_norm', type=float, default=0, help='use instance normalization')
        self.parser.add_argument('--syn_norm', action='store_true', help='use synchronize batch normalization')
        self.parser.add_argument('--tanh', action='store_true', help='tanh')
        self.parser.add_argument('--linear', action='store_true', help='tanh')
        self.parser.add_argument('--new_lr', action='store_true', help='tanh')
        self.parser.add_argument('--multiply', action='store_true', help='tanh')
        self.parser.add_argument('--noise', type=float, default=0, help='variance of noise')
        self.parser.add_argument('--input_linear', action='store_true', help='lieanr scaling input')
        self.parser.add_argument('--linear_add', action='store_true', help='lieanr scaling input')
        self.parser.add_argument('--latent_threshold', action='store_true', help='lieanr scaling input')
        self.parser.add_argument('--latent_norm', action='store_true', help='lieanr scaling input')
        self.parser.add_argument('--patchD', action='store_true', help='use patch discriminator')
        self.parser.add_argument('--patchD_3', type=int, default=5, help='choose the number of crop for patch discriminator')
        self.parser.add_argument('--D_P_times2', action='store_true', help='loss_D_P *= 2')
        self.parser.add_argument('--patch_vgg', action='store_true', help='use vgg loss between each patch')
        self.parser.add_argument('--hybrid_loss', action='store_true', help='use lsgan and ragan separately')
        self.parser.add_argument('--self_attention', action='store_true', help='adding attention on the input of generator')
        self.parser.add_argument('--times_residual', action='store_true', help='output = input + residual*attention')
        self.parser.add_argument('--low_times', type=int, default=200, help='choose the number of crop for patch discriminator')
        self.parser.add_argument('--high_times', type=int, default=400, help='choose the number of crop for patch discriminator')
        self.parser.add_argument('--norm_attention', action='store_true', help='normalize attention map')
        self.parser.add_argument('--vary', type=int, default=1, help='use light data augmentation')
        self.parser.add_argument('--lighten', action='store_true', help='normalize attention map')
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)
        
        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        return self.opt

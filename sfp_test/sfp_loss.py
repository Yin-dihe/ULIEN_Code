
from torch import nn

import random
from . import networks


class Sfp(nn.Module):
    def __init__(self,opt):
        super(Sfp,self).__init__()
        self.opt = opt
        self.vgg_loss = networks.PerceptualLoss(opt)
        self.vgg_loss.cuda()
        self.gpu_ids = opt.gpu_ids
        self.vgg = networks.load_vgg16("./model", self.gpu_ids)
        self.vgg.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self,lowlight_img,enhance_img):
        #  计算分块
        w = lowlight_img.size(3)
        h = lowlight_img.size(2)
        w_offset = random.randint(0, max(0, w - self.opt.patchSize - 1))
        h_offset = random.randint(0, max(0, h - self.opt.patchSize - 1))
        lowlight_img_patch = lowlight_img[:, :, h_offset:h_offset + self.opt.patchSize,
                           w_offset:w_offset + self.opt.patchSize]
        enhance_img_patch = enhance_img[:, :, h_offset:h_offset + self.opt.patchSize,
                          w_offset:w_offset + self.opt.patchSize]

        lowlight_img_patch1 = []
        enhance_img_patch1 = []  #
        for i in range(self.opt.patchD_3):
            w_offset_1 = random.randint(0, max(0, w - self.opt.patchSize - 1))
            h_offset_1 = random.randint(0, max(0, h - self.opt.patchSize - 1))
            lowlight_img_patch1.append(lowlight_img[:, :, h_offset_1:h_offset_1 + self.opt.patchSize,
                                      w_offset_1:w_offset_1 + self.opt.patchSize])
            enhance_img_patch1.append(enhance_img[:, :, h_offset_1:h_offset_1 + self.opt.patchSize,
                                     w_offset_1:w_offset_1 + self.opt.patchSize])


        vgg_w = 1
        self.loss_vgg_b = self.vgg_loss.compute_vgg_loss(self.vgg, enhance_img, lowlight_img) * self.opt.vgg

        loss_vgg_patch = self.vgg_loss.compute_vgg_loss(self.vgg, enhance_img_patch, lowlight_img_patch) * self.opt.vgg
        for i in range(self.opt.patchD_3):
            loss_vgg_patch += self.vgg_loss.compute_vgg_loss(self.vgg, enhance_img_patch1[i],
                                                             lowlight_img_patch1[i]) * self.opt.vgg
            self.loss_vgg_b += loss_vgg_patch / float(self.opt.patchD_3 + 1)
        SFPLOSS = self.loss_vgg_b * vgg_w

        # SFPLOSS=self.loss_vgg_b* vgg_w
        return SFPLOSS









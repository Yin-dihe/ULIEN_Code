
import torch

import torch.optim
import os

import argparse

import dataloader

import Myloss

import model

from sfp_test.sfp_loss import Sfp
from sfp_test.train_options import TrainOptions

from torch.utils.tensorboard import SummaryWriter


logger = SummaryWriter(log_dir="./log/logz")


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
def train(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    opt = TrainOptions().parse()

    ULIEN_net = model.ULIEN().cuda()

    ULIEN_net.apply(weights_init)
    if config.load_pretrain == True:
        ULIEN_net.load_state_dict(torch.load(config.pretrain_dir))
    train_dataset = dataloader.lowlight_loader(config.lowlight_images_path)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True,
                                               num_workers=config.num_workers, pin_memory=True)

    L_exp = Myloss.L_exp(16, 0.5)
    L_TV = Myloss.L_TV()
    Sfp_loss = Sfp(opt)
    optimizer = torch.optim.Adam(ULIEN_net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    ULIEN_net.train()
    costs = []
    for epoch in range(config.num_epochs):
        cost = 0.0
        for iteration, img_lowlight in enumerate(train_loader):

            img_lowlight = img_lowlight.cuda()

            for i in range(img_lowlight.shape[0]):
                r, g, b = img_lowlight.data[i][0] + 1, img_lowlight.data[i][1] + 1, img_lowlight.data[i][2] + 1
                A_gray = 1. - (0.299 * r + 0.587 * g + 0.114 * b)
                A_gray = torch.unsqueeze(A_gray, 0)
                A_gray = torch.unsqueeze(A_gray, 0)
                # A_gray=1.-A_gray;
                if i == 0:
                    gray = A_gray
                    continue
                else:
                    gray = torch.cat([gray, A_gray], dim=0)


            enhanced_image_1, enhanced_image, A = ULIEN_net(img_lowlight, gray)
            Loss_TV = 200 * L_TV(A)

            loss_exp = 20 * torch.mean(L_exp(enhanced_image))  # 原文
            Sfp_loss1 = 0.5 * Sfp_loss(img_lowlight, enhanced_image)

            loss = Loss_TV + loss_exp + Sfp_loss1

            cost += loss.item()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(ULIEN_net.parameters(), config.grad_clip_norm)
            optimizer.step()
            # logger.add_scalar(f"第{epoch + 1}轮loss", loss, iteration)
            if epoch % 10 == 0:
                logger.add_scalar(f"第{epoch + 1}轮loss", loss, iteration)
            if ((iteration + 1) % config.display_iter) == 0:
                print("Loss at iteration", iteration + 1, ":", loss.item())
            if ((iteration + 1) % config.snapshot_iter) == 0:
                torch.save(ULIEN_net.state_dict(), config.snapshots_folder + "Epoch" + str(epoch) + '.pth')
        costs.append(cost / len(train_loader))
        logger.add_scalar("loss", cost / len(train_loader), epoch)
    print(costs)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Input Parameters
    parser.add_argument('--lowlight_images_path', type=str, default="data/train_data/")
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--val_batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--display_iter', type=int, default=10)
    parser.add_argument('--snapshot_iter', type=int, default=10)
    parser.add_argument('--snapshots_folder', type=str, default="snapshots/")
    parser.add_argument('--load_pretrain', type=bool, default=False)
    parser.add_argument('--pretrain_dir', type=str, default="snapshots/Epoch24.pth")
    config = parser.parse_args()
    if not os.path.exists(config.snapshots_folder):
        os.mkdir(config.snapshots_folder)

    train(config)

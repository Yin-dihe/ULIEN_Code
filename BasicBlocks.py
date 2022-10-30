import torch.nn as nn
import torch
import torch.nn.functional as F


# SE BLOCK
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Residual_Block_Enhance(nn.Module):
    def __init__(self, in_num, out_num, dilation_factor):
        super(Residual_Block_Enhance, self).__init__()
        self.conv1 = (
            nn.Conv2d(in_channels=in_num, out_channels=out_num, kernel_size=3, stride=1, padding=dilation_factor,
                      dilation=dilation_factor, groups=1, bias=False))
        # self.in1 = nn.BatchNorm2d(out_num)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = (
            nn.Conv2d(in_channels=out_num, out_channels=out_num, kernel_size=3, stride=1, padding=dilation_factor,
                      dilation=dilation_factor, groups=1, bias=False))
        # self.in2 = nn.BatchNorm2d(out_num)
        self.se = SELayer(channel=out_num)

    def forward(self, x):
        identity_data = x
        output = self.relu((self.conv1(x)))
        output = (self.conv2(output))

        # USE SE BLOCK
        se = self.se(output)
        output = se + identity_data
        return output


def upsample(x):
    b, c, h, w = x.shape[0:4]
    avg = nn.AvgPool2d([4, 4], stride=4)
    output = avg(x).view(b, c, -1)
    return output


class nonlocalblock(nn.Module):
    def __init__(self, channel=64, avg_kernel=2):
        super(nonlocalblock, self).__init__()
        self.channel = channel // 4
        self.theta = nn.Conv2d(channel, self.channel, 1)
        self.phi = nn.Conv2d(channel, self.channel, 1)
        self.g = nn.Conv2d(channel, self.channel, 1)
        self.conv = nn.Conv2d(self.channel, channel, 1)
        self.avg = nn.AvgPool2d([avg_kernel, avg_kernel], stride=avg_kernel)

    def forward(self, x):
        H, W = x.shape[2:4]
        # u = F.interpolate(x,scale_factor=0.5)
        # avg = nn.AvgPool2d([2,2],stride=2)
        u = self.avg(x)
        b, c, h, w = u.shape[0:4]
        # avg = nn.AvgPool2d(5,stride=1,padding=2)
        # temp_x = torch.cat((x,avg(x)),dim=1)
        # avg = nn.AvgPool2d(11,stride=1,padding=5)
        # temp_x = torch.cat((temp_x,avg(x)),dim=1)
        theta_x = self.theta(u).view(b, self.channel, -1).permute(0, 2, 1)
        phi_x = self.phi(u)
        phi_x = upsample(phi_x)
        g_x = self.g(u)
        g_x = upsample(g_x).permute(0, 2, 1)

        # .view(batch_size,self.channel,-1)
        # theta_x = theta_x.permute(0,2,1)
        theta_x = torch.matmul(theta_x, phi_x)
        theta_x = F.softmax(theta_x, dim=-1)

        y = torch.matmul(theta_x, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(b, self.channel, h, w)
        y = self.conv(y)
        y = F.interpolate(y, size=[H, W])
        return y


class Residual_Block_Enhance_non_local(nn.Module):
    def __init__(self, in_num, out_num, dilation_factor):
        super(Residual_Block_Enhance_non_local, self).__init__()
        self.conv1 = (
            nn.Conv2d(in_channels=in_num, out_channels=out_num, kernel_size=3, stride=1, padding=dilation_factor,
                      dilation=dilation_factor, groups=1, bias=False))
        # self.in1 = nn.BatchNorm2d(out_num)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = (
            nn.Conv2d(in_channels=out_num, out_channels=out_num, kernel_size=3, stride=1, padding=dilation_factor,
                      dilation=dilation_factor, groups=1, bias=False))
        # self.in2 = nn.BatchNorm2d(out_num)
        self.nonlocalblock1= nonlocalblock(channel=out_num)

    def forward(self, x):
        identity_data = x
        output = self.relu((self.conv1(x)))
        output = (self.conv2(output))

        # USE SE BLOCK
        nonlocalblock1 = self.nonlocalblock1(output)
        output = nonlocalblock1 + identity_data
        return output

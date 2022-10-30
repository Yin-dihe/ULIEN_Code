import torch
import torch.nn as nn
import torch.nn.functional as F
import BasicBlocks

class ULIEN(nn.Module):

		def __init__(self):
			super(ULIEN, self).__init__()

			self.relu = nn.ReLU(inplace=True)
			self.relu1 = nn.LeakyReLU(inplace=True)

			self.e_conv1 = nn.Conv2d(4, 64, 3, 1, 1, bias=True)
			# BasicBlocks.Residual_Block_Enhance  RB残差块
			self.residual_group = BasicBlocks.Residual_Block_Enhance(64, 64, 1)

			# self.residual_group2 = BasicBlocks.Residual_Block_Enhance(64, 64, 1)
			# self.residual_group3 = BasicBlocks.Residual_Block_Enhance(64, 64, 1)
			self.se = BasicBlocks.SELayer(64)
			self.e_conv2 = nn.Conv2d(64, 24, 3, 1, 1)
			self.e_conv3 = nn.Conv2d(24, 3, 3, 1, 1)

		def forward(self, x, gray):
			x1 = self.relu1(self.e_conv1(torch.cat((x, gray), 1)))
			# x1 = self.e_conv1(torch.cat((x, gray), 1))
			res1 = self.residual_group(x1)
			# group_cat = self.se(torch.cat([res1, res2, res3], 1))
			x2 = self.relu1(self.e_conv2(res1))
			# x2 = self.e_conv2(res1)
			output = F.sigmoid(self.e_conv3(x2))
			output = output + gray
			enhance_image = torch.pow(x, output)
			return x,enhance_image,output

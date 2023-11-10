# @Time:2023/11/9 23:02 
# @Author:andrew
# @email:zengjunjine1026@163.com
# @File:nn_conv2d.py 
# @veision 1.0
import torch
import torchvision
import ssl
import ssl

from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor

ssl._create_default_https_context = ssl._create_unverified_context
dataset = torchvision.datasets.CIFAR10("../data", train=False, transform=ToTensor(), download=True)
dataloder = DataLoader(dataset, batch_size=64)


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, input):
        output = self.conv1(input)
        return output


tudui = Tudui()
writer = SummaryWriter('../logs')
i = 0
for data, target in dataloder:
    output = tudui(data)
    writer.add_image("input", data, i, dataformats='NCHW')
    output = torch.reshape(output, shape=(-1, 3, 30, 30))
    writer.add_image("output", output, i, dataformats='NCHW')
    i = i + 1

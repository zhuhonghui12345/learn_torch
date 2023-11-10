# @Time:2023/11/9 21:30 
# @Author:andrew
# @email:zengjunjine1026@163.com
# @File:nn_conv2d.py 
# @veision 1.0
import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torchvision.transforms import ToTensor

dataset=torchvision.datasets.CIFAR10('../data',train=False,download=True,transform=ToTensor())
input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]], dtype=torch.float32)

print(input.shape)
input = torch.reshape(input, (-1, 1, 5, 5))


print(input.shape)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3,ceil_mode=True)

    def forward(self, input):
        output = self.maxpool1(input)
        return output


tudui = Tudui()
output = tudui(input)
print(output)
# @Time:2023/11/9 22:49 
# @Author:andrew
# @email:zengjunjine1026@163.com
# @File:nn_relu.py 
# @veision 1.0
import torch
from torch import nn
from torch.nn import ReLU

input = torch.tensor([[1, -0.5], [-1, 3]])

input = torch.reshape(input, [-1, 1, 2, 2])
print(input.shape)


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.relu1 = ReLU()

    def forward(self, input):
        output = self.relu1(input)
        return output


tudui = Tudui()
output = tudui(input)
print(output)

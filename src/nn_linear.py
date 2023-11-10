# @Time:2023/11/10 11:19 
# @Author:andrew
# @email:zengjunjine1026@163.com
# @File:nn_linear.py 
# @veision 1.0
from torch import nn, flatten
from torch.nn import Linear
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor_size=64)


dataset = CIFAR10(root="../data", train=False, transform=ToTensor(), download=True)
dataloader = DataLoader(dataset, batch

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.linear1 = Linear(196608,10, bias=True)

    def forward(self, input):
        output = self.linear1(input)
        return output


tudui=Tudui()
for img, target in dataloader:
    print(img.shape)
    input = flatten(img)
    print(input.shape)
    output=tudui(input)
    print(output.shape)

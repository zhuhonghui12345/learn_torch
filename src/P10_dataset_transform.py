# @Time:2023/11/6 21:41 
# @Author:andrew
# @email:zengjunjine1026@163.com
# @File:P10_dataset_transform.py 
# @veision 1.0
import torchvision

train_set = torchvision.datasets.CIFAR10(root='./dataset', train=True, download=True)
test_set = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=True)

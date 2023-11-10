# @Time:2023/11/6 17:30 
# @Author:andrew
# @email:zengjunjine1026@163.com
# @File:P9_Transform.py 
# @veision 1.0
from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

image_path = '/dataset/train/ants/0013035.jpg'
img = Image.open(image_path)

writer = SummaryWriter('../logs')

tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)
print(tensor_img)


writer.add_image('Tensor_img',tensor_img)
writer.close()
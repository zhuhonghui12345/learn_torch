# @Time:2023/11/6 20:39 
# @Author:andrew
# @email:zengjunjine1026@163.com
# @File:P10_UsefulTransforms.py 
# @veision 1.0

from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter("../logs")
img = Image.open("/dataset/train/ants/0013035.jpg")
print(img)

# Totensor
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image("ToTensor", img_tensor)

# Normalization
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([1, 4, 6], [3, 7, 1])
# trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_norm = trans_norm(img_tensor)
writer.add_image("Normalize", img_norm, 2)

print(img_norm[0][0][0])

# Resize
print(img.size)
trans_resize = transforms.Resize((512, 512))
img_resize = trans_resize(img)
img_resize = trans_totensor(img_resize)

writer.add_image("img_resize", img_resize, 0)
print(img_resize)

# Compose-resize-2
trans_resize_2 = transforms.Resize(512)
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])
img_resize_2 = trans_compose(img)
writer.add_image("img_resize", img_resize_2, 1)

# RandomCrop
trans_random = transforms.RandomCrop((400,700))
trrans_compose_2 = transforms.Compose([trans_random, trans_totensor])
for i in range(10):
    img_crop = trrans_compose_2(img)
    writer.add_image("RandomCropHW", img_crop, i)

writer.close()

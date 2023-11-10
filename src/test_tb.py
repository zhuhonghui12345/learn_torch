# @Time:2023/11/6 13:02 
# @Author:andrew
# @email:zengjunjine1026@163.com
# @File:test_tb.py 
# @veision 1.0
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter("../logs")
image_path = '/dataset/train/ants/0013035.jpg'
img = Image.open(image_path)
img_array = np.array(img)
writer.add_image("test", img_array, 1, dataformats='HWC')
# writer.add_image()
for i in range(100):
    writer.add_scalar("y=2x", 2 * i, i)
writer.close()

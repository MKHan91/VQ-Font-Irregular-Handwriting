import os
import os.path as osp
from glob import glob

from PIL import ImageFont, Image, ImageDraw
from fontTools.ttLib import TTFont

train_font_dir = r"/home/dev/VQ-Font/datasets/train_font_image"
font_dict = {}
for folderName in os.listdir(train_font_dir):
    if 'UhBee' not in folderName: continue
    
    chars = [name[:-4] for name in os.listdir(osp.join(train_font_dir, folderName))]
    font_dict[folderName] = chars
a=1
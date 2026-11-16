import os.path as osp
import os
from glob import glob

train_font_image_dir = r"/home/sona/VQ-Font-Irregular-Handwriting/datasets/train_font_image"
val_font_image_dir = r"/home/sona/VQ-Font-Irregular-Handwriting/datasets/valid_font_image"

train_save_path = r"/home/sona/VQ-Font-Irregular-Handwriting/vqgan_data/train_custom_v2.txt"
with open(train_save_path, "w", encoding='utf-8') as f:
    for folderName in os.listdir(train_font_image_dir):
        if folderName == "reference_images": continue
        
        image_paths = glob(osp.join(train_font_image_dir, folderName, "*.png"))
        
        for image_path in image_paths:
            f.write(image_path + '\n')
            
            
val_save_path = r"/home/sona/VQ-Font-Irregular-Handwriting/vqgan_data/valid_custom_v2.txt"
with open(val_save_path, "w", encoding='utf-8') as f:
    for folderName in os.listdir(val_font_image_dir):
        image_paths = glob(osp.join(val_font_image_dir, folderName, "*.png"))
        
        for image_path in image_paths:
            f.write(image_path + '\n')
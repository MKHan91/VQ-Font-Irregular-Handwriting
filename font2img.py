from PIL import Image,ImageDraw,ImageFont
import matplotlib.pyplot as plt
import os
import os.path as osp
import numpy as np
import pathlib
import argparse
from fontTools.ttLib import TTFont


parser = argparse.ArgumentParser(description='Obtaining characters from .ttf')
parser.add_argument('--ttf_path', type=str, default='/home/dev/Project/VQ-Font/datasets/additional_train_font_ttf',help='ttf directory')
parser.add_argument('--chara', type=str, default='./total_korean.txt',help='characters')
parser.add_argument('--save_path', type=str, default='/home/dev/Project/VQ-Font/datasets/train_font_image',help='images directory')
parser.add_argument('--img_size', type=int, default=128, help='The size of generated images')
parser.add_argument('--chara_size', type=int, default=120, help='The size of generated characters')
args = parser.parse_args()

file_object = open(args.chara,encoding='utf-8')   
try:
	characters = file_object.read()
finally:
    file_object.close()


def draw_single_char(ch, font, canvas_size):
    # img = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))
    img = Image.new("L", (canvas_size, canvas_size), 255)
    draw = ImageDraw.Draw(img)
    
    # 글자 크기(bounding box) 계산
    bbox = draw.textbbox((0, 0), ch, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    
    # 중앙 정렬 오프셋 계산
    x_offset = (canvas_size - w) // 2 - bbox[0]
    y_offset = (canvas_size - h) // 2 - bbox[1]
    
    draw.text((x_offset, y_offset), ch, 0, font=font)
    return img

def draw_example(ch, src_font, canvas_size):
    src_img = draw_single_char(ch, src_font, canvas_size)
    example_img = Image.new("L", (canvas_size, canvas_size), 255)
    example_img.paste(src_img, (0, 0))
    return example_img



def get_char_list_from_ttf(font_file):
    f_obj = TTFont(font_file)
    m_dict = f_obj.getBestCmap()

    unicode_list = []
    for key, uni in m_dict.items():
        unicode_list.append(key)

    char_list = [chr(ch_unicode) for ch_unicode in unicode_list]
    return char_list


def font_has_char(ttf_path, ch):
    font = TTFont(ttf_path)
    cmap = font["cmap"].getBestCmap()
    return ord(ch) in cmap


def get_hangle_txt():
    start, end = 0xAC00, 0xD7A3

    with open("total_korean.txt", "w", encoding="utf-8") as f:
        for code in range(start, end + 1):
            f.write(chr(code))

    print(f"총 {end - start + 1}개의 글자를 저장했습니다.")
    
    
if __name__ == "__main__":
    data_root = pathlib.Path(args.ttf_path)

    all_image_paths = list(data_root.glob('*.*'))  # *.ttf TTF
    all_image_paths = [str(path) for path in all_image_paths]
    total_num = len(all_image_paths)

    os.makedirs(args.save_path, exist_ok=True)
        
    seq = list()
    for idx, (label, item) in enumerate(zip(range(len(all_image_paths)),all_image_paths)):
        print("{} / {} ".format(idx, total_num), item)
        
        src_font = ImageFont.truetype(item, size=args.chara_size)
        font_name = item.split('/')[-1].split('.')[0]
        if "UhBee" in font_name:
            font_name = "_".join(font_name.split(' '))
            
        # chars = get_char_list_from_ttf(item)

        img_cnt = 0
        filter_cnt = 0
        for (chara, cnt) in zip(characters, range(len(characters))):
            print(f"\r {chara}", end="")
            
            if not font_has_char(item, ch=chara): continue
            
            img = draw_example(chara, src_font, args.img_size)
            # path_full = osp.join(args.save_path,'id_%d'%(label))
            path_full = osp.join(args.save_path, font_name)
            os.makedirs(path_full, exist_ok=True)
        
            if args.img_size * args.img_size * 3 - np.sum(np.array(img) / 255.) < 100:
                filter_cnt += 1
            else:
                img_cnt += 1
                # img.save(osp.join(path_full, "%05d.png" % (cnt)))
                img.save(osp.join(path_full, f"{chara}.png"))
                
        print(filter_cnt,' characters are missing in this font')
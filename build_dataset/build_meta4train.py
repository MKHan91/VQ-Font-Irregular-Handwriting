
import glob
import json
import os
import os.path as osp
import argparse
import shutil
from tqdm import tqdm
from build_dataset import save_lmdb
import random


def getCharList(root):
    """[get all characters this font exists]

    Args:
        root (string): folder path

    Returns:
        [list]: char list
    """
    charlist = []
    for img_path in (glob.glob(root + '/*.jpg') + glob.glob(root + '/*.png')):
        ch = osp.basename(img_path).split('.')[0]
        charlist.append(ch)
    return charlist


def getMetaDict(font_path_list):
    """[generate a dict to save the relationship between font and its existing characters]
    Args:
        font_path_list (List): [training fonts list]

    Returns:
        [dict]: [description]
    """
    meta_dict = dict()
    print("ttf_path_list:", len(font_path_list))
    for font_path in tqdm(font_path_list):
        font_name = osp.basename(font_path)
        meta_dict[font_name] = {
            "path": font_path,
            "charlist": None
        }
        meta_dict[font_name]["charlist"] = getCharList(font_path)
    return meta_dict


def build_meta4train_lmdb(args):
    # saving directory
    out_dir = osp.join(args.saving_dir, 'meta')
    lmdb_path = osp.join(args.saving_dir, 'lmdb')
    os.makedirs(out_dir, exist_ok=True)
    if osp.exists(lmdb_path):
        shutil.rmtree(lmdb_path)
    os.makedirs(lmdb_path, exist_ok=True)
    
    trainset_dict_path = osp.join(out_dir, f'{trainset_dict}.json')
    # content_font = args.content_font
    
    #===================================================================#
    dict_save_path = osp.join(out_dir, f"{trainset_ori_meta}.json")
    font_chosen = []
    for font_name in os.listdir(args.train_font_dir):
        if font_name == "reference_images": continue
        font_chosen.append(osp.join(args.train_font_dir, font_name))
        
    font_chosen += glob.glob(args.val_font_dir + "/*")
    font_chosen = list(set(font_chosen))

    print('num of fonts: ', len(font_chosen))
    
    # add content font
    if args.content_font not in font_chosen:
        font_chosen.append(args.content_font)
        
    out_dict = getMetaDict(font_chosen)
    with open(dict_save_path, 'w') as fout:
        json.dump(out_dict, fout, indent=4, ensure_ascii=False)
       
    valid_dict = save_lmdb(lmdb_path, out_dict)
    with open(trainset_dict_path, "w") as f:
        json.dump(valid_dict, f, indent=4, ensure_ascii=False)
    
        
        
def build_train_meta(args):
    train_meta_root = osp.join(args.saving_dir, 'meta')
    # content
    # content_font_name = osp.basename(args.content_font) #'kaiti_xiantu'
    
#==============================================================================#
    save_path = osp.join(train_meta_root, f"{train_json}.json")
    meta_file = osp.join(train_meta_root, f"{trainset_dict}.json")

    with open(meta_file, 'r') as f_in:
        original_meta = json.load(f_in)
    with open(args.seen_unis_file) as f:
        seen_unis = json.load(f)
    with open(args.unseen_unis_file) as f:
        unseen_unis = json.load(f)

    # all font names
    all_style_fonts = list(original_meta.keys())

    unseen_ttf_dir = args.val_font_dir #"/ssd1/tanglc/cvpr_image/cu_font_122_val"
    unseen_ttf_list = [osp.basename(x) for x in glob.glob(unseen_ttf_dir + '/*')]
    unseen_style_fonts = [ttf for ttf in unseen_ttf_list]

    #get font in training set
    train_style_fonts = list(set(all_style_fonts) - set(unseen_style_fonts))

    train_dict = {
        "train": {},
        "avail": {},
        "valid": {}
    }

    for style_font in train_style_fonts:
        avail_unicodes = original_meta[style_font]
        train_unicodes = list(set.intersection(set(avail_unicodes), set(seen_unis)))
        train_dict["train"][style_font] = train_unicodes #list(intersection_unis)
        
    for style_font in all_style_fonts:
        avail_unicodes = original_meta[style_font]
        train_dict["avail"][style_font] = avail_unicodes

        
    print("all_style_fonts:", len(all_style_fonts))
    print("train_style_fonts:", len(train_dict["train"]))
    print("val_style_fonts:", len(unseen_style_fonts))
    print("seen_unicodes: ", len(seen_unis))
    print("unseen_unicodes: ", len(unseen_unis))

    # validation set
    train_dict["valid"] = {
        "seen_fonts":  list(train_dict["train"].keys()),
        "unseen_fonts": unseen_style_fonts,
        "seen_unis": seen_unis,
        "unseen_unis": unseen_unis,
    }  

    with open(save_path, 'w') as fout:
        json.dump(train_dict, fout, ensure_ascii = False, indent = 4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--saving_dir", help="directory where your lmdb file will be saved",
                        default='/home/dev/Project/VQ-Font/datasets/handwrite_dataset_v3')
    parser.add_argument("--content_font", help="root path of the content font images",
                        default='/home/dev/Project/VQ-Font/datasets/content_font_image/NanumBarunpenR')
    parser.add_argument("--train_font_dir", help="root path of the training font images",
                        default='/home/dev/Project/VQ-Font/datasets/train_font_image')
    parser.add_argument("--val_font_dir", help="root path of the validation font images",
                        default='/home/dev/Project/VQ-Font/datasets/valid_font_image')
    parser.add_argument("--seen_unis_file", help="json file of seen characters",
                        default="/home/dev/Project/VQ-Font/build_dataset/train_unis_v3.json")
    parser.add_argument("--unseen_unis_file", help="json file of unseen characters",
                        default="/home/dev/Project/VQ-Font/build_dataset/val_unis_v3.json")
    args = parser.parse_args()
    
    trainset_dict = 'trainset_dict'
    trainset_ori_meta = 'trainset_ori_meta'
    train_json = 'train'


    build_meta4train_lmdb(args)
    build_train_meta(args)

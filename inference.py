import argparse
import json
from pathlib import Path
from glob import glob
import glob
from evaluator import *
from models import generator_dispatch
from sconf import Config
from transform import setup_transforms
from datasets import load_json, get_fixedref_loader
from build_dataset.build_dataset import save_lmdb
import torch
from utils import Logger
from datasets import (load_lmdb, load_json, read_data_from_lmdb,
                      get_comb_trn_loader, get_cv_comb_loaders)

import os
import os.path as osp
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def getCharList(root):
    '''
    getCharList
    '''
    charlist = []
    for img_path in glob.glob(root + '/*.jpg') + glob.glob(root + '/*.png'):
        ch = osp.basename(img_path).split('.')[0]
        charlist.append(ch)
    return charlist


# 참조(Reference) 서체를 통해 추론(Inference) 가능한 모든 콘텐츠(Content) 서체를 찾는다.
def getMetaDict(image_root_list, content_name, cr_mapping):
    meta_dict = dict()
    random_sample_list = []
    for idx, file_path in enumerate(image_root_list):
        font_name = osp.basename(file_path)
        if font_name == content_name:
            content_path = file_path
            continue
        all_ch_list = getCharList(file_path)
        meta_dict[font_name] = {
            "path": file_path,
            "charlist": None
        }
        meta_dict[font_name]["charlist"] = all_ch_list

        # 11,172자 전체 생성: 구성요소 필터 없이 cr_mapping의 모든 글자를 추론 대상으로 포함
        # (없는 구성요소는 FixedRefDataset.sample_pair_style()에서 랜덤 대체됨)
        infer_unis = [chr(int(uni, 16)) for uni in cr_mapping.keys()]
            
    meta_dict[content_name] = {
        "path": content_path,
        "charlist": None
    }
    meta_dict[content_name]["charlist"] = infer_unis
    return meta_dict


def build_meta4build_dataset(meta_path, img_path_list, content_name, cr_mapping):
    '''
    build_meta4build_dataset
    '''
    out_dict_path = meta_path
    out_dict = getMetaDict(img_path_list, content_name, cr_mapping)
    with open(out_dict_path, 'w') as fout:
        json.dump(out_dict, fout, indent=4, ensure_ascii=False)
    print("dataset meta:", out_dict_path)


def build_dataset4inference(target_img_path, meta_path, content_root, lmdb_path, json_path, cr_mapping):
    '''
    target_img_path: test image directory
    content_root: content image directory
    meta_path: {font1: {root: '/path/', 'charlist': [ch1, ch2, ch3..]}, font2: ....}
    lmdb_path: lmdb directory
    json_path: {font1: [uni1, uni2, uni3, ..], font2: [uni1, uni2, uni3, ...]}
    '''
    ### START
    img_path_list = [target_img_path] + [content_root]
    content_name = osp.basename(content_root)
    build_meta4build_dataset(meta_path, img_path_list, content_name, cr_mapping)
    with open(meta_path) as f:
        fpc_meta = json.load(f)

    # valid_dict:font:[uni,uni,nui]
    valid_dict = save_lmdb(lmdb_path, fpc_meta)
    with open(json_path, "w") as f:
        json.dump(valid_dict, f)
    print("lmdb_path:", lmdb_path)
    print("test meta:", json_path)
    ### END


def build_testmeta4inference(target_name, target_root, content_name="kaiti_xiantu"):
    '''
    build_testmeta4inference
    '''
    meta_file = osp.join(target_root, "dataset_meta.json")
    save_path = osp.join(target_root, "test.json")
    avali_set = {}

    with open(meta_file, 'r') as fin:
        original_meta = json.load(fin)
    target_ori_unis = original_meta[target_name]
    
    # build test meta file
    test_dict = {
        "gen_fonts": [target_name],
        "gen_unis": original_meta[content_name],
        "ref_unis": target_ori_unis
    }
    with open(save_path, 'w') as fout:
        json.dump(test_dict, fout, ensure_ascii=False, indent=4)
    print("test metafile save to ", save_path)
    return save_path, avali_set



def eval_ckpt(args, cfg, target_root, gen):
    logger = Logger.get()
    
    content_name = cfg.content_name
    trn_transform, val_transform = setup_transforms(cfg)
    env = load_lmdb(cfg.data_path)
    env_get = lambda env, x, y, transform: transform(read_data_from_lmdb(env, f'{x}_{y}')['img'])
    # env_get = env_get_hex
    test_meta = load_json(args.test_meta)

    # g_kwargs = cfg.get('g_args', {})
    # g_cls = generator_dispatch()
    # gen = g_cls(1, cfg['C'], 1, cfg, **g_kwargs)
    # if cfg.use_half:
    #     gen.half()
    # gen.cuda()

    # weight = torch.load(args.weight)

    # if "generator_ema" in weight:
    #     weight = weight["generator_ema"]
    # gen.load_state_dict(weight)
    # logger.info(f"load checkpoint from {args.weight}")
    
    writer = None

    evaluator = Evaluator(
        env=env,
        env_get=env_get,
        cfg=cfg,
        logger=logger,
        writer=writer,
        batch_size=cfg["batch_size"],
        transform=val_transform,
        use_half=cfg.use_half
    )

    img_dir = Path(target_root)
    ref_unis = test_meta["ref_unis"]
    gen_unis = test_meta["gen_unis"]
    gen_fonts = test_meta["gen_fonts"]
    target_dict = {f: gen_unis for f in gen_fonts}
    loader = get_fixedref_loader(env=env,
                                 env_get=env_get,
                                 target_dict=target_dict,
                                 ref_unis=ref_unis,
                                 cfg=cfg,
                                 transform=val_transform,
                                 num_workers=cfg.n_workers,
                                 shuffle=False
                                 )[1]

    logger.info("Save CV results to {} ...".format(img_dir))
    saving_folder = evaluator.save_each_imgs(gen, loader, args.img_path, save_dir=target_root, reduction='mean')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_paths", nargs="+", default=["/home/sona/VQ-Font-Irregular-Handwriting/cfgs/custom_finetune.yaml"])
    parser.add_argument("--weight", help="path to weight to evaluate", default="/home/sona/VQ-Font-Irregular-Handwriting/vq_font_results/checkpoints/brush_finetune_v2/last.ckpt")
    parser.add_argument("--content_font", help="path to content font", default="/home/sona/VQ-Font-Irregular-Handwriting/datasets/content_font_image/NanumBarunpenR")
    parser.add_argument("--img_path", help="path of the your test img directory.", default="/home/sona/VQ-Font-Irregular-Handwriting/datasets/train_font_image/reference_images_v2")
    parser.add_argument("--saving_root", help="saving directory.", default="./inference_results/target_style_images")
    args = parser.parse_args()

    cfg = Config(*args.config_paths, default="/home/sona/VQ-Font-Irregular-Handwriting/cfgs/defaults.yaml")
    
    mapping_path = os.environ.get("RK_CR_MAPPING")
    if mapping_path:
        cfg.content_reference_json = mapping_path
        
    # 추론 시 batch_size를 적절히 설정 (VRAM에 맞게 조절)
    cfg.batch_size = 32
    
    
    g_kwargs = cfg.get("g_args", {})
    g_cls = generator_dispatch()
    gen = g_cls(1, cfg.C, 1, cfg, **g_kwargs)
    if cfg.use_half: gen.half()
    gen.cuda()

    weight = torch.load(args.weight, weights_only=False)

    if "generator_ema" in weight:
        weight = weight["generator_ema"]
    elif "generator" in weight:
        weight = weight["generator"]
    gen.load_state_dict(weight)
    gen.eval()
    
    # for target_folder in target_folder_all:
    content_root = args.content_font
    saving_root = args.saving_root
    content_name = osp.basename(content_root)
    target_name = osp.basename(args.img_path)
    target_root = osp.join(saving_root, target_name)
    with open(cfg.content_reference_json, 'r') as f:
        cr_mapping = json.load(f)

    os.makedirs(target_root, exist_ok=True)

    # lmdb directory
    lmdb_path = osp.join(target_root, "lmdb")
    # meta file
    json_path = osp.join(target_root, "dataset_meta.json")
    meta_path = osp.join(target_root, "ori_dataset_meta.json")

    build_dataset4inference(args.img_path, meta_path, content_root, lmdb_path, json_path, cr_mapping)

    ###test.json
    save_path, avail = build_testmeta4inference(target_name, target_root, content_name)
    args.test_meta = save_path

    cfg.content_name = content_name
    cfg.data_path = lmdb_path

    print("======> Test Params:")
    print("test meta:", args.test_meta)
    print("content font:", cfg.content_name)
    print("lmdb datasets:", cfg.data_path)

    with torch.no_grad():
        eval_ckpt(args, cfg, target_root, gen)
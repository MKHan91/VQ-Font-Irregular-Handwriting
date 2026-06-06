import sys
import os
import os.path as osp
import shutil
import cv2
from pathlib import Path
import argparse
import torch
import torch.optim as optim
import numpy as np
from sconf import Config, dump_args
import utils
from utils import Logger
from transform import setup_transforms
from models import generator_dispatch, disc_builder
from datasets import (load_lmdb, load_json, read_data_from_lmdb,
                      get_comb_trn_loader, get_cv_comb_loaders)
from trainer import load_checkpoint, load_checkpoint_torch, CombinedTrainer
from evaluator import Evaluator
from models.modules import weights_init
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
import torch.backends.cudnn as cudnn
from models.vq import VectorQuantizedVAE
import torch
import  collections
from taming.modules.discriminator.model import NLayerDiscriminator

torch.autograd.set_detect_anomaly(True)

def cleanup():
    dist.destroy_process_group()

def is_main_worker(gpu):
    return (gpu <= 0)

def train_ddp(gpu, args, cfg, world_size):
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:" + str(cfg.port),
        world_size=world_size,
        rank=gpu,
    )
    cfg.batch_size = cfg.batch_size // world_size
    train(args, cfg, ddp_gpu=gpu)
    cleanup()

# region - config
def setup_args_and_config():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--name", default="vq_font_v4.0")
    parser.add_argument("--name", default="brush_finetune_v2")
    # parser.add_argument("--config_paths", nargs="+", default=["/home/dev/Project/VQ-Font/cfgs/custom.yaml"])
    parser.add_argument("--config_paths", nargs="+", default=["/home/dev/Project/VQ-Font/cfgs/custom_finetune.yaml"])
    parser.add_argument("--vq_gan_resume", default="/home/dev/Project/VQ-Font/taming/experiments/checkpoints/2026-01-11T10-58-28_custom_vqgan/epoch=000781.ckpt")
    parser.add_argument("--vq_font_resume", default="./vq_font_results/checkpoints/vq_font_v4.0/last.ckpt")
    parser.add_argument("--use_unique_name", default=False, action="store_true", help="whether to use name with timestamp")

    args, left_argv = parser.parse_known_args()
    assert not args.name.endswith(".yaml")
    cfg = Config(*args.config_paths, default="cfgs/defaults.yaml",
                 colorize_modified_item=True)
    cfg.argv_update(left_argv)

    if cfg.use_ddp: cfg.n_workers = 0

    cfg.work_dir = Path(cfg.work_dir)
    cfg.work_dir.mkdir(parents=True, exist_ok=True)

    if args.use_unique_name:
        timestamp = utils.timestamp()
        unique_name = f"{timestamp}_{args.name}"
    else:
        unique_name = args.name

    cfg.unique_name = unique_name
    cfg.name = args.name
    
    args.ckptdir = cfg.work_dir / "checkpoints" / cfg.name
    args.codesdir = cfg.work_dir / "codes" / cfg.name
    args.logdir = cfg.work_dir / "logs" / cfg.name
    args.runsdir = cfg.work_dir / "runs" / cfg.name
    
    args.ckptdir.mkdir(parents=True, exist_ok=True)
    args.codesdir.mkdir(parents=True, exist_ok=True)
    args.logdir.mkdir(parents=True, exist_ok=True)
    args.runsdir.mkdir(parents=True, exist_ok=True)

    if cfg.save_freq % cfg.val_freq:
        raise ValueError("save_freq has to be multiple of val_freq.")
    
    # items = os.listdir(os.getcwd())
    # for item in items:
    #     if item == "vq_font_results" or item == '.git' or item == '.gitignore': continue
            
    #     src = osp.join(os.getcwd(), item)
    #     if item == 'datasets':
    #         os.makedirs(args.codesdir / item, exist_ok=True)
    #         shutil.copy2(osp.join(src, "__init__.py"), args.codesdir/item)
    #         shutil.copy2(osp.join(src, "dataset_transformer.py"), args.codesdir/item)
    #         shutil.copy2(osp.join(src, "datautils.py"), args.codesdir/item)
    #         shutil.copy2(osp.join(src, "lmdbutils.py"), args.codesdir/item)
    #         continue
        
    #     if not osp.isdir(src):
    #         shutil.copy2(src, args.codesdir)
    
    return args, cfg


# region - train
def train(args, cfg, ddp_gpu=-1):
    cfg.gpu = 0
    torch.cuda.set_device(cfg.gpu)
    cudnn.benchmark = True

    log_path = args.logdir / f"{cfg.name}.log"
    logger = Logger.get(file_path=log_path, level="info", colorize=True)

    image_scale = 0.6
    writer_path = cfg.work_dir / "runs" / cfg.unique_name
    
    writer = utils.TBWriter(writer_path, scale=image_scale)

    args_str = dump_args(args)
    logger.info("Run Argv:\n> {}".format(" ".join(sys.argv)))
    logger.info(f"Args:\n{args_str}")
    logger.info(f"Configs:\n{cfg.dumps()}")
    logger.info(f"Unique name: {cfg.unique_name}")
    logger.info("Get dataset ...")

    trn_transform, val_transform = setup_transforms(cfg)

    env = load_lmdb(cfg.data_path)
    env_get = lambda env, x, y, transform: transform(read_data_from_lmdb(env, f'{x}_{y}')['img'])
    data_meta = load_json(cfg.data_meta)

    trn_dset, trn_loader = get_comb_trn_loader(env, 
                                               env_get,
                                               cfg,
                                               data_meta["train"],
                                               trn_transform,
                                               num_workers=cfg.n_workers,
                                               shuffle=False)
    if is_main_worker(ddp_gpu):
        cv_loaders = get_cv_comb_loaders(env,
                                         env_get,
                                         cfg,
                                         data_meta,
                                         val_transform,
                                         num_workers=cfg.n_workers,
                                         shuffle=False)
    else:
        cv_loaders = get_cv_comb_loaders(env,
                                         env_get,
                                         cfg,
                                         data_meta,
                                         val_transform,
                                         num_workers=cfg.n_workers,
                                         shuffle=False)


    logger.info("Build model ...")
    g_kwargs = cfg.get("g_args", {})

    g_cls = generator_dispatch()
    gen = g_cls(C_in=1, C=cfg.C, C_out=1, cfg=cfg, **g_kwargs)
    gen.cuda()
    
    if cfg.gan_w > 0.:
        d_kwargs = cfg.get("d_args", {})
        disc = disc_builder(cfg.C, trn_dset.n_fonts, trn_dset.n_content_chars, **d_kwargs)
        disc.cuda()
        disc.apply(weights_init(cfg.init))        
        
        # disc = NLayerDiscriminator(input_nc=1,n_layers=3,use_actnorm=False,ndf=64)
        # cp=torch.load('/home/yms/taming-transformers/vqgan/1024_16*16_vaecoder.ckpt')
        # dt = collections.OrderedDict()
        # for k,v in cp['state_dict'].items():
        #     if k.startswith('loss.discriminator') :
        #         dt[k[19:]]=v
        # disc.load_state_dict(dt)

        # disc = NLayerDiscriminator(input_nc=1,n_layers=3,use_actnorm=False,ndf=64).cuda()
        # # disc.apply(weights_init(cfg.init))        
        # cp=torch.load('/data/yms/formerfont/1024_16*16_vaecoder.ckpt')
        # dt = collections.OrderedDict()
        # for k,v in cp['state_dict'].items():
        #     if k.startswith('loss.discriminator') :
        #         dt[k[19:]]=v
        # disc.load_state_dict(dt)
    else:
        disc = None
    
    
    # # ✅ 2단계: train.py에서 Generator 로드 후 추가
    # for name, param in gen.named_parameters():
    #     if 'component_encoder' in name or 'content_encoder' in name:
    #         param.requires_grad_(False)  # 인코더 고정
    #     if 'former' in name or 'mlp_head' in name or 'vqgan.decoder' in name:
    #         param.requires_grad_(True)   # 디코더/트랜스포머만 학습
    
    # 인코더 동결: 가중치 로드 전에 모델만 로드하거나, 로드 후 옵티마이저를 새로 생성해야 합니다.
    for name, param in gen.named_parameters():
        if 'component_encoder' in name or 'content_encoder' in name:
            param.requires_grad_(False)

    # 체크포인트에서 모델 가중치만 로드 (optimizer 상태는 로드하지 않음)
    st_step = 1
    if args.vq_font_resume:
        _, loss = load_checkpoint_torch(args.vq_font_resume, gen, disc, device=device)
        loss = f"{loss:7.3f}" if loss is not None else "N/A"
        logger.info(f"Resumed generator weights from {args.vq_font_resume} (Loss {loss})")
        # 파인튜닝은 step 1부터 새로 시작
        st_step = 1
    else:
        if args.vq_gan_resume:
            _, loss = load_checkpoint_torch(args.vq_gan_resume, gen, disc, device=device, load_codebook_only=True)
            loss = f"{loss:7.3f}" if loss is not None else "N/A"
            logger.info(f"Resumed VQGAN checkpoint from {args.vq_gan_resume} (Loss {loss})")
            st_step = 1

    # 옵티마이저는 requires_grad=True인 파라미터만 포함하여 생성
    g_optim = optim.Adam(filter(lambda p: p.requires_grad, gen.parameters()), lr=cfg.g_lr)
    d_optim = optim.Adam(disc.parameters(), lr=cfg.d_lr) if disc is not None else None
    gen_scheduler = optim.lr_scheduler.StepLR(g_optim,step_size=cfg['step_size'],gamma=cfg['g_gamma'])
    dis_scheduler = optim.lr_scheduler.StepLR(d_optim,step_size=cfg['step_size'],gamma=cfg['d_gamma']) if disc is not None else None
    # gen_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(g_optim, T_0=cfg['step_size'], T_mult=1)
    # dis_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(d_optim, T_0=cfg['step_size'], T_mult=1) if disc is not None else None 

    # st_step = 1
    # if args.vq_font_resume:
    #     st_step, loss = load_checkpoint(args.vq_font_resume, gen, g_optim, gen_scheduler, disc, d_optim, dis_scheduler, device=device)
    #     loss = f"{loss:7.3f}" if loss is not None else "N/A"
    #     logger.info(f"Resumed checkpoint from {args.vq_font_resume} (Step {st_step-1}, Loss {loss})" )
        
    # else:
    #     if args.vq_gan_resume:
    #         # st_step, loss = load_checkpoint(args.resume, gen, disc, g_optim, d_optim, gen_scheduler, dis_scheduler)
    #         st_step, loss = load_checkpoint(args.vq_gan_resume, gen, g_optim, gen_scheduler, disc, d_optim, dis_scheduler, 
    #                                         device=device, load_codebook_only=True)
    #         loss = f"{loss:7.3f}" if loss is not None else "N/A"
    #         logger.info(f"Resumed checkpoint from {args.vq_gan_resume} (Step {st_step-1}, Loss {loss})" )
            
    #         if cfg.overwrite:
    #             st_step = 1
    #         else:
    #             pass

    evaluator = Evaluator(env,
                          env_get,
                          cfg,
                          logger,
                          writer,
                          cfg.batch_size,
                          val_transform,
                        #   content_font,
                          use_half=cfg.use_half
                          )
    
    trainer = CombinedTrainer(ddp_gpu, gen, disc, g_optim, d_optim, gen_scheduler, dis_scheduler,
                      logger, evaluator, cv_loaders, cfg, writer)
    trainer.train(trn_loader, st_step, cfg["iter"])


def main():
    args, cfg = setup_args_and_config()
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    if cfg.use_ddp:
        ngpus_per_node = torch.cuda.device_count()
        world_size = ngpus_per_node 
        mp.spawn(train_ddp, nprocs=ngpus_per_node, args=(args, cfg, world_size))
    else:
        train(args, cfg)


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    main()

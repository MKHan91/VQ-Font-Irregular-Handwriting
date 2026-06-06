

import torch
from collections import OrderedDict


def load_checkpoint(path, gen, g_optim, g_scheduler, disc, d_optim, d_scheduler, device, load_codebook_only=False):
    """
    load_checkpoint
    """
    # ckpt = torch.load(path,map_location={'cuda:1': 'cuda:0'})
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if load_codebook_only:
        state_dict = ckpt.get("state_dict", ckpt)  # Lightning ckpt와 일반 ckpt 모두 지원
        gen.load_state_dict({k.replace("generator.", ""): v for k, v in state_dict.items() if k.startswith("generator.")}, strict=False)
        print("✅ Full generator weights loaded.")
        
        disc_keys = [k for k in state_dict.keys() if k.startswith("discriminator.")]
        dt = OrderedDict()
        for k in disc_keys:
            dt[k.replace("discriminator.", "")] = state_dict[k]
        disc.load_state_dict(dt, strict=False)
        print("✅ Discriminator weights loaded.")
        
        st_epoch = ckpt.get('epoch', 0) + 1
        loss = ckpt.get('loss', None)
        
    else:
        gen.load_state_dict(ckpt['generator'])
        g_optim.load_state_dict(ckpt['optimizer'])
        g_scheduler.load_state_dict(ckpt['g_scheduler'])
    
        disc.load_state_dict(ckpt['discriminator'])
        d_optim.load_state_dict(ckpt['d_optimizer'])
        d_scheduler.load_state_dict(ckpt['d_scheduler'])
        
        st_epoch = ckpt['epoch'] + 1
        loss = ckpt['loss']

    return st_epoch, loss


def load_checkpoint_torch(ckpt_path, gen, disc, device='cpu', load_codebook_only=False):
    """
    PyTorch ckpt 로드용
    - gen: vq-font generator
    - disc: discriminator (optional)
    - load_codebook_only: True면 generator의 codebook만 로드
    
    지원하는 체크포인트 형식:
    1. VQ-Font 자체 저장 형식: {'generator': state_dict, 'generator_ema': state_dict, ...}
    2. Lightning 형식: {'state_dict': {'generator.xxx': tensor, ...}}
    3. Raw state_dict 형식: {'layer.weight': tensor, ...}
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # 체크포인트 형식 판별
    if "generator_ema" in ckpt:
        # VQ-Font 자체 저장 형식 → EMA 가중치 사용 (파인튜닝에 최적)
        gen_state_dict = ckpt["generator_ema"]
        disc_state_dict = ckpt.get("discriminator", None)
        print(f"✅ Detected VQ-Font checkpoint format (using generator_ema)")
    elif "generator" in ckpt and isinstance(ckpt["generator"], (dict, OrderedDict)):
        # VQ-Font 자체 저장 형식 (EMA 없는 경우)
        gen_state_dict = ckpt["generator"]
        disc_state_dict = ckpt.get("discriminator", None)
        print(f"✅ Detected VQ-Font checkpoint format (using generator)")
    elif "state_dict" in ckpt:
        # Lightning 형식
        state_dict = ckpt["state_dict"]
        gen_state_dict = {k.replace("generator.", ""): v for k, v in state_dict.items() if k.startswith("generator.")}
        disc_state_dict = {k.replace("discriminator.", ""): v for k, v in state_dict.items() if k.startswith("discriminator.")}
        print(f"✅ Detected Lightning checkpoint format")
    else:
        # Raw state_dict (모델 가중치만 저장된 경우)
        gen_state_dict = ckpt
        disc_state_dict = None
        print(f"✅ Detected raw state_dict format")

    if load_codebook_only:
        # codebook key만 골라서 로드
        codebook_keys = [k for k in gen_state_dict.keys() if "quantize.embedding" in k or "vqgan.quantize" in k]
        if codebook_keys:
            for k in codebook_keys:
                v = gen_state_dict[k]
                gen.codebook.embedding.data.copy_(v)
            print("✅ Codebook loaded into generator (encoder/decoder untouched).")
        else:
            # VQGAN 관련 가중치만 로드
            vqgan_keys = {k: v for k, v in gen_state_dict.items() if k.startswith("vqgan.")}
            if vqgan_keys:
                gen.load_state_dict(vqgan_keys, strict=False)
                print("✅ VQGAN weights loaded into generator.")
            else:
                print("⚠️ No codebook/VQGAN keys found in checkpoint.")
    else:
        # 전체 generator state_dict 로드
        gen.load_state_dict(gen_state_dict, strict=False)
        print(f"✅ Full generator weights loaded ({len(gen_state_dict)} keys).")

    # discriminator 로드 (있으면) — shape mismatch 키는 부분 복사
    if disc is not None and disc_state_dict is not None and len(disc_state_dict) > 0:
        model_state = disc.state_dict()
        filtered_state_dict = OrderedDict()
        partial_keys = []
        for k, v in disc_state_dict.items():
            if k in model_state and model_state[k].shape != v.shape:
                # shape 불일치: 작은 쪽 기준으로 부분 복사 (기존 학습 임베딩 보존)
                target = model_state[k].clone()
                slices = tuple(slice(0, min(s1, s2)) for s1, s2 in zip(v.shape, target.shape))
                target[slices] = v[slices]
                filtered_state_dict[k] = target
                partial_keys.append(k)
            else:
                filtered_state_dict[k] = v
        if partial_keys:
            print(f"⚠️ Partially loaded {len(partial_keys)} disc keys (shape mismatch, old weights preserved): {partial_keys}")
        disc.load_state_dict(filtered_state_dict, strict=False)
        print(f"✅ Discriminator weights loaded ({len(filtered_state_dict)}/{len(disc_state_dict)} keys).")

    # PyTorch에서는 optimizer / scheduler는 domain mismatch 가능성 있으므로 **로드하지 않음**
    st_epoch = ckpt.get('epoch', 0) + 1
    loss = ckpt.get('loss', None)

    return st_epoch, loss
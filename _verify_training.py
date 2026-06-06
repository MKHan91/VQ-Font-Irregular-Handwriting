"""학습 체크포인트 검증 스크립트"""
import torch
import os
import sys

ckpt_dir = '/home/dev/Project/VQ-Font/vq_font_results/checkpoints/brush_finetune_v2'

# 1. 파일 크기 확인
print("=" * 60)
print("1. 체크포인트 파일 목록 및 크기")
print("=" * 60)
files = sorted(os.listdir(ckpt_dir))
for f in files:
    path = os.path.join(ckpt_dir, f)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  {f:40s} {size_mb:8.1f} MB")

# 2. last.ckpt 내부 구조 확인
print("\n" + "=" * 60)
print("2. last.ckpt 내부 키 구조")
print("=" * 60)
ckpt = torch.load(os.path.join(ckpt_dir, 'last.ckpt'), map_location='cpu', weights_only=False)
print(f"  Top-level keys: {list(ckpt.keys())}")

if 'generator_ema' in ckpt:
    print(f"  ✅ generator_ema 존재 (추론용 EMA 가중치)")
    print(f"     파라미터 수: {len(ckpt['generator_ema'])} keys")
else:
    print(f"  ❌ generator_ema 없음!")

if 'generator' in ckpt:
    print(f"  ✅ generator 존재")
    print(f"     파라미터 수: {len(ckpt['generator'])} keys")

if 'discriminator' in ckpt:
    print(f"  ✅ discriminator 존재")
    print(f"     파라미터 수: {len(ckpt['discriminator'])} keys")
    # font_emb 확인
    for k, v in ckpt['discriminator'].items():
        if 'font_emb' in k:
            print(f"     font_emb shape: {v.shape}")

if 'optimizer' in ckpt:
    print(f"  ✅ optimizer 존재")
if 'g_scheduler' in ckpt:
    print(f"  ✅ g_scheduler 존재")
if 'd_scheduler' in ckpt:
    print(f"  ✅ d_scheduler 존재")
if 'epoch' in ckpt:
    print(f"  epoch: {ckpt['epoch']}")
if 'loss' in ckpt:
    print(f"  loss: {ckpt['loss']}")

# 3. 초기 vs 최종 체크포인트 비교 (학습이 실제로 진행되었는지)
print("\n" + "=" * 60)
print("3. 학습 진행 여부 확인 (5000 step vs last)")
print("=" * 60)
early_ckpt_path = os.path.join(ckpt_dir, '005000-brush_finetune_v2.ckpt')
if os.path.exists(early_ckpt_path):
    early = torch.load(early_ckpt_path, map_location='cpu', weights_only=False)
    
    # generator_ema 기준으로 비교
    key_to_compare = 'generator_ema' if 'generator_ema' in ckpt else 'generator'
    if key_to_compare in early and key_to_compare in ckpt:
        early_state = early[key_to_compare]
        last_state = ckpt[key_to_compare]
        
        # 몇 개 레이어의 차이 확인
        diffs = []
        for k in list(early_state.keys())[:50]:
            if k in last_state and early_state[k].shape == last_state[k].shape:
                diff = (early_state[k].float() - last_state[k].float()).abs().mean().item()
                diffs.append((k, diff))
        
        diffs.sort(key=lambda x: -x[1])
        print(f"  비교 기준: {key_to_compare}")
        print(f"  상위 변화량 파라미터:")
        for k, d in diffs[:10]:
            print(f"    {k:50s} diff={d:.6f}")
        
        avg_diff = sum(d for _, d in diffs) / len(diffs) if diffs else 0
        print(f"\n  평균 변화량: {avg_diff:.6f}")
        if avg_diff > 0.001:
            print(f"  ✅ 학습이 실제로 진행됨 (가중치 변화 확인)")
        elif avg_diff > 0:
            print(f"  ⚠️ 변화량이 매우 작음 - 학습률이 낮거나 수렴했을 수 있음")
        else:
            print(f"  ❌ 가중치 변화 없음 - 학습이 안 된 것으로 보임")
    del early

# 4. 중간 체크포인트 간 loss 변화 추적 (가능한 경우)
print("\n" + "=" * 60)
print("4. 체크포인트별 저장된 loss 추이")
print("=" * 60)
for f in sorted(files):
    path = os.path.join(ckpt_dir, f)
    try:
        c = torch.load(path, map_location='cpu', weights_only=False)
        loss_val = c.get('loss', 'N/A')
        epoch_val = c.get('epoch', 'N/A')
        print(f"  {f:40s} epoch={epoch_val}, loss={loss_val}")
        del c
    except Exception as e:
        print(f"  {f:40s} 로드 실패: {e}")

# 5. 로그 파일 확인
print("\n" + "=" * 60)
print("5. 학습 로그 확인")
print("=" * 60)
log_dirs = [
    '/home/dev/Project/VQ-Font/vq_font_results/logs/',
]
for log_dir in log_dirs:
    if os.path.exists(log_dir):
        for root, dirs, fnames in os.walk(log_dir):
            for fname in sorted(fnames):
                if 'brush' in fname.lower() or 'v2' in fname.lower() or fname.endswith('.log'):
                    log_path = os.path.join(root, fname)
                    size = os.path.getsize(log_path)
                    print(f"  {log_path} ({size/1024:.1f} KB)")
                    # 마지막 몇 줄 출력
                    if size > 0 and fname.endswith('.log'):
                        with open(log_path, 'r') as lf:
                            lines = lf.readlines()
                            print(f"    총 {len(lines)}줄")
                            print(f"    마지막 5줄:")
                            for line in lines[-5:]:
                                print(f"      {line.rstrip()}")

print("\n✅ 검증 완료")

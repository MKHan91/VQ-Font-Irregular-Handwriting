# VQ-Font 한글 손글씨 Few-Shot 폰트 생성 — 실험 결과

**기반**: VQ-Font (AAAI 2024) · **도메인**: 한글 손글씨/붓글씨 · **작성일**: 2026-06-06

> 사용자가 쓴 **78자** 손글씨 → 한글 완성형 **11,172자** 자동 생성. 본 문서는 학습 과정·중간 산출물·최종 결과를 **그래프와 이미지 중심**으로 정리한다.

---

## 1. 한눈에 보는 결과

| 항목 | 값 |
|---|---|
| 최종 모델 | `brush_finetune_v2/last.ckpt` (≈ 544 MB) |
| 학습 단계 | **Stage 0** VQGAN → **Stage 1** vq_font_v4.0 (76 폰트, 1.5 M iter) → **Stage 2** brush_finetune_v2 (78 자, 50 K iter) |
| 추론 결과 | **11,172 / 11,172 (100 %)** 한글 음절 생성 성공 |
| 파인튜닝 손실 | 1.033 → **0.325** |
| 참조 78자 PSNR / SSIM | **19.83 dB** / **0.852** |
| 78 자 reference 의 자모 커버리지 | 초성 **19/19**, 중성 **21/21**, 종성 **19/27** (8 결손) |

---

## 2. 파이프라인 개요

```
[Stage 0] VQGAN 사전학습             →  코드북 1024 토큰 (한글 시각 prior)
        ↓
[Stage 1] 76 폰트 본학습 (1.5 M it)   →  vq_font_v4.0/last.ckpt
        ↓
[Stage 2] 78 자 파인튜닝 (50 K it)    →  brush_finetune_v2/last.ckpt
        ↓
[Inference] 11,172 자 한글 전체
```

| 항목 | Stage 0 | Stage 1 | Stage 2 |
|---|---|---|---|
| 설정 | `vqgan/custom_vqgan.yaml` | `cfgs/custom.yaml` | `cfgs/custom_finetune.yaml` |
| iter | epoch 782 | **1,500,001** | **50,000** |
| LR(G) | — | `2e-4` | **`2e-5`** |
| `gan_w` / `style_w` | — | 1.0 / 5.0 | **0.1** / 5.0 |
| 동결 영역 | — | — | comp/content encoder, decoder layers ≥ 3 |

---

## 3. 학습 곡선 (TensorBoard 추출)

> 산출: [scripts/export_tb_curves.py](../scripts/export_tb_curves.py) · 인덱스: [tb_curves/INDEX.md](tb_curves/INDEX.md)

### 3.1 Stage 0 — VQGAN 사전학습

| 총 손실 (step) | 재구성 손실 (step) | VQ quantization loss |
|---|---|---|
| ![](tb_curves/plots/vqgan/train_total_loss_step.png) | ![](tb_curves/plots/vqgan/train_rec_loss_step.png) | ![](tb_curves/plots/vqgan/train_quant_loss_step.png) |

| Discriminator 손실 | LPIPS (perceptual) | 검증 재구성 (epoch) |
|---|---|---|
| ![](tb_curves/plots/vqgan/train_disc_loss_step.png) | ![](tb_curves/plots/vqgan/train_p_loss_step.png) | ![](tb_curves/plots/vqgan/val_rec_loss_epoch.png) |

### 3.2 Stage 1 — VQ-Font 본학습 (`vq_font_v4.0`)

| Generator 손실 | Discriminator 손실 | L1 손실 |
|---|---|---|
| ![](tb_curves/plots/vqfont/optimization_loss_generator.png) | ![](tb_curves/plots/vqfont/optimization_loss_discriminator.png) | ![](tb_curves/plots/vqfont/optimization_loss_L1.png) |

| LPIPS 손실 | Cross-Entropy (코드북) | Style Consistency |
|---|---|---|
| ![](tb_curves/plots/vqfont/optimization_loss_lpips.png) | ![](tb_curves/plots/vqfont/optimization_loss_cross_entropy.png) | ![](tb_curves/plots/vqfont/optimization_loss_style_consist.png) |

| 검증 PSNR | 검증 SSIM | LPIPS (Alex/VGG) |
|---|---|---|
| ![](tb_curves/plots/vqfont/evaluation_error_psnr.png) | ![](tb_curves/plots/vqfont/evaluation_error_ssim.png) | ![](tb_curves/plots/vqfont/evaluation_acc_lpips_alex.png) |

### 3.3 Stage 2 — 파인튜닝 (`brush_finetune_v2`)

| Generator 손실 (1.033 → 0.325) | Discriminator 손실 | L1 손실 |
|---|---|---|
| ![](tb_curves/plots/finetune/optimization_loss_generator.png) | ![](tb_curves/plots/finetune/optimization_loss_discriminator.png) | ![](tb_curves/plots/finetune/optimization_loss_L1.png) |

| LPIPS 손실 | Cross-Entropy | Style Consistency |
|---|---|---|
| ![](tb_curves/plots/finetune/optimization_loss_lpips.png) | ![](tb_curves/plots/finetune/optimization_loss_cross_entropy.png) | ![](tb_curves/plots/finetune/optimization_loss_style_consist.png) |

| 검증 PSNR | 검증 SSIM | RMSE |
|---|---|---|
| ![](tb_curves/plots/finetune/evaluation_error_psnr.png) | ![](tb_curves/plots/finetune/evaluation_error_ssim.png) | ![](tb_curves/plots/finetune/evaluation_acc_Rmse.png) |

### 3.4 Stage 1 vs Stage 2 직접 비교 (overlay)

> 같은 축에 두 단계를 겹쳐서 **파인튜닝이 손실/지표를 얼마나 더 끌어내렸는지** 직관적으로 확인.

| Generator 손실 | L1 손실 | Cross-Entropy |
|---|---|---|
| ![](tb_curves/plots/_overlay/optimization_loss_generator.png) | ![](tb_curves/plots/_overlay/optimization_loss_L1.png) | ![](tb_curves/plots/_overlay/optimization_loss_cross_entropy.png) |

| 검증 PSNR | 검증 SSIM | LPIPS (Alex) |
|---|---|---|
| ![](tb_curves/plots/_overlay/evaluation_error_psnr.png) | ![](tb_curves/plots/_overlay/evaluation_error_ssim.png) | ![](tb_curves/plots/_overlay/evaluation_acc_lpips_alex.png) |

---

## 4. 정량 결과

> 산출: [scripts/coverage_quality_analysis.py](../scripts/coverage_quality_analysis.py) · 데이터: [coverage_quality/per_char_metrics.csv](coverage_quality/per_char_metrics.csv)

### 4.1 GT-paired 지표 (참조 78자 한정)

| 지표 | 평균 | 표준편차 | n |
|---|---:|---:|---:|
| PSNR (dB) | **19.83** | 6.59 | 78 |
| SSIM | **0.852** | 0.164 | 78 |

> 참고: 78자 reference에는 한글 19/19 초성, 21/21 중성이 모두 포함되어 있으므로 GT-paired 표본은 곧 "결손 자모 0" 그룹의 부분집합이다.

### 4.2 GT-free 지표 (11,172 자 전체)

| 지표 | 평균 | 표준편차 | 의미 |
|---|---:|---:|---|
| ink_ratio | **0.1450** | 0.0262 | 잉크 픽셀 비율 (글자 굵기 분포) |
| bbox_fill | **0.6681** | 0.0882 | 글자 영역이 셀을 채우는 비율 |
| edge_density | **0.1469** | 0.0276 | sobel 에지 밀도 (획 거칠기 proxy) |
| n_components | **5.31** | 2.26 | 8-연결 컴포넌트 수 (획 끊김 proxy) |

> ink_ratio·edge_density 의 표준편차가 평균 대비 작다는 것은(`std/mean ≈ 0.18~0.19`) 11,172 자 전체에 걸쳐 **굵기·획 거칠기 분포가 일관**되게 유지됨을 시사한다.

### 4.3 자모 커버리지 효과

#### 결손 자모 수별 표본 분포 및 GT-free 지표

| 결손 자모 수 | n | ink_ratio mean | bbox_fill mean | edge_density mean | n_components mean |
|---:|---:|---:|---:|---:|---:|
| 0 | 7,980 | 0.1456 | 0.6692 | 0.1471 | 5.32 |
| 1 | 3,192 | 0.1433 | 0.6655 | 0.1462 | 5.30 |

> 참조 78자가 19/19 초성·21/21 중성을 모두 포함하므로, 결손은 **종성 8개**에서만 발생 → 11,172자 중 28.6%가 "결손 자모=1" 그룹에 속한다. 두 그룹의 GT-free 지표 평균이 거의 동일(`Δink_ratio<0.003`, `Δbbox_fill<0.004`)하다는 것은 모델이 **참조에 없는 종성을 합성할 때도 굵기·면적이 일관**됨을 뜻한다.

| 결손 자모 수별 분포 | PSNR vs 결손 자모 수 | PSNR vs ink_ratio |
|---|---|---|
| ![](coverage_quality/plots/coverage_hist.png) | ![](coverage_quality/plots/psnr_by_coverage.png) | ![](coverage_quality/plots/scatter_psnr_ink.png) |

| ink_ratio by coverage | bbox_fill by coverage | edge_density by coverage |
|---|---|---|
| ![](coverage_quality/plots/ink_ratio_by_coverage.png) | ![](coverage_quality/plots/bbox_fill_by_coverage.png) | ![](coverage_quality/plots/edge_density_by_coverage.png) |

### 4.4 구조 카테고리별 품질

> 산출: [scripts/structure_quality.py](../scripts/structure_quality.py) · 데이터: [structure_quality/](structure_quality/)
> 축: ① 받침 유무(2) ② 중성 형태 horizontal/vertical/mixed(3) ③ 둘의 조합(6).

| 받침 유무 (PSNR) | 중성 형태 (PSNR) | 6 구조 그룹 (PSNR) |
|---|---|---|
| ![](structure_quality/plots/has_jongseong_psnr.png) | ![](structure_quality/plots/jung_shape_psnr.png) | ![](structure_quality/plots/structure_class_psnr.png) |

| 6 구조 그룹 (SSIM) | 6 구조 그룹 (ink_ratio) | 6 구조 그룹 (그룹 크기) |
|---|---|---|
| ![](structure_quality/plots/structure_class_ssim.png) | ![](structure_quality/plots/structure_class_ink_ratio.png) | ![](structure_quality/plots/structure_class_counts.png) |

---

## 5. 정성 결과

### 5.1 비교 그리드 — `[Content | Ref×3 | Generated | (GT)]`

> 산출: [scripts/make_comparison_grid.py](../scripts/make_comparison_grid.py)

**체리픽 (저자 선정)**
![Cherry-picked](grids/cherry.png)

**무작위 24자 (참조 포함 12 + 비포함 12)**
![Random Mixed p1](grids/random_mixed/grid_p01.png)
![Random Mixed p2](grids/random_mixed/grid_p02.png)

### 5.2 11,172자 출력 시트 (sample sheet)

> 산출: [scripts/sample_sheet.py](../scripts/sample_sheet.py)

**KS X 1001 (2,350 자) — p.1**
![](sample_sheet_ks/sheet_p01.png)

**KS X 1001 — p.2**
![](sample_sheet_ks/sheet_p02.png)

> 전체 페이지(KS X 1001 ~3 페이지, 전체 한글 11,172자 ~6 페이지)는 [sample_sheet_ks/](sample_sheet_ks/), [sample_sheet_full/](sample_sheet_full/) 참고.

---

## 6. 데이터셋 요약

| 단계 | 폰트/이미지 | 개수 | 위치 |
|---|---|---:|---|
| Stage 0 | 한글 글자 (스타일 무관) | 수십만 장 | `vqgan_data/train_custom_v2.txt` |
| Stage 1 | Uhbee + 추가 폰트 | **76 종** | `datasets/train_font_ttf/` + `additional_train_font_ttf/` |
| Stage 2 | 사용자 손글씨 | **78 자** | `datasets/train_font_image/reference_images_v2/` |
| 콘텐츠 (공통) | NanumBarunpenR | 1 종 | `datasets/content_font_image/NanumBarunpenR/` |
| 추론 타깃 | 한글 완성형 | **11,172 자** | `cr_mapping_v2.json` |

---

## 7. 부록 — 결과 재현 명령

```bash
# (0) TB 곡선 (Stage 0/1/2 + overlay)
python3 scripts/export_tb_curves.py \
  --run vqgan=taming/experiments/testtube/version_1 \
  --run vqfont=vq_font_results/runs/vq_font_v4.0 \
  --run finetune=vq_font_results/runs/brush_finetune_v2 \
  --overlay vqfont,finetune --smooth 0.9 \
  --out report/tb_curves

# (1) 비교 그리드
python3 scripts/make_comparison_grid.py --chars "가,각,꽃,울,행,헐,훿" \
  --title "Cherry-picked" --out report/grids/cherry.png
python3 scripts/make_comparison_grid.py --auto 24 --auto-strategy mixed \
  --seed 42 --rows-per-figure 12 --out report/grids/random_mixed

# (2) 자모 커버리지 × 품질
python3 scripts/coverage_quality_analysis.py --with-components \
  --out report/coverage_quality

# (3) 구조 카테고리별 품질 (위 CSV 재사용)
python3 scripts/structure_quality.py \
  --metrics-csv report/coverage_quality/per_char_metrics.csv \
  --out report/structure_quality

# (4) 11,172자 sample sheet
python3 scripts/sample_sheet.py --ks-x-1001 \
  --cols 35 --rows-per-page 35 --cell 56 \
  --out report/sample_sheet_ks
python3 scripts/sample_sheet.py \
  --cols 40 --rows-per-page 50 --cell 48 \
  --out report/sample_sheet_full

# (5) PDF 빌드
bash scripts/build_pdf.sh --install            # 최초 1회
bash scripts/build_pdf.sh -i report/report.md -o report/report.pdf
```

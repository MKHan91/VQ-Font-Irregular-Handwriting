# VQ-Font 학습/추론 프로젝트 정리

> 본 문서는 VQ-Font 워크스페이스의 학습·추론 파이프라인을 분석하여,
> **무엇을(목적)**, **왜(문제 정의)**, **어떻게(방법)**, **결과로 무엇을 얻는가(기대 효과)**
> 를 보고서 형식으로 정리한 문서입니다.

---

## 1. 프로젝트 한 줄 요약

> **소수(K-shot)의 손글씨/붓글씨 참조 이미지만으로 한글 11,172자 전체를 동일한 스타일로 자동 생성하는 Few-Shot 한글 폰트 생성 모델 파인튜닝 프로젝트.**

기반 논문: *VQ-Font: Few-Shot Font Generation with Structure-Aware Enhancement and Quantization* (AAAI 2024) — 원본은 한자(Chinese) 도메인이며, 본 워크스페이스는 이를 **한글(KOR) 손글씨/붓글씨 도메인**으로 적응(adaptation)·파인튜닝한 버전입니다.

---

## 2. 풀고자 하는 문제 (Problem Statement)

### 2-1. 도메인 문제
- 한글은 11,172자(완성형 기준)에 달하기 때문에, 새로운 손글씨/캘리그래피 스타일의 폰트 한 벌을 만들려면 **사람이 11,172자를 모두 써야 한다.**
- 사용자가 실제로 쓸 수 있는 양은 일반적으로 **수십~수백 자 수준**(본 프로젝트 reference는 78자)에 불과.
- 따라서 **"사용자가 쓴 몇십 글자 → 11,172자 전체 폰트"** 로 확장하는 자동화가 필요.

### 2-2. 기술적 난제
1. **Few-Shot**: 참조 이미지가 매우 적음 (k-shot, 본 설정 `kshot=3`).
2. **스타일 충실도(Style Fidelity)**: 붓글씨/손글씨는 획의 두께·곡률·먹 번짐 등 변화가 크며, 기존 GAN 기반 방식은 디테일 손실·획 왜곡(distorted strokes)·구성요소 누락이 빈번.
3. **구조 보존(Content Consistency)**: 스타일을 바꿔도 글자가 그 글자로 읽혀야 함 (초·중·종성 구조 보존).
4. **도메인 갭(Domain Gap)**: 합성 글자와 실제 글자 사이의 분포 차이.

### 2-3. VQ-Font 논문이 제안한 모델 아키텍처

> 아래 표는 **VQ-Font 원 논문(AAAI 2024)에서 제안한 모델 구성 요소** 이다. 즉, "모델이 어떻게 생겼는가"를 설명하는 것이며, 본 프로젝트에서 새로 만든 것이 아니라 논문 구조를 그대로 가져와 한글 도메인에 적용한 것이다.
>
> **※ 본 프로젝트의 고유한 기여인 "3단계 학습 전략(Stage 0/1/2)"은 §3에서 별도로 설명한다.** 모델 구조(=§2-3)와 학습 절차(=§3)는 서로 다른 층위의 이야기이므로 혼동하지 말 것.

| 구성 요소 | 역할 | 해결하는 문제 |
|---|---|---|
| **사전학습 VQGAN + Codebook** | 폰트 토큰 prior를 코드북에 압축 저장 | 합성 글자와 실제 글자의 도메인 갭 제거 (token prior refinement) |
| **Component Encoder (다중 스케일)** | 참조 글자의 스타일을 부품(radical) 단위로 인코딩 | 적은 참조로도 부품 단위 스타일 재사용 가능 |
| **Content Encoder** | 콘텐츠 폰트(`NanumBarunpenR`)에서 글자 골격 추출 | 글자 형태 보존 |
| **Structure-aware Transformer (`former`)** | 12종 구조 태그(좌우, 상하, 포위 등) 기반으로 부품 attention 재조정 | 한자/한글의 조합 구조를 반영한 정교한 스타일 매칭 |
| **Memory 모듈** | 참조 글자 스타일 latent 누적·검색 | k-shot 참조로부터 부품 단위 스타일 라이브러리 구축 |
| **Discriminator (font/uni/structure 멀티 헤드)** | 폰트 ID·글자 ID·구조 타입을 동시 판별 | 스타일 정합성, 구조 정합성을 동시에 강제 |

---

## 3. 학습 파이프라인 — 본 프로젝트의 3단계 학습 전략

본 프로젝트의 학습은 **총 3단계(Three-stage)** 로 구성됩니다.

```
[Stage 0] VQGAN 사전학습         →  한글 글자 시각 prior(코드북) 확보
        ↓ 코드북 + 디코더 가중치
[Stage 1] 일반 다중 폰트 본학습   →  76종 폰트로 "한글 스타일 일반화" 능력 학습
        ↓ vq_font_v4.0/last.ckpt
[Stage 2] 타깃 도메인 파인튜닝    →  78자 사용자 손글씨 스타일로 특화 (brush_finetune_v2)
        ↓ brush_finetune_v2/last.ckpt
[Inference] 11,172자 한글 전체 생성
```

> **왜 이렇게 3단계로 나누는가?** — 본 보고서 §3-4에서 상세 설명. 요약: ① 코드북은 도메인 prior이므로 한 번만 학습하면 재사용 가능, ② 78자만으로 처음부터 학습하면 모델이 **글자 구조 자체를 학습하지 못함**, ③ 일반 학습으로 "한글이라는 글자 시스템"을 먼저 익힌 뒤 특정 스타일로 좁히는 것이 적은 데이터로 안정적인 결과를 얻는 표준 전략(=transfer learning).

---

### 3-0. Stage 0 — VQGAN 사전학습 (코드북 구축)
- **실행**: [train_vqgan.sh](train_vqgan.sh) → `python taming/main.py --base vqgan/custom_vqgan.yaml -t True`
- **설정**: [vqgan/custom_vqgan.yaml](vqgan/custom_vqgan.yaml)
  - 입력: 128×128 흑백 (`in_channels: 1`)
  - 코드북: `n_embed: 1024`, `embed_dim: 256`
  - 데이터 리스트: [vqgan_data/train_custom_v2.txt](vqgan_data/train_custom_v2.txt), [vqgan_data/valid_custom_v2.txt](vqgan_data/valid_custom_v2.txt)
- **결과 체크포인트**: `taming/experiments/checkpoints/2026-01-11T10-58-28_custom_vqgan/epoch=000781.ckpt`
- **목적**: 한글 글자 이미지의 **시각적 prior(획·부품 패턴)** 를 1024개 토큰의 코드북으로 압축. Stage 1·2에서 generator의 backbone으로 재사용되며, decoder/post-quant 부분만 일부 fine-tune된다.

---

### 3-1. Stage 1 — 일반 다중 폰트 본학습 (Pre-training of VQ-Font)

> **"여러 한글 폰트로 모델이 한글 글자 구조와 스타일 변형 그 자체를 폭넓게 학습하는 단계."**

#### 학습 데이터 (총 **76종** 폰트)
| 분류 | 위치 | 개수 | 비고 |
|---|---|---:|---|
| **Uhbee 손글씨 폰트** | [datasets/train_font_ttf/](datasets/train_font_ttf) | **67종** | `UhBee_*.ttf` 시리즈 (디자이너 손글씨 기반 무료 폰트) |
| **추가 일반/붓글씨 폰트** | [datasets/additional_train_font_ttf/](datasets/additional_train_font_ttf) | **9종** | BMKIRANGHAERANG, ChosunCentennial, EBS_Hunminjeongeum L/R/SB, KoPubWorld Batang Bold/Light/Medium, Nanum_Brush_Script |
| **콘텐츠(content) 폰트** | `train_font_ttf/NanumBarunpenR.ttf` | (제외) | 골격 공급용. 스타일 학습 대상이 아님 |
| **합계 (스타일 학습 대상)** | | **76종** | — |
| **추론 타깃(78자 사용자 손글씨)** | `train_font_image/reference_images_v2/` | (제외) | Stage 1에서는 **사용하지 않음** — 평가/추론 전용으로 분리 |

#### 학습 설정
- **설정 파일**: [cfgs/custom.yaml](cfgs/custom.yaml)
- **주요 하이퍼파라미터**
  - `iter`: **1,500,001** (충분히 오래)
  - `g_lr`: `2e-4`, `d_lr`: `8e-4`
  - `batch_size`: 32, `kshot`: 3
  - `style_w`: 5.0, `gan_w`: 1.0
- **명령**: `python train.py --name vq_font_v4.0 --config_paths cfgs/custom.yaml --vq_gan_resume <Stage 0 ckpt>`
- **결과 체크포인트**: `vq_font_results/checkpoints/vq_font_v4.0/last.ckpt`

#### 이 단계의 목적
1. **한글 글자 구조 학습**: 11,172자 한글의 초·중·종성 조합, 12종 구조 타입(좌우/상하/포위 등), 부품 위치 관계를 모델이 학습.
2. **스타일 다양성 학습**: 76종이라는 다양한 스타일 분포를 보여줌으로써 모델이 **"스타일을 골격과 분리해 다루는 능력"** 자체를 획득.
3. **부품 단위 latent 공간 구성**: Component Encoder · Memory가 한글 부품 단위의 의미있는 표현을 학습.

---

### 3-2. Stage 2 — 타깃 도메인 파인튜닝 (Few-shot Fine-tuning)

> **"Stage 1에서 학습된 일반 모델을, 사용자가 제공한 78자 손글씨 스타일에 특화시키는 단계."**

#### 학습 데이터
- 타깃 스타일: `datasets/train_font_image/reference_images_v2/` (사용자 손글씨 **78자**)
- 보조 데이터: 일부 Uhbee/일반 폰트 일부를 혼합하여 catastrophic forgetting 방지 가능 (`handwrite_dataset_v3` LMDB)

#### 학습 설정
- **설정 파일**: [cfgs/custom_finetune.yaml](cfgs/custom_finetune.yaml)
- **주요 하이퍼파라미터** (Stage 1 대비 변경)
  - `iter`: **50,000** (짧게)
  - `g_lr`: **`2e-5`** (Stage 1의 1/10 — 사전학습 지식 보존)
  - `style_w`: 5.0 (붓글씨 스타일 강조)
  - `gan_w`: 0.1 (Stage 1보다 낮춰 안정화)
- **명령**: `python train.py --name brush_finetune_v2 --config_paths cfgs/custom_finetune.yaml --vq_font_resume vq_font_results/checkpoints/vq_font_v4.0/last.ckpt`
- **결과 체크포인트**: `vq_font_results/checkpoints/brush_finetune_v2/last.ckpt`

#### 파인튜닝 특이사항 ([train.py](train.py), [models/generator.py](models/generator.py))
- **인코더 동결**: `component_encoder`, `content_encoder`의 `requires_grad=False` — Stage 1에서 학습된 스타일/콘텐츠 인식 능력 보존, 디코더 + Transformer만 미세 조정.
- **VQGAN decoder 일부 unfreeze**: `decoder.layers.{0,1,2}` 및 `post_quant_conv`만 학습 허용 → 새 도메인 디테일 표현 가능.
- **Latent augmentation**: 학습 시 스타일 latent에 σ=0.03 가우시안 노이즈 + L2 정규화 → 78자라는 적은 데이터에서 과적합 완화.

---

### 3-3. **왜 "Stage 1 일반학습 + Stage 2 파인튜닝" 구조가 필수인가?**

| 만약 ~ 만 한다면 | 발생 문제 |
|---|---|
| ❌ **78자만으로 처음부터 학습** | (1) 78자에는 한글 11,172자에 나타나는 모든 부품(초·중·종성 조합)이 포함되어 있지 않음. 모델이 **본 적 없는 부품을 합성하는 일반화 능력**을 학습할 수 없음. (2) 데이터 양이 절대적으로 부족하여 모델 파라미터(수천만)가 수렴 자체를 못 함 → 극심한 overfitting, mode collapse. |
| ❌ **Stage 1만 하고 파인튜닝 생략** | Stage 1 모델은 "76종 평균적인 손글씨"는 잘 만들지만, **특정 사용자의 78자 스타일을 그대로 재현하지는 못함**. 사용자의 개성(획 두께, 곡률, 기울기)이 평균에 묻혀버림. |
| ✅ **Stage 1 + Stage 2 (현재 방식)** | Stage 1에서 **"한글이라는 글자 시스템 + 스타일 다양성"** 의 일반 지식을 학습, Stage 2에서 그 지식 위에 **"특정 사용자 스타일"** 만 얇게 덧붙임. ⇒ 적은 데이터로도 안정적·고품질 결과. |

이는 NLP의 BERT/GPT 사전학습→파인튜닝, 비전의 ImageNet→downstream과 동일한 **Transfer Learning** 패러다임이며, Few-shot Font Generation 분야의 사실상 표준 접근법이다.

추가로, 본 프로젝트의 3단계 분리에는 다음 실용적 이점도 있다.
- **재사용성**: Stage 0(코드북)·Stage 1(일반 모델)은 한 번만 학습해 두면, 새로운 사용자 손글씨가 들어올 때마다 Stage 2(50K iter, 수 시간)만 다시 돌리면 됨.
- **안정성**: 큰 LR로 처음부터 78자에 fitting하면 사전학습 가중치가 망가지지만, Stage 2의 `g_lr=2e-5`(1/10)는 사전학습 지식을 보존하며 점진적으로 적응.
- **디버깅 용이**: 단계별 체크포인트가 분리되어 어느 단계에서 품질이 깨지는지 진단 가능.

---

### 3-4. 학습 시 적용되는 손실(Loss) 구성
[trainer/combined_trainer.py](trainer/combined_trainer.py) 기준 (Stage 1·2 공통):

| Loss | 목적 |
|---|---|
| **L1 (pixel)** | 픽셀 단위 일치 — 글자 형태 정확도 |
| **LPIPS (perceptual)** | 사람이 느끼는 시각적 유사도 |
| **GAN (gen/disc)** | 실제 폰트 분포에 가깝게 — 사실감 |
| **Style loss** | 참조 스타일과의 통계적 일치 (`style_w=5.0` 붓글씨에서 강조) |
| **Feature Matching** | Discriminator 중간 feature 일치 — 학습 안정화 |
| **Cross-entropy (codebook indices)** | 생성 결과의 VQ 토큰 분포가 실제 토큰 분포와 일치 (token prior refinement) |
| **Style Consistency (multi-scale)** | 1.0×/1.2×/0.8× 스케일 스타일 latent 일관성 |

---

## 4. 추론 파이프라인

### 4-1. 진입점
- **스크립트**: [inference.py](inference.py) (실행 도우미 [infer_vqfont.sh](infer_vqfont.sh))
- **기본 인자**
  - `--weight`: `vq_font_results/checkpoints/brush_finetune_v2/last.ckpt`
  - `--content_font`: `datasets/content_font_image/NanumBarunpenR` (글자 골격 공급)
  - `--img_path`: `datasets/train_font_image/reference_images_v2` (사용자 손글씨 78자)
  - `--saving_root`: `./inference_results/target_style_images`

### 4-2. 단계별 흐름
1. **메타 생성** — `getMetaDict`가 reference 폴더와 content 폰트를 읽고, **`cr_mapping_v2.json`의 전체 11,172자**를 추론 대상으로 설정한다.
   - 각 타깃 글자는 `cr_mapping`에 의해 "이 글자를 만들기 위해 참고해야 하는 3개의 부품 글자"가 미리 지정되어 있다.
   - 만약 reference 폴더(78자)에 해당 부품 글자가 없는 경우, **무작위로 대체하지 않고 "자모 유사도 기반" fallback**을 수행한다. 즉, 누락된 글자를 초성·중성·종성으로 분해하여 보유 78자 중 일치하는 자모 개수(0~3개)가 가장 많은 글자를 골라 대신 사용한다. 이렇게 하면 형태가 비슷한 글자를 참조하게 되어 품질 저하를 최소화한다.
2. **LMDB 패키징** — [build_dataset/build_dataset.py](build_dataset/build_dataset.py)의 `save_lmdb`로 빠른 학습/추론용 DB 생성.
3. **모델 로드** — `generator_ema` → `generator` 순으로 가중치 로드 (EMA 우선).
4. **3-shot 스타일 추출** — 각 타깃 글자마다 `cr_mapping`이 지정한 참조 부품 글자 3장을 골라 `Component Encoder`에 입력, 다중 스케일(1.0/1.2/0.8)로 latent 추출 → Memory에 기록.
5. **콘텐츠 합성** — 콘텐츠 폰트의 글자 골격 + Memory에서 읽어온 스타일을 Transformer/Integrator가 결합 → 디코더가 128×128 결과 이미지 생성.
6. **저장** — `inference_results/target_style_images/reference_images_v2/images/<글자>.png`

### 4-3. 평가 지표 ([evaluator.py](evaluator.py))
- **PSNR / SSIM** — 픽셀·구조 유사도
- **L1 / RMSE** — 오차 크기
- **LPIPS (Alex / VGG)** — 지각 유사도
- TensorBoard에 동시 기록 (`vq_font_results/runs/`).

---

## 5. 단계별 데이터셋 구성

전체 데이터셋 디렉토리 구조는 다음과 같다.

```
datasets/
├── content_font_image/NanumBarunpenR/      ← 콘텐츠(골격) 폰트 — 모든 단계 공통
├── train_font_ttf/                         ← Uhbee 67종 + NanumBarunpenR (원본 TTF)
├── additional_train_font_ttf/              ← 추가 9종 (붓글씨·바탕·고딕·캘리)
├── train_font_image/                       ← TTF → PNG 변환된 학습용 이미지
│   ├── UhBee_*/                            ← Uhbee 67종 글자 이미지
│   ├── BMKIRANGHAERANG-TTF/ ...            ← 추가 9종 글자 이미지
│   └── reference_images_v2/                ← Stage 2 타깃 — 사용자 손글씨 78자
├── valid_font_image/                       ← 검증용
├── handwrite_dataset_v3/lmdb/              ← Stage 1·2 학습용 LMDB
└── handwrite_dataset_v3/meta/train.json    ← 학습 메타
```

### 5-1. Stage 0 (VQGAN 사전학습) 데이터
- **사용 데이터**: 76종 폰트로 렌더링한 한글 글자 이미지 (스타일 무관, 단일 이미지 단위)
- **데이터 리스트**: `vqgan_data/train_custom_v2.txt`, `vqgan_data/valid_custom_v2.txt`
- **단위**: 페어(pair)가 아닌 **개별 글자 이미지** (수십만 장)
- **목적**: 한글 글자 시각 prior(획·부품 형태) 학습용. 스타일 라벨이 필요 없으며, "한글이 어떻게 생겼는가"만 학습한다.

### 5-2. Stage 1 (일반 다중 폰트 본학습) 데이터
- **스타일 폰트**: **76종** (Uhbee 67종 + 추가 9종)
- **콘텐츠 폰트**: `NanumBarunpenR` (1종, 학습 대상 아님)
- **글자 범위**: `train_unis_v3.json`(학습용) / `val_unis_v3.json`(검증용)으로 분할
- **단위**: "콘텐츠 글자 + k-shot(3장) 참조 스타일 글자 + 타깃 글자" **페어** — 수십만 페어
- **제외 데이터**: `reference_images_v2/`의 78자 사용자 손글씨는 Stage 1에서 **사용하지 않음**

### 5-3. Stage 2 (파인튜닝) 데이터
- **주 타깃 스타일**: `train_font_image/reference_images_v2/` — 사용자 손글씨 **78자**
- **보조 데이터(선택)**: Stage 1 데이터의 일부 — catastrophic forgetting(이전 지식 망각) 방지
- **추론 입력으로도 동일하게 사용됨**: 즉 Stage 2의 학습 데이터 = 추론 시 reference

---

## 6. 단계별 기대 결과 (Outcomes)

### 6-1. Stage 0 결과
| 항목 | 값 / 설명 |
|---|---|
| 산출물 | `taming/experiments/checkpoints/.../epoch=000781.ckpt` |
| 핵심 성과 | 1024 토큰 한글 코드북 완성 — 임의의 한글 이미지를 코드북 토큰으로 인코딩/디코딩 가능 |
| 검증 방법 | 입력 글자 ↔ VQGAN 재구성 결과의 시각적 일치도 (코드북이 한글 prior를 잘 표현하는지 확인) |

### 6-2. Stage 1 결과 (`vq_font_v4.0`)
| 항목 | 값 / 설명 |
|---|---|
| 산출물 | `vq_font_results/checkpoints/vq_font_v4.0/last.ckpt` |
| 학습량 | 약 1.5M iter |
| 핵심 성과 | 76종 폰트에 대해 3-shot Few-Shot 생성 가능 — 임의의 한글 11,172자에 대해 "본 적 있는 스타일"이라면 합성 가능 |
| 평가 | val_unis_v3 글자에 대한 PSNR / SSIM / LPIPS (Alex·VGG) / L1 / RMSE |
| 한계 | 사용자 손글씨처럼 **학습에 포함되지 않은 새로운 스타일**은 어색하거나 평균화된 결과 산출 |

### 6-3. Stage 2 결과 (`brush_finetune_v2` — 본 프로젝트 최종 산출물)
| 항목 | 값 |
|---|---|
| 산출물 | `vq_font_results/checkpoints/brush_finetune_v2/last.ckpt` (≈544.4 MB, `generator_ema` 포함) |
| 학습 step | 41,000 / 50,000 |
| 최종 loss | 1.033 → **0.325** |
| 추론 가능 글자 | **11,172 / 11,172 (100%)** |
| 추론 결과 파일 | 11,172개 PNG (`inference_results/target_style_images/reference_images_v2/images/`) |

**정성적 성과**
- 78자의 참조만으로 사용자 손글씨 스타일이 반영된 **완전한 한글 폰트 세트** 자동 생성.
- 구조 태그·부품 매핑·VQ 코드북 정제 덕분에 **획 왜곡·디테일 손실 최소화**.
- 콘텐츠 폰트(`NanumBarunpenR`) 골격 보존으로 **가독성 유지**.

---

## 7. 단계별 알려진 한계 및 개선 포인트

### 7-1. Stage 0 (VQGAN 사전학습) 한계
| 한계 | 원인 | 권장 대응 |
|---|---|---|
| 코드북 활용도(perplexity) 미측정 | VQGAN의 1024 토큰 중 실제로 사용되는 토큰 비율을 확인하지 않음 | codebook usage 모니터링 추가, dead code 발생 시 EMA codebook 또는 reset 적용 |
| 학습 데이터가 76종 폰트에 한정 | 새로운 스타일(예: 캘리그래피, 옛한글)의 prior 부족 가능 | 코드북 사전학습 데이터에 다양한 스타일 추가 |

### 7-2. Stage 1 (일반 본학습) 한계
| 한계 | 원인 | 권장 대응 |
|---|---|---|
| Uhbee 폰트 편중 | 76종 중 67종이 Uhbee 시리즈 — 비슷한 "디지털 손글씨" 스타일에 치우침 | 바탕·명조·붓글씨·캘리그래피 등 다양성 보강 |
| 학습 시간 매우 김 | 1.5M iter | mixed precision, DDP 멀티 GPU 활용 |
| 검증 글자(val_unis_v3) 의존 | 동일 폰트의 unseen 글자에 대한 검증이므로, 진짜 unseen 스타일에 대한 일반화는 보장 못함 | leave-one-font-out 평가 추가 |

### 7-3. Stage 2 (파인튜닝) 한계
| 한계 | 원인 | 권장 대응 |
|---|---|---|
| 일부 글자 품질 편차 | reference 78자로 한글 부품 커버리지가 불완전 | 참조 이미지를 **200~300자**로 확대 (자모 분포 균형 고려) |
| 붓글씨 획 변형 학습 부족 | `iter=50,000`은 붓글씨 도메인에 부족할 수 있음 | `iter≥100,000` 권장 |
| 인코더 동결로 부품 학습 한계 | `component_encoder` 완전 동결 → 새로운 획 패턴 학습 불가 | 매우 작은 LR(1e-6)로 부분 해제 시도 |
| 추론 시 `reduction='mean'`로 디테일 뭉개짐 | 다중 참조 latent를 단순 평균화 | 가중 평균 / attention 기반 융합 검토 |
| Catastrophic forgetting 가능성 | 78자에 강하게 fitting 시 Stage 1에서 학습한 일반 구조 지식 손실 | Stage 1 데이터 일부를 섞어 학습(rehearsal) |

---

## 8. 단계별 핵심 파일 인덱스

### 8-1. Stage 0 (VQGAN 사전학습)
- 진입점: [train_vqgan.sh](train_vqgan.sh), [taming/main.py](taming/main.py)
- 설정: [vqgan/custom_vqgan.yaml](vqgan/custom_vqgan.yaml)
- 데이터 리스트: [vqgan_data/train_custom_v2.txt](vqgan_data/train_custom_v2.txt), [vqgan_data/valid_custom_v2.txt](vqgan_data/valid_custom_v2.txt), [vqgan_data/make_train_val_txt.py](vqgan_data/make_train_val_txt.py)
- 모델: [taming/models/](taming/models)

### 8-2. Stage 1·2 공통 (VQ-Font 학습)
- 진입점: [train.py](train.py), [train_vqfont.sh](train_vqfont.sh)
- 트레이너: [trainer/combined_trainer.py](trainer/combined_trainer.py), [trainer/base_trainer.py](trainer/base_trainer.py)
- 모델: [models/generator.py](models/generator.py), [models/comp_encoder.py](models/comp_encoder.py), [models/content_encoder.py](models/content_encoder.py), [models/former.py](models/former.py), [models/memory.py](models/memory.py), [models/decoder.py](models/decoder.py), [models/vq.py](models/vq.py), [models/discriminator.py](models/discriminator.py)
- 공통 설정: [cfgs/defaults.yaml](cfgs/defaults.yaml)
- 데이터 로더: [datasets/dataset_transformer.py](datasets/dataset_transformer.py), [datasets/datautils.py](datasets/datautils.py), [datasets/lmdbutils.py](datasets/lmdbutils.py)

### 8-3. Stage 1 (일반 본학습 전용)
- 설정: [cfgs/custom.yaml](cfgs/custom.yaml) (`iter=1.5M`, `g_lr=2e-4`)
- 결과 체크포인트: `vq_font_results/checkpoints/vq_font_v4.0/last.ckpt`
- 데이터 빌드: [build_dataset/build_meta4train.py](build_dataset/build_meta4train.py), [build_dataset/build_dataset.py](build_dataset/build_dataset.py)
- 메타 자원: [build_dataset/cr_mapping_v2.json](build_dataset/cr_mapping_v2.json), [build_dataset/structure_tags.json](build_dataset/structure_tags.json), [build_dataset/de_v2.json](build_dataset/de_v2.json), [build_dataset/train_unis_v3.json](build_dataset/train_unis_v3.json), [build_dataset/val_unis_v3.json](build_dataset/val_unis_v3.json)
- TTF → PNG 변환: [font2img.py](font2img.py)

### 8-4. Stage 2 (파인튜닝 전용)
- 설정: [cfgs/custom_finetune.yaml](cfgs/custom_finetune.yaml) (`iter=50K`, `g_lr=2e-5`)
- 결과 체크포인트: `vq_font_results/checkpoints/brush_finetune_v2/last.ckpt`
- 입력 데이터: `datasets/train_font_image/reference_images_v2/` (사용자 손글씨 78자)

### 8-5. 추론
- 진입점: [inference.py](inference.py), [infer_vqfont.sh](infer_vqfont.sh), [cfgs/custom_infer.yaml](cfgs/custom_infer.yaml)
- 평가기: [evaluator.py](evaluator.py)
- 출력 위치: `inference_results/target_style_images/<reference_폴더명>/images/<글자>.png`

### 8-6. 보조 분석 스크립트
- [_check_coverage.py](_check_coverage.py): cr_mapping 대비 reference 폴더의 부품 커버리지 확인
- [_check_brush.py](_check_brush.py): 붓글씨 데이터 점검
- [_analyze.py](_analyze.py): 학습 로그 분석
- [_verify_training.py](_verify_training.py): 학습 무결성 검증
- [_generate_report.py](_generate_report.py): 결과 보고서 자동 생성

---

## 9. 결론

본 프로젝트는 **AAAI 2024 VQ-Font** 의 한자 도메인 모델을 **한글 손글씨/붓글씨 도메인으로 이전**하기 위해, 다음과 같은 **3단계 학습 전략**을 채택한다.

1. **Stage 0 (VQGAN 사전학습)** — 한글 글자 시각 prior를 1024개 토큰 코드북에 압축.
2. **Stage 1 (일반 다중 폰트 본학습, `vq_font_v4.0`)** — Uhbee 손글씨 67종 + 추가 폰트 9종 = **총 76종** 의 폰트로 1.5M iter 오래 학습하여, **"한글이라는 글자 시스템 + 다양한 스타일을 다루는 일반 능력"** 을 모델에 주입.
3. **Stage 2 (타깃 도메인 파인튜닝, `brush_finetune_v2`)** — Stage 1의 가중치를 출발점으로, 사용자가 직접 쓴 **78자** 손글씨에 1/10 LR(`2e-5`), 50K iter로 가볍게 특화.

이 3단계 구조가 필요한 본질적 이유는, **78자만으로는 모델이 한글 글자 시스템 자체를 학습할 수 없기 때문**이다. 반드시 다양한 폰트로 한글 일반 지식을 먼저 학습(Stage 1)한 뒤, 그 위에 사용자 스타일을 얇게 입혀야(Stage 2) 11,172자라는 광범위한 생성 범위에서도 안정적인 품질을 얻을 수 있다.

최종 결과로, **사용자가 직접 쓴 78자의 손글씨만으로 11,172자 완성형 한글 전체를 동일 스타일로 생성**할 수 있게 되었으며, 현재 학습은 loss 0.325 수준까지 수렴했다. 추가 데이터 보강·학습 반복 증가·인코더 미세조정을 통해 품질을 추가 향상할 여지가 있다.

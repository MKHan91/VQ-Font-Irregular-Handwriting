# Coverage × Quality Analysis

- 생성 이미지 수: **11,172**
- 참조(GT 보유) 글자 수: **78**, PSNR 산출 가능: **78**
- 참조 폴더: `datasets/train_font_image/reference_images_v2`
- 생성 폴더: `inference_results/target_style_images/reference_images_v2/images`
- scipy(연결 컴포넌트): O
- skimage(SSIM): O

## 산출물
- `per_char_metrics.csv` — 음절별 raw 지표
- `summary_by_coverage.csv` — 커버리지 그룹별 요약 통계
- `plots/coverage_hist.png` — 결손 자모 수 분포
- `plots/ink_ratio_by_coverage.png`
- `plots/bbox_fill_by_coverage.png`
- `plots/edge_density_by_coverage.png`
- `plots/n_components_by_coverage.png` *(--with-components)*
- `plots/psnr_by_coverage.png` *(참조 78자 한정)*
- `plots/scatter_psnr_ink.png` *(참조 78자 한정)*

> 결손 자모 수가 0 인 그룹과 1+ 그룹의 분포·평균 차이를 보고서에 인용한다.
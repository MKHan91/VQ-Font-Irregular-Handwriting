# Structure × Quality Analysis

- 총 분석 글자: **11,172**
- PSNR 산출 가능 (참조 78자): **78**
- skimage SSIM: O

## 분류 축
- `has_jongseong` (2 그룹) — structure_tags.json 원본 라벨
- `jung_shape` (3 그룹) — horizontal / vertical / mixed
- `structure_class` (6 그룹) — `jung_shape × has_jongseong`

## 산출 파일
- `per_char_structure.csv`
- `summary_has_jongseong.csv`
- `summary_jung_shape.csv`
- `summary_structure_class.csv`
- `plots/<axis>_counts.png`, `plots/<axis>_<metric>.png`

> psnr/ssim 박스플롯은 참조 78자에서만 의미 있음 (n 표기 참고).
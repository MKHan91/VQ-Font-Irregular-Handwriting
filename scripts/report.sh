
# coverage_quality 9장 + per_char_metrics.csv
python3 scripts/coverage_quality_analysis.py --with-components \
  --out report/coverage_quality

# structure_quality 4장
python3 scripts/structure_quality.py \
  --metrics-csv report/coverage_quality/per_char_metrics.csv \
  --out report/structure_quality

# 비교 그리드
python3 scripts/make_comparison_grid.py --chars "가,각,꽃,울,행,헐,훿" \
  --title "Cherry-picked" --out report/grids/cherry.png
python3 scripts/make_comparison_grid.py --auto 24 --auto-strategy mixed \
  --seed 42 --rows-per-figure 12 --out report/grids/random_mixed

# KS X 1001 시트
python3 scripts/sample_sheet.py --ks-x-1001 \
  --cols 35 --rows-per-page 35 --cell 56 \
  --out report/sample_sheet_ks
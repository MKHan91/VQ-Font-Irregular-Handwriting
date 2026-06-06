# # (1) 체리픽 — 잘 나온 7글자
# python3 scripts/make_comparison_grid.py \
#   --chars "가,각,꽃,울,행,헐,훿" \
#   --title "Cherry-picked Samples" \
#   --out report/grids/cherry.png


# (2) 자동 24자 (78참조 포함/비포함 반반)
python3 scripts/make_comparison_grid.py \
  --auto 24 --auto-strategy mixed --seed 42 \
  --rows-per-figure 12 \
  --title "Random Mixed (with/without GT)" \
  --out report/grids/random_mixed
"""Exact McNemar test between two probe runs (H2: from-scratch vs full pipeline).
Usage: python rk_mcnemar.py full_correct.npy scratch_correct.npy"""
import sys, numpy as np
from scipy.stats import binomtest
a = np.load(sys.argv[1]); b = np.load(sys.argv[2])
n10 = int((a & ~b).sum()); n01 = int((~a & b).sum())
p = binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
print(f'full {a.mean():.4f}  scratch {b.mean():.4f}  diff {100*(a.mean()-b.mean()):.1f} pp  discordant {n10}/{n01}  exact McNemar p = {p:.3g}')
print('H2 supported' if (a.mean() - b.mean()) >= 0.20 and p < 0.05 else 'H2 not supported under the stated rule')

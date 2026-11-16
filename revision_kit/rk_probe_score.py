"""Scores a folder of generated glyph images with the frozen jamo-recognition probe (Section 3.5).
Usage: python rk_probe_score.py <glyph_folder> <out.json>
Files must be named by the syllable (e.g. 가.png). A missing or blank image counts as not read.
Requires probe.pt, train_probe.py and build_cache.py in the same folder (included in the kit)."""
import sys, os, json, glob, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_cache import norm, label
from train_probe import Probe, predict
UNSEEN_FINALS = {2, 5, 6, 7, 14, 22, 24, 25}   # finals absent from the 78 references
folder, out = sys.argv[1], sys.argv[2]
m = Probe(); m.load_state_dict(torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'probe.pt'), map_location='cpu'))
X, Y, ok_img, chars = [], [], [], []
for c in range(11172):
    ch = chr(0xAC00 + c); lab = label(ch)
    path = None
    for ext in ('.png', '.jpg', '.PNG'):
        if os.path.exists(os.path.join(folder, ch + ext)): path = os.path.join(folder, ch + ext); break
    img = norm(path) if path else None
    chars.append(ch); Y.append(lab); ok_img.append(img is not None)
    X.append(img if img is not None else np.full((64, 64), 255, np.uint8))
X = np.stack(X); Y = np.array(Y); ok_img = np.array(ok_img)
P = predict(m, X)
corr = (P == Y).all(1) & ok_img
uns = np.isin(Y[:, 2], list(UNSEEN_FINALS))
res = {'folder': folder, 'n': 11172, 'images_found_nonblank': int(ok_img.sum()),
       'acc_all': float(corr.mean()), 'acc_unseen_final': float(corr[uns].mean()), 'acc_covered': float(corr[~uns].mean()),
       'head_acc': [float(((P[:, j] == Y[:, j]) & ok_img).mean()) for j in range(3)]}
np.save(out.replace('.json', '_correct.npy'), corr)
json.dump(res, open(out, 'w'), indent=1); print(json.dumps(res, indent=1))

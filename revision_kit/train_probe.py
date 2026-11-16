"""Jamo-recognition probe: a small CNN with three heads (19 initial, 21 medial, 28 final incl. none).
Trained only on the Stage 1 font corpus (the writer's references and generated glyphs are never seen).
A syllable counts as read correctly when all three heads are correct.
"""
import json, sys, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

torch.manual_seed(0); np.random.seed(0)
torch.set_num_threads(2)


class Probe(nn.Module):
    def __init__(self):
        super().__init__()
        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True),
                                 nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True), nn.MaxPool2d(2))
        self.f = nn.Sequential(blk(1, 16), blk(16, 32), blk(32, 64), blk(64, 128))  # 64 -> 4
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(128 * 16, 512)
        self.h = nn.ModuleList([nn.Linear(512, n) for n in (19, 21, 28)])

    def forward(self, x):
        z = F.relu(self.fc(self.drop(self.f(x).flatten(1))))
        return [h(z) for h in self.h]


def to_t(X):
    return 1.0 - torch.from_numpy(X).float().unsqueeze(1) / 255.0  # ink = 1


def augment(x):
    B = x.shape[0]
    ang = (torch.rand(B) - 0.5) * 2 * np.deg2rad(8)
    sc = 1 + (torch.rand(B) - 0.5) * 0.2
    sh = (torch.rand(B) - 0.5) * 0.3
    tx, ty = (torch.rand(B) - 0.5) * 0.12, (torch.rand(B) - 0.5) * 0.12
    c, s = torch.cos(ang) / sc, torch.sin(ang) / sc
    th = torch.stack([torch.stack([c, -s + sh, tx], 1), torch.stack([s, c, ty], 1)], 1)
    g = F.affine_grid(th, x.shape, align_corners=False)
    x = F.grid_sample(x, g, align_corners=False, padding_mode="zeros")
    # stroke-width jitter: dilation or erosion on a random half
    k = torch.rand(B) < 0.5
    if k.any():
        x[k] = F.max_pool2d(x[k], 3, 1, 1)
    e = (torch.rand(B) < 0.25) & ~k
    if e.any():
        x[e] = -F.max_pool2d(-x[e], 3, 1, 1)
    return x


@torch.no_grad()
def predict(m, X, bs=1024):
    m.eval(); P = []
    for i in range(0, len(X), bs):
        o = m(to_t(X[i:i + bs]))
        P.append(torch.stack([t.argmax(1) for t in o], 1))
    return torch.cat(P).numpy()


def acc(P, Y):
    ok = (P == Y).all(1)
    return float(ok.mean()), [float((P[:, j] == Y[:, j]).mean()) for j in range(3)]


if __name__ == "__main__":
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    X = np.load("cache_train_X.npy", mmap_mode="r"); Y = np.load("cache_train_Y.npy").astype(np.int64)
    Xv = np.load("cache_valid_X.npy"); Yv = np.load("cache_valid_Y.npy").astype(np.int64)
    Xr = np.load("cache_refs_X.npy"); Yr = np.load("cache_refs_Y.npy").astype(np.int64)
    n = len(X); print("train", n, "valid", len(Xv), flush=True)
    m = Probe(); opt = torch.optim.AdamW(m.parameters(), 2e-3, weight_decay=1e-4)
    bs = 256; steps = epochs * (n // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps)
    t0 = time.time(); step = 0; log = []
    for ep in range(epochs):
        m.train(); perm = np.random.permutation(n)
        for i in range(0, n - bs + 1, bs):
            idx = np.sort(perm[i:i + bs])
            x = augment(to_t(X[idx])); y = torch.from_numpy(Y[idx])
            out = m(x)
            loss = sum(F.cross_entropy(o, y[:, j], label_smoothing=0.05) for j, o in enumerate(out))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); step += 1
            if step % 200 == 0:
                print(f"ep{ep} step{step}/{steps} loss {loss.item():.3f} {time.time() - t0:.0f}s", flush=True)
        sub = np.random.RandomState(ep).choice(len(Xv), min(8000, len(Xv)), replace=False)
        va = acc(predict(m, Xv[sub]), Yv[sub]); ra = acc(predict(m, Xr), Yr)
        log.append({"epoch": ep, "valid_unseen_fonts": va, "writer_refs78": ra})
        print("EPOCH", ep, "unseen-font syllable acc %.4f" % va[0], "writer refs %.4f" % ra[0], flush=True)
        torch.save(m.state_dict(), "probe.pt")
    va = acc(predict(m, Xv), Yv)
    json.dump({"log": log, "final_valid_full": va}, open("probe_train_log.json", "w"), indent=1)
    print("FINAL unseen fonts (all)", va, flush=True)

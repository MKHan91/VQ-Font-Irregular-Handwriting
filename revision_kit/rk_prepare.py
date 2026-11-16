#!/usr/bin/env python3
"""
rk_prepare.py - builds every mapping, fold split and meta file used by the revision runs.
Pure Python (json + shutil); no GPU, no torch. Deterministic.

    python revision_kit/rk_prepare.py \
        --refs datasets/train_font_image/reference_images_v2 \
        --cr_mapping build_dataset/cr_mapping_v2.json \
        --meta /path/to/the/meta/train.json/used/for/Stage2 \
        --out revision_runs/prep

Outputs (under --out)
  mappings/cr_selfexcl78.json  78 keys. S(u) of the published mapping with u itself replaced by the
                               best-scoring reference not already in S(u) (inference-time exclusion, E1)
  mappings/selfexcl78_table.csv  original S(u) and its replacement, for the Supplement
  folds/fold{k}/cr_mapping.json  11,172 keys; the 3 references of every target are chosen from the
                                 TRAINING part of fold k only, with the selection rule of
                                 build_dataset/cr_mapping.py (top-3 jamo-overlap count)
  folds/fold{k}/cr_heldout.json  same mapping restricted to the held-out targets (inference)
  folds/fold{k}/meta.json        copy of --meta with train['reference_images_v2'] = training part only
  folds/fold{k}/refs/            copies of the training-part reference images (inference img_path)
  folds/folds.json               fold membership, stratified by the six structural strata x final
"""
import argparse, csv, json, random, shutil
from collections import Counter
from pathlib import Path

CHO = list("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ")
JUNG = list("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")
JONG = [""] + list("ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ")
HORIZ = set("ㅗㅛㅜㅠㅡ"); VERT = set("ㅏㅐㅑㅒㅓㅔㅕㅖㅣ")  # remaining 7 medials are 'mixed'


def dec(ch):  # identical to decompose_hangul() in build_dataset/cr_mapping.py
    c = ord(ch) - 0xAC00
    return [CHO[c // 588], JUNG[(c % 588) // 28]] + ([JONG[c % 28]] if c % 28 else [])


def stratum(ch):
    c = ord(ch) - 0xAC00; m = JUNG[(c % 588) // 28]; f = c % 28 != 0
    base = 1 if m in HORIZ else 2 if m in VERT else 3
    return base + (3 if f else 0)  # S1..S6 as in the manuscript


def score(t, r):  # identical to select_references() scoring
    a, b = Counter(dec(t)), Counter(dec(r))
    return sum(min(a[c], b.get(c, 0)) for c in a)


def pos_sim(t, r):  # positional (initial, medial, final) agreement, as FixedRefDataset.jamo_similarity
    a, b = ord(t) - 0xAC00, ord(r) - 0xAC00
    return (a // 588 == b // 588) + ((a % 588) // 28 == (b % 588) // 28) + (a % 28 == b % 28)


def rank_key(t):
    # primary: jamo-overlap count (original rule); secondary: positional agreement; then code point.
    # (The published mapping broke ties in os.listdir order, which is filesystem dependent.)
    return lambda r: (-score(t, r), -pos_sim(t, r), r)


def top3(t, pool):
    return sorted(pool, key=rank_key(t))[:3]


def hx(ch): return "%04X" % ord(ch)
def ch_(h): return chr(int(h, 16))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", required=True)
    ap.add_argument("--cr_mapping", required=True)
    ap.add_argument("--meta", required=False, help="meta json used for Stage 2 (for fold metas)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    refs_dir, out = Path(a.refs), Path(a.out)
    refs = sorted(p.stem for p in refs_dir.glob("*.png"))
    assert len(refs) == 78, f"expected 78 references, found {len(refs)}"
    cr = json.load(open(a.cr_mapping, encoding="utf-8"))
    (out / "mappings").mkdir(parents=True, exist_ok=True)

    # ---------- E1: inference-time self-exclusion on the 78 ----------
    # loo, rows = {}, []
    loo, rows = dict(cr), []
    for u in refs:
        S = [ch_(h) for h in cr[hx(u)]]
        assert all(s in refs for s in S)
        if u in S:
            cand = sorted((r for r in refs if r != u and r not in S), key=rank_key(u))
            rep = cand[0]
            S2 = [rep if s == u else s for s in S]
        else:
            rep, S2 = "", S
        loo[hx(u)] = [hx(s) for s in S2]
        rows.append([u, "".join(S), "".join(S2), rep, score(u, rep) if rep else "", pos_sim(u, rep) if rep else ""])
    json.dump(loo, open(out / "mappings/cr_selfexcl78.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    with open(out / "mappings/selfexcl78_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["target", "S_original", "S_selfexcluded", "replacement", "replacement_overlap", "replacement_positional"]); w.writerows(rows)
    print(f"[E1] self-excluded mapping: {sum(1 for r in rows if r[3])} of 78 targets had themselves in S(u)")

    # ---------- E3/E4: stratified k-fold split of the 78 ----------
    rng = random.Random(a.seed)
    by = {}
    for u in refs: by.setdefault(stratum(u), []).append(u)
    order = []
    for s in sorted(by):
        g = by[s][:]; rng.shuffle(g); order += g
    folds = [order[i::a.k] for i in range(a.k)]  # round-robin over strata-ordered list
    fdir = out / "folds"; fdir.mkdir(exist_ok=True)
    meta = json.load(open(a.meta, encoding="utf-8")) if a.meta else None
    allk = [hx(chr(c)) for c in range(0xAC00, 0xD7A4)]
    summary = {"seed": a.seed, "k": a.k, "folds": []}
    for i, held in enumerate(folds):
        train = [u for u in refs if u not in held]
        d = fdir / f"fold{i}"; (d / "refs").mkdir(parents=True, exist_ok=True)
        mp = {k: [hx(r) for r in top3(ch_(k), train)] for k in allk}
        json.dump(mp, open(d / "cr_mapping.json", "w", encoding="utf-8"), ensure_ascii=False)
        json.dump({hx(u): mp[hx(u)] for u in held}, open(d / "cr_heldout.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        for u in train: shutil.copy2(refs_dir / f"{u}.png", d / "refs" / f"{u}.png")
        (d / "heldout.txt").write_text("\n".join(held), encoding="utf-8")
        (d / "train.txt").write_text("\n".join(train), encoding="utf-8")
        if meta is not None:
            m = json.loads(json.dumps(meta))
            key = "reference_images_v2"
            assert key in m["train"], f"'{key}' not in meta['train']; check --meta"
            m["train"][key] = [hx(u) for u in train]
            json.dump(m, open(d / "meta.json", "w", encoding="utf-8"), ensure_ascii=False)
        assert not set(held) & {ch_(h) for v in mp.values() for h in v}, "held-out glyph leaked into mapping"
        summary["folds"].append({"fold": i, "heldout": held, "n_train": len(train),
                                 "strata_heldout": dict(Counter(stratum(u) for u in held))})
        print(f"[fold{i}] held-out {len(held)}: {''.join(held)}")
    json.dump(summary, open(fdir / "folds.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    if meta is not None:
        print("[meta] fonts in meta['train']:", len(meta["train"]), "->", sorted(meta["train"])[:5], "...")
    print("Prepared under", out)


if __name__ == "__main__":
    main()

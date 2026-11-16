#!/usr/bin/env python3
"""
rk_audit.py - E0: which parameters actually changed in Stage 1 and Stage 2?
CPU is enough. Compares the saved checkpoints tensor by tensor and groups the result by module.

    python revision_kit/rk_audit.py \
        --stage0 taming/experiments/checkpoints/<run>_custom_vqgan/epoch=000781.ckpt \
        --stage1 vq_font_results/checkpoints/vq_font_v4.0/last.ckpt \
        --stage2 vq_font_results/checkpoints/brush_finetune_v2/last.ckpt \
        --meta1 <meta/train.json used for Stage 1> --meta2 <meta/train.json used for Stage 2> \
        --out revision_runs/E0_audit

Writes audit_modules.csv (one row per module group and comparison), audit_summary.json, meta_check.json.
"""
import argparse, csv, json, re
from collections import defaultdict
from pathlib import Path
import torch


def load_gen(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    out = {}
    for key in ("generator", "generator_ema"):
        if key in ck and isinstance(ck[key], dict):
            out[key] = ck[key]
    if not out and "state_dict" in ck:
        out["state_dict"] = ck["state_dict"]
    meta = {k: (v if isinstance(v, (int, float, str)) else type(v).__name__) for k, v in ck.items() if not isinstance(v, dict)}
    return out, meta


def group(name):
    n = name[len("vqgan."):] if name.startswith("vqgan.") else name
    pre = "vqgan." if name.startswith("vqgan.") else ""
    m = re.match(r"(decoder\.layers\.\d+|decoder\.[a-z_]+|encoder\.[a-z_]+|quantize|post_quant_conv|quant_conv|loss)", n)
    if pre and m:
        return pre + m.group(1)
    return pre + n.split(".")[0]


def compare(a, b, prefix_b=""):
    rows = defaultdict(lambda: {"tensors": 0, "changed": 0, "max_abs": 0.0, "rel_l2_num": 0.0, "rel_l2_den": 0.0, "missing": 0})
    for k, va in a.items():
        kb = prefix_b + k
        g = group(kb)
        r = rows[g]
        if kb not in b or not torch.is_tensor(va) or not va.is_floating_point():
            if kb not in b: r["missing"] += 1
            continue
        vb = b[kb]
        if vb.shape != va.shape:
            r["missing"] += 1; continue
        d = (vb.float() - va.float())
        r["tensors"] += 1
        mx = float(d.abs().max()) if d.numel() else 0.0
        r["changed"] += int(mx > 0)
        r["max_abs"] = max(r["max_abs"], mx)
        r["rel_l2_num"] += float((d ** 2).sum()); r["rel_l2_den"] += float((va.float() ** 2).sum())
    out = {}
    for g, r in rows.items():
        r["rel_l2"] = (r["rel_l2_num"] ** 0.5) / (r["rel_l2_den"] ** 0.5 + 1e-12)
        r["status"] = "unchanged" if r["changed"] == 0 and r["tensors"] > 0 else ("changed" if r["changed"] else "n/a")
        out[g] = r
    return out


def meta_check(path):
    if not path:
        return None
    m = json.load(open(path, encoding="utf-8"))
    tr = m.get("train", {})
    return {"n_fonts_train": len(tr),
            "contains_reference_images_v2": "reference_images_v2" in tr,
            "n_reference_images_v2": len(tr.get("reference_images_v2", [])),
            "n_glyphs_train": sum(len(v) for v in tr.values()),
            "fonts": sorted(tr)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage0", required=True); ap.add_argument("--stage1", required=True)
    ap.add_argument("--stage2", required=True); ap.add_argument("--meta1"); ap.add_argument("--meta2")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    s0 = torch.load(a.stage0, map_location="cpu", weights_only=False)
    s0 = {k: v for k, v in s0.get("state_dict", s0).items() if not k.startswith("loss.")}
    g1, m1 = load_gen(a.stage1); g2, m2 = load_gen(a.stage2)
    rows, summary = [], {"stage1_ckpt_scalars": m1, "stage2_ckpt_scalars": m2}
    for key in g1:
        res = compare(s0, g1[key], prefix_b="vqgan.")
        for g, r in sorted(res.items()):
            rows.append(["Stage0->Stage1", key, g, r["tensors"], r["changed"], r["missing"], f'{r["max_abs"]:.3e}', f'{r["rel_l2"]:.3e}', r["status"]])
    for key in g1:
        if key not in g2: continue
        res = compare(g1[key], g2[key])
        for g, r in sorted(res.items()):
            rows.append(["Stage1->Stage2", key, g, r["tensors"], r["changed"], r["missing"], f'{r["max_abs"]:.3e}', f'{r["rel_l2"]:.3e}', r["status"]])
    with open(out / "audit_modules.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["comparison", "weights", "module_group", "n_tensors", "n_changed", "n_missing", "max_abs_diff", "rel_l2_change", "status"])
        w.writerows(rows)
    key_q = [r for r in rows if r[2] in ("vqgan.quantize",) or r[2].startswith(("component_encoder", "content_encoder", "vqgan.decoder.layers"))]
    summary["key_rows"] = key_q
    json.dump(summary, open(out / "audit_summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    json.dump({"stage1": meta_check(a.meta1), "stage2": meta_check(a.meta2)},
              open(out / "meta_check.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("comparison       weights        module_group                    changed/tensors  rel_l2")
    for r in key_q:
        print(f"{r[0]:16s} {r[1]:14s} {r[2]:32s} {r[4]:>4}/{r[3]:<6} {r[7]}")
    print("Wrote", out)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
rk_patch.py - minimal, idempotent hooks for the revision experiments.

Run once from the root of the VQ-Font project (the folder that holds train.py):

    python revision_kit/rk_patch.py --root .

Every hook falls back to the ORIGINAL behaviour when its environment variable is
unset, so the published configuration is reproduced exactly unless a variable is
set. A backup of each edited file is written as <file>.rk_orig before the first edit.

Hooks
  RK_LATENT_SIGMA   models/generator.py   training-time latent noise (default 0.03)
  RK_FREEZE         train.py              'encoders' (default, as in the code used),
                                          'none' (all generator parameters trainable),
                                          'strict' (encoders frozen AND, inside the VQGAN,
                                          only decoder.layers.0-2 + post_quant_conv trainable,
                                          i.e. the configuration written in the manuscript)
  RK_TRAINABLE_LOG  train.py              path of a text file listing trainable parameters
  RK_PY_SEED        train.py              seeds Python's `random` (dataset sampling)
  RK_CR_MAPPING     evaluator.py, inference.py (if present)
                                          path of the content-reference mapping JSON
"""
import argparse, re, shutil, sys
from pathlib import Path

MARK = "# [RK-PATCH]"


def backup(p: Path):
    b = p.with_suffix(p.suffix + ".rk_orig")
    if not b.exists():
        shutil.copy2(p, b)


def ensure_import_os(src: str) -> str:
    if re.search(r"^import os\b", src, flags=re.M):
        return src
    return "import os  " + MARK + "\n" + src


def patch_generator(root: Path):
    p = root / "models" / "generator.py"
    s = p.read_text(encoding="utf-8")
    if MARK + " sigma" in s:
        print("[skip] generator.py already patched"); return
    pat = re.compile(r"^(\s*)noise_strength\s*=\s*0\.03.*$", flags=re.M)
    if not pat.search(s):
        sys.exit("[FAIL] generator.py: line 'noise_strength = 0.03' not found; patch by hand.")
    s = pat.sub(lambda m: f'{m.group(1)}noise_strength = float(os.environ.get("RK_LATENT_SIGMA", "0.03"))  {MARK} sigma', s, count=1)
    s = ensure_import_os(s)
    backup(p); p.write_text(s, encoding="utf-8"); print("[ok] generator.py: RK_LATENT_SIGMA hook")


FREEZE_OLD = re.compile(
    r"^(\s*)for name, param in gen\.named_parameters\(\):\n"
    r"\s*if 'component_encoder' in name or 'content_encoder' in name:\n"
    r"\s*param\.requires_grad_\(False\)\n", flags=re.M)

FREEZE_NEW = '''{i}_rk_freeze = os.environ.get("RK_FREEZE", "encoders")  {m} freeze
{i}_rk_vq_keep = ("decoder.layers.0.", "decoder.layers.1.", "decoder.layers.2.", "post_quant_conv.")
{i}for name, param in gen.named_parameters():
{i}    if _rk_freeze in ("encoders", "strict") and ('component_encoder' in name or 'content_encoder' in name):
{i}        param.requires_grad_(False)
{i}    if _rk_freeze == "strict" and name.startswith("vqgan.") and not name[len("vqgan."):].startswith(_rk_vq_keep):
{i}        param.requires_grad_(False)
{i}    if _rk_freeze == "none":
{i}        param.requires_grad_(True)
{i}_rk_tr = [n for n, q in gen.named_parameters() if q.requires_grad]
{i}print(f"[RK] freeze={{_rk_freeze}} trainable_tensors={{len(_rk_tr)}} trainable_params={{sum(q.numel() for q in gen.parameters() if q.requires_grad)}} total_params={{sum(q.numel() for q in gen.parameters())}}")
{i}if os.environ.get("RK_TRAINABLE_LOG"):
{i}    open(os.environ["RK_TRAINABLE_LOG"], "w").write("\\n".join(_rk_tr))
'''


def patch_train(root: Path):
    p = root / "train.py"
    s = p.read_text(encoding="utf-8")
    if MARK + " freeze" in s:
        print("[skip] train.py already patched"); return
    m = FREEZE_OLD.search(s)
    if not m:
        sys.exit("[FAIL] train.py: encoder-freezing loop not found; patch by hand.")
    s = s[:m.start()] + FREEZE_NEW.format(i=m.group(1), m=MARK) + s[m.end():]
    # optional Python-random seeding next to the numpy/torch seeds
    s = s.replace('    torch.manual_seed(cfg["seed"])\n',
                  '    torch.manual_seed(cfg["seed"])\n'
                  '    if os.environ.get("RK_PY_SEED"):  ' + MARK + ' seed\n'
                  '        import random as _r; _r.seed(int(os.environ["RK_PY_SEED"]))\n', 1)
    s = ensure_import_os(s)
    backup(p); p.write_text(s, encoding="utf-8"); print("[ok] train.py: RK_FREEZE / RK_TRAINABLE_LOG / RK_PY_SEED hooks")


LIT = re.compile(r"""(['"])([^'"\n]*cr_mapping_v2\.json)\1""")


def patch_mapping_literal(p: Path):
    if not p.exists():
        print(f"[info] {p.name} not found (skipped)"); return
    s = p.read_text(encoding="utf-8")
    if MARK + " crmap" in s:
        print(f"[skip] {p.name} already patched"); return
    n = len(LIT.findall(s))
    if n == 0:
        print(f"[warn] {p.name}: no hard-coded 'cr_mapping_v2.json' literal. If it reads the mapping from the "
              f"config, pass --content_reference_json on the command line instead."); return
    s = LIT.sub(lambda m: f'os.environ.get("RK_CR_MAPPING", {m.group(0)})', s)
    s = ensure_import_os(s) + f"\n{MARK} crmap ({n} literal(s) redirected)\n"
    backup(p); p.write_text(s, encoding="utf-8"); print(f"[ok] {p.name}: {n} mapping literal(s) -> RK_CR_MAPPING")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    a = ap.parse_args()
    root = Path(a.root).resolve()
    if not (root / "train.py").exists():
        sys.exit(f"[FAIL] {root} does not contain train.py")
    patch_generator(root)
    patch_train(root)
    patch_mapping_literal(root / "evaluator.py")
    patch_mapping_literal(root / "inference.py")
    print("Done. Original files kept as *.rk_orig. Unset variables reproduce the original behaviour.")

"""Look inside the downloaded CISLR-Data bundle without loading anything big.

    python inspect_cislr.py "C:\\Users\\<you>\\Downloads\\CISLR-Data"
    python inspect_cislr.py "<folder>" --glosses hello head ache stomach need doctor I you he she good morning night afternoon

Prints: (1) how the keypoint files look (shape, value range, empty frames),
(2) which of your glosses exist in the INCLUDE part, (3) the top of the small JSON files,
(4) the first characters of portable_mapping.json (never the whole 1.3 GB).
Copy the whole output back to Claude.
"""
import argparse, collections, glob, json, os, re
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("--glosses", nargs="*", default=None, help="words to look for in the INCLUDE file names")
ap.add_argument("--samples", type=int, default=4)
a = ap.parse_args()
R = a.root


def describe(path):
    x = np.load(path, allow_pickle=False)
    f = x.astype(np.float64) if x.dtype.kind in "fiu" else None
    s = f"{os.path.basename(path)}: shape {x.shape} dtype {x.dtype}"
    if f is not None and f.size:
        nan = np.isnan(f).mean()
        g = np.nan_to_num(f)
        flat = g.reshape(g.shape[0], -1) if g.ndim >= 2 else g[None]
        empty = (np.abs(flat).sum(1) == 0).mean()
        s += f" | min {g.min():.3f} max {g.max():.3f} | NaN {nan:.1%} | all-zero frames {empty:.1%}"
        if g.ndim == 3:
            s += f" | per-keypoint zero share (first 5 of {g.shape[1]}): " + \
                 " ".join(f"{(np.abs(g[:, k]).sum(1) == 0).mean():.0%}" for k in range(min(5, g.shape[1])))
    return s


for part in ("include", "cislr"):
    files = sorted(glob.glob(os.path.join(R, "2_keypoints", part, "*.npy")))
    print(f"\n=== 2_keypoints/{part}: {len(files)} files")
    for p in files[:a.samples]:
        try:
            print("  " + describe(p))
        except Exception as e:
            print(f"  {os.path.basename(p)}: could not read ({e})")
    if len(files) > a.samples:                                  # a few from the middle/end too
        for p in (files[len(files) // 2], files[-1]):
            try:
                print("  " + describe(p))
            except Exception as e:
                print(f"  {os.path.basename(p)}: could not read ({e})")

inc = glob.glob(os.path.join(R, "2_keypoints", "include", "*.npy"))
gl = collections.Counter()
for p in inc:
    n = os.path.basename(p)[:-4]
    n = re.sub(r"^include_", "", n)
    n = re.split(r"_?MVI_", n)[0]                                # 'you__plural_' / 'thank_you' / '' (no gloss)
    gl[n.strip("_")] += 1
print(f"\n=== INCLUDE: {len(gl)} distinct name stems, {sum(gl.values())} files; files without a name: {gl.get('', 0)}")
print("most common:", gl.most_common(12))
if a.glosses:
    print("\nYour glosses found as whole words in INCLUDE file names (exact stem / stems containing the word):")
    for w in a.glosses:
        exact = gl.get(w.lower(), 0)
        pat = re.compile(r"(^|_)" + re.escape(w.lower()) + r"(_|$)")
        within = sorted(((k, v) for k, v in gl.items() if pat.search(k.lower()) and k.lower() != w.lower()), key=lambda t: -t[1])
        print(f"  {w:10s} exact: {exact:3d}   also in: {within[:5]}")

print("\n=== small JSON files")
for rel in ("7_splits/split_metadata.json", "7_splits/shards/shard_metadata.json"):
    p = os.path.join(R, rel)
    if os.path.exists(p):
        print(f"--- {rel}\n" + open(p, encoding="utf-8").read(1500))
for rel in ("7_splits/train_indices.json",):
    p = os.path.join(R, rel)
    if os.path.exists(p):
        print(f"--- {rel} (first 300 chars)\n" + open(p, encoding="utf-8").read(300))

p = os.path.join(R, "6_corpus_final", "portable_mapping.json")
if os.path.exists(p):
    print(f"\n=== {p}: {os.path.getsize(p) / 1e6:.0f} MB; first 1500 characters only")
    print(open(p, encoding="utf-8").read(1500))

"""Fine-tune a small T5 to turn sign-order glosses into English sentences.

    python train_gloss_t5.py                       # google/flan-t5-small, about 10-20 minutes on a laptop CPU
    python train_gloss_t5.py --base google/flan-t5-base --epochs 15   # larger, slower, usually a bit better

Run it once with internet (it downloads the base model into the Hugging Face cache).
It saves the result to artifacts/gloss_t5, which realtime.py loads with no internet.

Training data: Dataset/gloss_pairs.jsonl, written by gloss_data.py (run automatically).
10% of the pairs are kept aside to measure whether the model generalises; the 3 HELD_OUT
glosses are never trained on and are printed at the end. Look at those lines: they are the
honest test of whether the model learned the patterns or memorised sentences.
"""
import argparse
import json
import os
import random
import time

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import gloss_data
from translator import DEFAULT_DIR, LICENSE_FILE, PREFIX, build_licenses, omitted, unsupported

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="google/flan-t5-small")
ap.add_argument("--epochs", type=int, default=25)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--lr", type=float, default=5e-4)
ap.add_argument("--out", default=DEFAULT_DIR)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

random.seed(args.seed); torch.manual_seed(args.seed)
pairs = gloss_data.generate()
train_all, held = gloss_data.split(pairs)
noisy = gloss_data.generate_noisy(pairs)                                # glosses with a stray pronoun (see gloss_data.py)
noisy_test = gloss_data.noisy_held_out(pairs)                           # the same on held-out sentences: test only
items = sorted(train_all.items()); random.shuffle(items)
n_val = max(10, len(items) // 10)
must_train = [it for it in items if it[0] in gloss_data.RECORDED]       # the sentences you recorded are always trained on
others = [it for it in items if it[0] not in gloss_data.RECORDED]
val, train = others[:n_val], others[n_val:] + must_train + sorted(noisy.items())
print(f"{len(train)} training pairs ({len(noisy)} with a stray pronoun), {len(val)} validation pairs, {len(held)} held-out sentences")

device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(args.base)
model = AutoModelForSeq2SeqLM.from_pretrained(args.base).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)


def batches(rows, bs, shuffle):
    rows = list(rows)
    if shuffle:
        random.shuffle(rows)
    for i in range(0, len(rows), bs):
        chunk = rows[i:i + bs]
        enc = tok([PREFIX + g for g, _ in chunk], return_tensors="pt", padding=True, truncation=True, max_length=64)
        lab = tok([e for _, e in chunk], return_tensors="pt", padding=True, truncation=True, max_length=96).input_ids
        lab[lab == tok.pad_token_id] = -100
        yield {k: v.to(device) for k, v in enc.items()}, lab.to(device)


def predict(glosses):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(glosses), 32):
            enc = tok([PREFIX + g for g in glosses[i:i + 32]], return_tensors="pt", padding=True).to(device)
            gen = model.generate(**enc, num_beams=4, max_new_tokens=64)
            out += [t.strip() for t in tok.batch_decode(gen, skip_special_tokens=True)]
    return out


def exact_match(rows):
    preds = predict([g for g, _ in rows])
    wrong = [(g, e, p) for (g, e), p in zip(rows, preds) if p != e]
    return 1 - len(wrong) / max(len(rows), 1), wrong


t0 = time.time()
for ep in range(1, args.epochs + 1):
    model.train()
    total = 0.0; n = 0
    for enc, lab in batches(train, args.batch, True):
        loss = model(**enc, labels=lab).loss
        loss.backward(); opt.step(); opt.zero_grad()
        total += loss.item(); n += 1
    msg = f"epoch {ep:2d}/{args.epochs}  loss {total / n:.4f}  ({time.time() - t0:.0f}s)"
    if ep % 5 == 0 or ep == args.epochs:
        acc, _ = exact_match(val)
        msg += f"  validation exact match {acc:.0%}"
    print(msg, flush=True)

acc, wrong = exact_match(val)
print(f"\nValidation exact match: {acc:.0%} ({len(val) - len(wrong)}/{len(val)})")
for g, e, p in wrong[:8]:
    print(f"   {g!r}\n      wanted: {e!r}\n      got:    {p!r}")

rec_rows = sorted(gloss_data.RECORDED.items())
rec_acc, rec_wrong = exact_match(rec_rows)
print(f"\nYour {len(rec_rows)} recorded demo sentences (trained on): {len(rec_rows) - len(rec_wrong)}/{len(rec_rows)} exact")
for g, e, p in rec_wrong:
    print(f"   {g!r}\n      wanted: {e!r}\n      got:    {p!r}")
if rec_wrong:
    print("   -> not all exact. Train longer (--epochs 40) or use --base google/flan-t5-base before the demo.")

print("\nHeld-out sentences (never trained on):")
held_rows = sorted(held.items())
for (g, e), p in zip(held_rows, predict([g for g, _ in held_rows])):
    print(f"   {g!r} -> {p!r}   {'OK' if p == e else 'wanted ' + repr(e)}")

print("\nHeld-out sentences with a stray pronoun added (never trained on):")
noisy_rows = sorted(noisy_test.items())
n_ok = 0
for (g, e), p in zip(noisy_rows, predict([g for g, _ in noisy_rows])):
    n_ok += p == e
    print(f"   {g!r} -> {p!r}   {'OK' if p == e else 'wanted ' + repr(e)}")

licenses = build_licenses(pairs)                                         # which sign licenses which English word
flagged = [(g, e) for g, e in list(pairs.items()) + list(noisy.items()) if unsupported(g.split(), e, licenses) or omitted(g.split(), e, licenses)]
print(f"\nInvented-word check: {len(licenses)} words tied to signs; "
      f"{len(flagged)} of {len(pairs) + len(noisy)} reference translations would be wrongly refused")
for g, e in flagged[:5]:
    print(f"   {g!r} -> {e!r}: {unsupported(g.split(), e, licenses) + omitted(g.split(), e, licenses)}")

os.makedirs(args.out, exist_ok=True)
with open(os.path.join(args.out, LICENSE_FILE), "w", encoding="utf-8") as f:
    json.dump(licenses, f, indent=1)
model.save_pretrained(args.out); tok.save_pretrained(args.out)
with open(os.path.join(args.out, "train_info.json"), "w") as f:
    json.dump({"base": args.base, "epochs": args.epochs, "pairs": len(train), "val_exact_match": acc}, f)
print(f"\nSaved to {args.out}")

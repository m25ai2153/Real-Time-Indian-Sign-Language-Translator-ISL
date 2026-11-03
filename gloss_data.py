"""Training pairs (glosses -> English) for the gloss translator.

A transformer learns the conversion from examples, so it needs examples. For a small
vocabulary these are cheap to produce: this file lists the vocabulary by role (subject,
symptom, need, ...) and a handful of sentence patterns, and generates every combination.
When you add signs, add them to the role lists (or add a pattern) and re-run
train_gloss_t5.py; no code elsewhere changes. For a much larger vocabulary, the same
file format (gloss line -> English line) can be produced by a large language model once,
offline, and the small model fine-tuned on that.

Patterns follow sign order (glosses as printed by realtime.py), e.g. a symptom may come
before its subject ("head ache I need medicine"), and a missing subject defaults to "I"
(the signer). The three HELD_OUT glosses are never trained on: they test whether the
model learned the patterns or memorised sentences.

    python gloss_data.py        # writes Dataset/gloss_pairs.jsonl and prints a summary
"""
import itertools
import json
import os
import random

SUBJECTS = {"I": "I", "you": "You"}                   # gloss -> English (sentence start)
SYMPTOMS = {"head ache": "a headache", "stomach ache": "a stomach ache"}
NEEDS = {"doctor": "a doctor", "medicine": "medicine", "water": "water", "help": "help"}
GREETINGS = {"hello": "Hello", "good afternoon": "Good afternoon", "thank_you": "Thank you"}
ADDRESS = {"doctor": "doctor"}
STANDALONE = {"yes": "Yes.", "no": "No."}

HELD_OUT_ORDER = ["you stomach ache", "doctor need medicine", "I head ache need doctor"]   # the order they are signed in tests
HELD_OUT = set(HELD_OUT_ORDER)


def _low(subject_en):                                  # "You" -> "you" inside a sentence, "I" stays
    return subject_en if subject_en == "I" else subject_en.lower()


def _have(subj, sym):
    return f"{SUBJECTS[subj]} have {SYMPTOMS[sym]}"


def _need(subj, need):
    return f"{SUBJECTS[subj]} need {NEEDS[need]}"


def _both(subj, sym, need):
    return f"{SUBJECTS[subj]} have {SYMPTOMS[sym]} and need {NEEDS[need]}"


def generate():
    pairs = {}

    def add(gloss, english):
        pairs[gloss] = english

    for g, e in STANDALONE.items():
        add(g, e)
    for g, e in GREETINGS.items():
        add(g, e + ".")
        for a, ae in ADDRESS.items():
            add(f"{g} {a}", f"{e}, {ae}.")
    for sym in SYMPTOMS:
        add(sym, f"I have {SYMPTOMS[sym]}.")           # subject missing -> the signer
    for need in NEEDS:
        add(f"need {need}", f"I need {NEEDS[need]}.")
        add(f"doctor need {need}", f"Doctor, I need {NEEDS[need]}.")
    add("doctor help", "Doctor, please help.")

    clauses = []                                       # (gloss variants, english clause) per subject
    for s, sym in itertools.product(SUBJECTS, SYMPTOMS):
        c = _have(s, sym)
        add(f"{s} {sym}", c + ".")
        clauses.append(([f"{s} {sym}"], c))
    for s, need in itertools.product(SUBJECTS, NEEDS):
        c = _need(s, need)
        add(f"{s} need {need}", c + ".")
        clauses.append(([f"{s} need {need}"], c))
    for s, sym, need in itertools.product(SUBJECTS, SYMPTOMS, NEEDS):
        c = _both(s, sym, need)
        for g in (f"{sym} {s} need {need}", f"{s} {sym} need {need}", f"{s} {sym} {s} need {need}"):
            add(g, c + ".")
            clauses.append(([g], c))
    for s1, s2 in itertools.permutations(SUBJECTS, 2):  # different subjects -> two sentences
        for sym, need in itertools.product(SYMPTOMS, NEEDS):
            add(f"{s1} {sym} {s2} need {need}", f"{_have(s1, sym)}. {_need(s2, need)}.")

    # greeting + doctor in front, thanks after, yes/no in front
    for gl, ge in GREETINGS.items():
        if gl == "thank_you":
            continue
        for variants, c in clauses:
            if not c.startswith("I "):
                continue
            for v in variants:
                add(f"{gl} doctor {v}", f"{ge} doctor, {c}.")
    for variants, c in clauses:
        for v in variants:
            add(f"{v} thank_you", f"{c}. Thank you.")
            add(f"yes {v}", f"Yes, {_low(c.split()[0])} {' '.join(c.split()[1:])}.")
            add(f"no {v}", f"No, {_low(c.split()[0])} {' '.join(c.split()[1:])}.")
    return pairs


def generate_noisy(pairs, seed=0, per_kind=70):
    """Pairs for glosses that carry a stray sign, built from the clean pairs.

    Live runs showed that between two signs the sign model sometimes adds a pronoun
    (a dangling "I" or "you" at the end of a sentence, a doubled "I I") and that the
    translator then invented a symptom to explain it ("no yes I" -> "No, I have a headache").
    These pairs teach it the honest reading: a pronoun with nothing after it is dropped, a
    doubled pronoun counts once, and a pronoun on its own is just that pronoun. They are
    made from the clean pairs by rule, so they grow with the vocabulary. Held-out sentences
    are never used here.
    """
    rng = random.Random(seed)
    pronouns = {g.split()[0] for g in SUBJECTS}                    # "I", "you"
    clean = [(g, e) for g, e in sorted(pairs.items()) if g not in HELD_OUT]
    noisy = {}
    for p in pronouns:                                             # a pronoun on its own
        noisy[p] = SUBJECTS[p] + "."
    ends = [(g, e) for g, e in clean if g.split()[-1] not in pronouns]
    for g, e in rng.sample(ends, min(per_kind, len(ends))):        # dangling pronoun at the end
        noisy[f"{g} {rng.choice(sorted(pronouns))}"] = e
    for g, e in clean:                                             # ... always for the recorded sentences
        if g in RECORDED and g.split()[-1] not in pronouns:
            for p in sorted(pronouns):
                noisy[f"{g} {p}"] = e
    starts = [(g, e) for g, e in clean if g.split()[0] in pronouns]
    for g, e in rng.sample(starts, min(per_kind, len(starts))):    # doubled pronoun at the start
        noisy[f"{g.split()[0]} {g}"] = e
    for g in list(noisy):                                          # never clash with a clean pair
        if g in pairs:
            noisy.pop(g)
    return noisy


def noisy_held_out(pairs):
    """The same kinds of stray sign added to the held-out sentences, only used to test."""
    out = {}
    for g in HELD_OUT_ORDER:
        if g in pairs:
            out[g + " I"] = pairs[g]
            out[g + " you"] = pairs[g]
    return out


def split(pairs):
    train = {g: e for g, e in pairs.items() if g not in HELD_OUT}
    held = {g: pairs[g] for g in HELD_OUT if g in pairs}
    return train, held


# the 20 sentences recorded for the demo; generate() must reproduce them exactly (checked in __main__)
RECORDED = {
    "hello": "Hello.", "good afternoon": "Good afternoon.", "hello doctor": "Hello, doctor.",
    "I need doctor": "I need a doctor.", "you need doctor": "You need a doctor.",
    "I head ache": "I have a headache.", "I stomach ache": "I have a stomach ache.",
    "I need medicine": "I need medicine.", "you need medicine": "You need medicine.",
    "I need water": "I need water.", "I need help": "I need help.", "doctor help": "Doctor, please help.",
    "head ache I need medicine": "I have a headache and need medicine.",
    "stomach ache I need doctor": "I have a stomach ache and need a doctor.",
    "hello doctor I head ache": "Hello doctor, I have a headache.",
    "thank_you": "Thank you.", "thank_you doctor": "Thank you, doctor.", "yes": "Yes.", "no": "No.",
    "good afternoon doctor": "Good afternoon, doctor.",
}

if __name__ == "__main__":
    pairs = generate()
    train, held = split(pairs)
    noisy = generate_noisy(pairs)
    bad = {g: (e, pairs.get(g)) for g, e in RECORDED.items() if pairs.get(g) != e}
    assert not bad, f"generator does not reproduce the recorded sentences: {bad}"
    os.makedirs("Dataset", exist_ok=True)
    rng = random.Random(0)
    rows = list(train.items()); rng.shuffle(rows)
    with open(os.path.join("Dataset", "gloss_pairs.jsonl"), "w", encoding="utf-8") as f:
        for g, e in rows:
            f.write(json.dumps({"glosses": g, "english": e}, ensure_ascii=False) + "\n")
        for g, e in noisy.items():
            f.write(json.dumps({"glosses": g, "english": e, "noisy": True}, ensure_ascii=False) + "\n")
    print(f"{len(noisy)} noisy pairs (stray pronoun) generated")
    print(f"{len(train)} training pairs written to Dataset/gloss_pairs.jsonl; {len(held)} held out:")
    for g, e in held.items():
        print(f"   {g!r} -> {e!r}")
    for g, e in rng.sample(list(train.items()), 8):
        print(f"   e.g. {g!r} -> {e!r}")

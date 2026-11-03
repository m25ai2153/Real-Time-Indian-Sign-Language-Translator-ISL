"""Glosses -> English with a small fine-tuned transformer (T5).

Training: python train_gloss_t5.py   (see gloss_data.py for the examples it learns from)
The model is saved in artifacts/gloss_t5 and loaded from there, so no internet is needed
at run time.

The model runs in its own process (this file started with --serve). TensorFlow (the sign
model) and PyTorch (T5) crash when loaded into the same process on some machines, and a
separate process also keeps the translator's CPU use away from the camera loop.

If the folder is missing, or anything goes wrong, translate() returns the plain glosses
("I stomach ache.") so the demo never breaks.
"""
import atexit
import json
import os
import queue
import re
import subprocess
import sys
import threading

PREFIX = "translate glosses to English: "
DEFAULT_DIR = os.path.join("artifacts", "gloss_t5")
TAG = "@@"                                         # marks protocol lines among any library output
LICENSE_FILE = "licenses.json"                     # written by train_gloss_t5.py next to the model


def build_licenses(pairs, min_pairs=2, min_link=0.9, max_glosses=3):
    """Which sign "licenses" which English word? (computed from the clean training pairs)

    A word is tied to a gloss when they (almost) always appear together, in both directions:
    "headache" appears in every pair that has the gloss "head" and the gloss "head" appears
    in nothing else. Function words ("have", "a", "and") are tied to nothing and stay free.
    Returns {word: [glosses]}; a tied word may only appear in a translation when at least
    one of its glosses was signed. Built from data, so it grows with the vocabulary.
    Only short pairs (at most `max_glosses` signs) are counted: in the long combined
    sentences every word co-occurs with every other and nothing is specific any more.
    """
    word_n, gloss_n, both = {}, {}, {}
    for g, e in pairs.items():
        if len(g.split()) > max_glosses:
            continue
        gs, ws = set(g.split()), set(re.findall(r"[a-z']+", e.lower()))
        for w in ws:
            word_n[w] = word_n.get(w, 0) + 1
        for x in gs:
            gloss_n[x] = gloss_n.get(x, 0) + 1
        for w in ws:
            for x in gs:
                both[(w, x)] = both.get((w, x), 0) + 1
    lic = {}
    for (w, x), n in both.items():
        if word_n[w] >= min_pairs and n / word_n[w] >= min_link and n / gloss_n[x] >= min_link:
            lic.setdefault(w, []).append(x)
    return {w: sorted(v) for w, v in sorted(lic.items())}


def omitted(glosses, english, licenses):
    """Signed glosses whose English word is missing from `english` (empty list = nothing was dropped).

    The other direction of unsupported(): a sign tied to an English word ("head" -> "headache")
    must show up in the translation. When several sentences run together the small model tends
    to drop signs silently; this makes that visible. Signs tied to no word (yes, no, I, you) are not checked.
    """
    words = set(re.findall(r"[a-z']+", english.lower()))
    tied = {}
    for w, gs in licenses.items():
        for g in gs:
            tied.setdefault(g, set()).add(w)
    seen, out = set(), []
    for g in glosses:
        for part in str(g).split():
            if part in tied and part not in seen and not tied[part] & words:
                out.append(part)
            seen.add(part)
    return out


def unsupported(glosses, english, licenses):
    """English words in `english` that need a sign which was not signed (empty list = fine)."""
    signed = set()
    for g in glosses:
        signed.update(str(g).split())
    words = set(re.findall(r"[a-z']+", english.lower()))
    return sorted(w for w in words if w in licenses and not signed.intersection(licenses[w]))


def plain(glosses):
    text = " ".join(str(g).replace("_", " ") for g in glosses)
    return (text[:1].upper() + text[1:] + ".") if text else ""


class GlossTranslator:
    def __init__(self, model_dir=DEFAULT_DIR, timeout=15.0, startup_timeout=180.0, python=None):
        self.proc = None
        self.timeout = timeout
        self.cache = {}
        self.note = ""
        self.licenses = {}
        self.guard_note = {}                        # signs tuple -> why the translation was replaced by plain glosses
        self._lock = threading.Lock()
        self._lines = queue.Queue()
        if not os.path.isdir(model_dir):
            self.note = f"no model in {model_dir} (run train_gloss_t5.py); showing plain glosses"
            return
        try:
            with open(os.path.join(model_dir, LICENSE_FILE), encoding="utf-8") as f:
                self.licenses = json.load(f)
        except (OSError, ValueError):
            self.note = "no licenses.json (re-run train_gloss_t5.py): the invented-word check is off"
        try:
            self.proc = subprocess.Popen(
                [python or sys.executable, os.path.abspath(__file__), "--serve", model_dir],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8", bufsize=1)
            threading.Thread(target=self._read, daemon=True).start()
            first = self._lines.get(timeout=startup_timeout)
            if first.get("ready") is not True:
                raise RuntimeError(first.get("error", "translator did not start"))
            atexit.register(self.close)
        except Exception as e:                      # never stop the demo for the translator
            self.note = f"could not start the translator: {e}; showing plain glosses"
            self.close()

    def _read(self):
        for line in self.proc.stdout:
            if line.startswith(TAG):
                try:
                    self._lines.put(json.loads(line[len(TAG):]))
                except ValueError:
                    pass
        self._lines.put({"error": "translator process ended"})

    @property
    def ready(self):
        return self.proc is not None and self.proc.poll() is None

    def close(self):
        proc, self.proc = self.proc, None
        if proc is not None and proc.poll() is None:
            try:
                proc.stdin.close()
                proc.wait(timeout=2)
            except Exception:
                proc.kill()

    def translate(self, glosses, announce=False):
        """English for the glosses. announce=True prints a console line when the guard replaced it
        (use it for finished sentences; half-signed sentences change every second and would flood the console)."""
        glosses = [str(g) for g in glosses]
        if not glosses:
            return ""
        key = tuple(glosses)
        if key in self.cache:
            if announce and key in self.guard_note:
                print(self.guard_note[key])
            return self.cache[key]
        if not self.ready:
            return plain(glosses)
        with self._lock:                            # one request at a time
            try:
                while not self._lines.empty():
                    self._lines.get_nowait()
                self.proc.stdin.write(TAG + json.dumps(glosses) + "\n")
                self.proc.stdin.flush()
                reply = self._lines.get(timeout=self.timeout)
                text = reply.get("text", "").strip() or plain(glosses)
            except Exception:
                return plain(glosses)
        extra = unsupported(glosses, text, self.licenses) if self.licenses else []
        dropped = omitted(glosses, text, self.licenses) if self.licenses else []
        if extra or dropped:                        # invented a word, or silently dropped a sign: show the glosses instead
            why = (f"{' '.join(extra)!r} is not in the signs" if extra else f"the sign(s) {' '.join(dropped)!r} are missing from the English")
            self.guard_note[key] = f"  translator guard: {why} ({' '.join(glosses)}); showing the glosses"
            if announce:
                print(self.guard_note[key])
            text = plain(glosses)
        self.cache[key] = text
        if len(self.cache) > 200:
            self.cache.pop(next(iter(self.cache)))
        return text


def _serve(model_dir):
    """Child process: load the model once, then answer one request per line."""
    def send(obj):
        sys.stdout.write(TAG + json.dumps(obj, ensure_ascii=False) + "\n")
        sys.stdout.flush()
    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        torch.set_num_threads(2)                    # leave CPU for MediaPipe and the sign model
        tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, local_files_only=True).eval()

        def run(glosses):
            enc = tok(PREFIX + " ".join(glosses), return_tensors="pt")
            with torch.no_grad():
                out = model.generate(**enc, num_beams=4, max_new_tokens=64)
            return tok.decode(out[0], skip_special_tokens=True).strip()
        run(["hello"])                              # warm up so the first real sentence is not slow
    except Exception as e:
        send({"error": f"{type(e).__name__}: {e}"})
        return
    send({"ready": True})
    for line in sys.stdin:
        if not line.startswith(TAG):
            continue
        try:
            send({"text": run(json.loads(line[len(TAG):]))})
        except Exception as e:
            send({"error": str(e)})


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--serve":
        _serve(sys.argv[2])
    else:
        print(__doc__)

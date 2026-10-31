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
import subprocess
import sys
import threading

PREFIX = "translate glosses to English: "
DEFAULT_DIR = os.path.join("artifacts", "gloss_t5")
TAG = "@@"                                         # marks protocol lines among any library output


def plain(glosses):
    text = " ".join(str(g).replace("_", " ") for g in glosses)
    return (text[:1].upper() + text[1:] + ".") if text else ""


class GlossTranslator:
    def __init__(self, model_dir=DEFAULT_DIR, timeout=15.0, startup_timeout=180.0, python=None):
        self.proc = None
        self.timeout = timeout
        self.cache = {}
        self.note = ""
        self._lock = threading.Lock()
        self._lines = queue.Queue()
        if not os.path.isdir(model_dir):
            self.note = f"no model in {model_dir} (run train_gloss_t5.py); showing plain glosses"
            return
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

    def translate(self, glosses):
        glosses = [str(g) for g in glosses]
        if not glosses:
            return ""
        key = tuple(glosses)
        if key in self.cache:
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

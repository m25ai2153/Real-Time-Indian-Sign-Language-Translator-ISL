"""Turns the model's per-frame probabilities into discrete glosses (numpy only).

Why not just "probability above a threshold for N frames"?
  * Signed at conversational speed, a sign only keeps the model confident for
    a few frames, and with several classes the peak is often 0.6-0.8, not 0.9.
  * When two glosses look alike (hello vs head) the model flips between them
    at around 0.5/0.5. Printing either one is a guess.

How it works: a "candidate" opens when a non-idle gloss is the top class with
at least `enter` probability AND clearly ahead of the runner-up (`margin`).
  * EARLY commit: `confirm` consecutive predictions >= `trigger` -> print now.
  * LATE commit: if the sign was brief and never reached `trigger` twice, the
    candidate is still printed when it closes, provided it lasted
    `min_frames` predictions and its peak reached `min_peak`.
  * One candidate = one gloss. It closes when the gloss' probability falls
    below `exit` (the sign has passed) or another gloss takes over, so holding
    a sign prints it once, and the next sign can follow immediately.
  * `cooldown` (per gloss, seconds) stops flicker from printing duplicates.
Ambiguous frames (small margin) never open a candidate: better to print
nothing than the wrong word.
"""
import numpy as np


class GlossDetector:
    def __init__(self, actions, idle="idle", enter=0.55, trigger=0.75, confirm=2,
                 exit=0.35, min_peak=0.60, min_frames=2, margin=0.20, cooldown=0.5):
        self.actions = [str(a) for a in actions]
        self.idle_idx = self.actions.index(idle) if idle in self.actions else -1
        self.enter, self.trigger, self.confirm = enter, trigger, confirm
        self.exit, self.min_peak, self.min_frames = exit, min_peak, min_frames
        self.margin, self.cooldown = margin, cooldown
        self.cand = None
        self.last_t = {}

    def _commit(self, idx, prob, now):
        if now - self.last_t.get(idx, -1e9) < self.cooldown:
            return []
        self.last_t[idx] = now
        return [(self.actions[idx], float(prob))]

    @property
    def candidate(self):
        """Label currently being tracked (for the on-screen display), or None."""
        return None if self.cand is None else self.actions[self.cand["idx"]]

    def update(self, probs, now):
        """Feed one probability vector. Returns a list of (label, prob) to print."""
        probs = np.asarray(probs, dtype=float)
        order = np.argsort(probs)[::-1]
        idx = int(order[0])
        p = float(probs[idx])
        second = float(probs[order[1]]) if len(order) > 1 else 0.0
        events = []
        c = self.cand

        # 1. close the open candidate: its sign has passed, or another gloss took over
        if c is not None:
            switched = idx != c["idx"] and idx != self.idle_idx and p >= self.enter
            if probs[c["idx"]] < self.exit or switched:
                if (not c["committed"] and c["n"] >= self.min_frames
                        and c["peak"] >= self.min_peak):
                    events += self._commit(c["idx"], c["peak"], now)
                self.cand = c = None

        # 2. open or update a candidate
        if idx != self.idle_idx and p >= self.enter and p - second >= self.margin:
            if c is None or c["idx"] != idx:
                c = self.cand = {"idx": idx, "peak": p, "n": 1,
                                 "hi": 1 if p >= self.trigger else 0, "committed": False}
            else:
                c["peak"] = max(c["peak"], p)
                c["n"] += 1
                c["hi"] = c["hi"] + 1 if p >= self.trigger else 0
            if not c["committed"] and c["hi"] >= self.confirm:
                events += self._commit(idx, p, now)
                c["committed"] = True
        return events

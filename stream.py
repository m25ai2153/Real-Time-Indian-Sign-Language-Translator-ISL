"""Clean-up of the printed gloss stream (pure Python).

The detector prints a gloss every time the model settles on one. Signs that have two
phases (ache, morning, afternoon) or last a second or two make the model switch label in
the middle of ONE sign, which shows up as repeats and in-betweens. Two simple rules, both
checked against real recordings of continuous signing:

  * repeat:   the same gloss again within `repeat_gap` seconds of its previous print is the
              same sign continuing, not a new one.
  * sandwich: A, B, A with each step shorter than `sandwich_gap` seconds (e.g. ache,
              afternoon, ache) is one sign A whose middle looked like B; B is dropped.

Everything else passes through untouched. The detector keeps printing the raw events; this
class decides what goes into the sentence. Tune the two numbers below.
"""

REPEAT_GAP = 2.0        # seconds
SANDWICH_GAP = 1.2      # seconds


class GlossStream:
    def __init__(self, repeat_gap=REPEAT_GAP, sandwich_gap=SANDWICH_GAP, max_items=8):
        self.repeat_gap, self.sandwich_gap, self.max_items = repeat_gap, sandwich_gap, max_items
        self.items = []                      # [label, time of its latest sighting]

    @property
    def labels(self):
        return [label for label, _ in self.items]

    def clear(self):
        self.items = []

    def push(self, label, t):
        """Add a detected gloss. Returns (kept, note); note says why it was not added."""
        it = self.items
        if it and it[-1][0] == label and t - it[-1][1] < self.repeat_gap:
            it[-1][1] = t                    # the same sign is still going on
            return False, "repeat"
        if (len(it) >= 2 and it[-2][0] == label and it[-1][0] != label
                and t - it[-1][1] < self.sandwich_gap and it[-1][1] - it[-2][1] < self.sandwich_gap):
            dropped = it.pop()[0]
            it[-1][1] = t
            return False, f"dropped '{dropped}' between two '{label}'"
        it.append([label, t])
        del it[:-self.max_items]
        return True, ""
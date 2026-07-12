"""Build the COMPLETE card template bank from hand-labelled harvested faces.

Labels were assigned offline by eye from harvest/montage.png (permitted for
template building; no VLM, no runtime model). Ambiguous/two-card/noisy crops
were dropped -- every rank and suit still has multiple clean exemplars.
Also folds in the clean hand-01 exemplars via build_card_templates.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
import cv2
import read_cards as rc
import read_hero as rh

HARV = "/Users/jasonli/Documents/GitHub/PokerTrainer/cv_lab/harvest"

# idx -> card label (rank+suit). Dropped: 5,6,16,25,29,30,31,52,62,69,70 (ambiguous).
LABELS = {
    0: "9d", 1: "Ac", 2: "5s", 3: "7h", 7: "6h", 8: "4h", 9: "7s", 10: "8c",
    11: "Tc", 12: "Ah", 13: "Kd", 14: "Tc", 15: "Tc", 17: "7h", 18: "Tc",
    19: "2h", 20: "Jd", 21: "Ks", 22: "Kh", 23: "Ks", 24: "Kh", 26: "Kh",
    27: "Qh", 28: "9h", 32: "2c", 33: "Tc", 34: "Qc", 35: "Jh", 36: "Qc",
    37: "Jh", 38: "8d", 39: "Qc", 40: "Jh", 41: "Qc", 42: "7c", 43: "9h",
    44: "Jh", 45: "Qc", 46: "Jh", 47: "Jh", 48: "Qc", 49: "Qd", 50: "Qs",
    51: "Qd", 53: "Td", 54: "3s", 55: "9c", 56: "Qd", 57: "6c", 58: "Qs",
    59: "Qd", 60: "Jc", 61: "Qd", 63: "Qd", 64: "Qd", 65: "Qd", 66: "4h",
    67: "Kh", 68: "Kh", 71: "9d",
}

meta = {m["idx"]: m for m in json.load(open(f"{HARV}/harvest_meta.json"))}
ranks, suits = {}, {}


def add(card, rg, sg):
    ranks.setdefault(card[:-1], []).append(rg)
    suits.setdefault(card[-1], []).append(sg)


for idx, card in LABELS.items():
    crop = cv2.imread(f"{HARV}/card_{idx:03d}.png")
    if crop is None:
        print("missing", idx); continue
    tag = meta[idx]["tag"]
    if tag.startswith("hero"):
        rsub = rh._sub(crop, rh.IDX_X0, rh.IDX_X1, rh.RANK_Y0, rh.RANK_Y1)
        ssub = rh._sub(crop, rh.IDX_X0, rh.IDX_X1, rh.SUIT_Y0, rh.SUIT_Y1)
        add(card, rh._glyph(rsub, rc.RANK_SIZE), rh._glyph(ssub, rc.SUIT_SIZE))
    else:
        add(card, rc._rank_glyph(crop), rc._suit_glyph(crop))

tmpl = {"ranks": ranks, "suits": suits}
rc.save_templates(tmpl, "/Users/jasonli/Documents/GitHub/PokerTrainer/cv_lab/models/card_templates.npz")
print("ranks:", {k: len(v) for k, v in sorted(ranks.items())})
print("suits:", {k: len(v) for k, v in sorted(suits.items())})

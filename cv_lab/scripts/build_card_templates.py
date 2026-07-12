"""Rebuild card_templates.npz with tight-crop glyph framing.

Templates are LABELLED offline (I know these hands' cards); runtime is pure
template match. Board slots + the two hero index corners feed the same bank so
board and hero share one set of rank/suit templates.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import cv2
from landmark_anchor import anchor
import read_cards as rc
import read_hero as rh

DF = "/Users/jasonli/Documents/GitHub/PokerTrainer/cv_lab/hand01/decision_frames"
ST = "/Users/jasonli/Documents/GitHub/PokerTrainer/cv_lab/hand01/spade_test"

# (image_path, [cards left->right]) for board slots
BOARD_LABELS = [
    (f"{DF}/t0360.00s.png", ["8d", "Tc", "7c", "9h"]),
    (f"{DF}/t0390.00s.png", ["8d", "Tc", "7c", "9h", "Kd"]),
    (f"{ST}/t0256.00s.png", ["6h", "2c", "Ks"]),
    (f"{ST}/t0272.00s.png", ["6h", "2c", "Ks", "Ts"]),
]
# hero index corners (image, [left_card, right_card])
HERO_LABELS = [
    (f"{DF}/t0360.00s.png", ["Qc", "Jh"]),
]

ranks, suits = {}, {}

def add(rk, su, rank_glyph, suit_glyph):
    ranks.setdefault(rk, []).append(rank_glyph)
    suits.setdefault(su, []).append(suit_glyph)

for path, cards in BOARD_LABELS:
    img = cv2.imread(path)
    a = anchor(img)
    for slot, card in zip(rc.BOARD_SLOTS, cards):
        crop = rc.slot_crop(img, a["map_roi"], slot)
        if not rc.card_present(crop):
            print("WARN: no card at", path, card); continue
        rank = card[:-1].replace("10", "T")
        add(rank, card[-1], rc._rank_glyph(crop), rc._suit_glyph(crop))

for path, cards in HERO_LABELS:
    img = cv2.imread(path)
    a = anchor(img)
    x0, x1, y0, y1 = a["map_roi"](rh.hero_roi_box())
    roi = img[max(y0, 0):y1, max(x0, 0):x1]
    for face_spec, card in zip(rh.FACES, cards):
        face = rh._deskew(roi, face_spec)
        rsub = rh._sub(face, rh.IDX_X0, rh.IDX_X1, rh.RANK_Y0, rh.RANK_Y1)
        ssub = rh._sub(face, rh.IDX_X0, rh.IDX_X1, rh.SUIT_Y0, rh.SUIT_Y1)
        add(card[:-1], card[-1], rh._glyph(rsub, rc.RANK_SIZE), rh._glyph(ssub, rc.SUIT_SIZE))

tmpl = {"ranks": ranks, "suits": suits}
rc.save_templates(tmpl, "/Users/jasonli/Documents/GitHub/PokerTrainer/cv_lab/models/card_templates.npz")
print("ranks:", {k: len(v) for k, v in ranks.items()})
print("suits:", {k: len(v) for k, v in suits.items()})

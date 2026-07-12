import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import cv2, numpy as np
from landmark_anchor import anchor
from read_pot import load_bank
import read_seats, read_pills, read_markers

path = sys.argv[1] if len(sys.argv) > 1 else \
    "/Users/jasonli/Documents/GitHub/PokerTrainer/cv_lab/hand01/decision_frames/t0328.00s.png"
img = cv2.imread(path)
a = anchor(img)
scale = a["s"]
bank = load_bank("/Users/jasonli/Documents/GitHub/PokerTrainer/cv_lab/models/pot_digits.npz")
outdir = "/Users/jasonli/Documents/GitHub/PokerTrainer/cv_lab/hand01/full_probe"
os.makedirs(outdir, exist_ok=True)

print(f"=== {os.path.basename(path)}  s={scale:.4f} ===")
EXPECT_BET = {0:3.0,2:3.0,3:1.0,5:0.5,7:12.0}
seats = read_seats.read_seats(img, a["map_roi"], bank, with_bets=True)
pills = read_pills.read_pills(img, a["map_roi"], word_bank=None)
bok = sum(1 for k,v in EXPECT_BET.items() if seats[k]["bet"]==v)
print(f"(bets expected {EXPECT_BET})")
for idx in range(8):
    # save crops
    bx = a["map_roi"](read_seats._seat_bet_box(idx))
    px = a["map_roi"](read_pills._pill_box(idx))
    cv2.imwrite(f"{outdir}/seat{idx}_bet.png", img[max(bx[2],0):bx[3], max(bx[0],0):bx[1]])
    cv2.imwrite(f"{outdir}/seat{idx}_pill.png", img[max(px[2],0):px[3], max(px[0],0):px[1]])
    s, p = seats[idx], pills[idx]
    print(f"seat{idx}: stack={s['stack']} bet={s['bet']} '{s['bet_raw']}' | "
          f"pill present={p['present']} color={p['color']}")

print(f"BETS {bok}/{len(EXPECT_BET)}")
act, ainfo = read_markers.active_seat(img, a["map_roi"], scale)
dlr, dinfo = read_markers.dealer_seat(img, a["map_roi"], scale)
print("ACTIVE seat:", act, ainfo)
print("DEALER seat:", dlr, dinfo)

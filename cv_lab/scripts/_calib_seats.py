import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import cv2, numpy as np
from landmark_anchor import anchor, REF_SEAT_COINS
from read_pot import load_bank
import read_seats

EXPECT = {0:158.90,1:99.50,2:141.20,3:109.80,4:1156.60,5:170.40,6:288.20,7:244.30}
img = cv2.imread(sys.argv[1] if len(sys.argv)>1 else
    "/Users/jasonli/Documents/GitHub/PokerTrainer/cv_lab/hand01/decision_frames/t0360.00s.png")
a = anchor(img)
print("anchor:", None if a is None else f"s={a['s']:.4f} seats={a['n_seats']}")
bank = load_bank("/Users/jasonli/Documents/GitHub/PokerTrainer/cv_lab/models/pot_digits.npz")

outdir = "/Users/jasonli/Documents/GitHub/PokerTrainer/cv_lab/hand01/seat_probe"
os.makedirs(outdir, exist_ok=True)
res = read_seats.read_seats(img, a["map_roi"], bank, with_bets=True)
ok = 0
for r in res:
    idx = r["seat"]
    x0,x1,y0,y1 = a["map_roi"](read_seats._seat_stack_box(idx))
    crop = img[max(y0,0):y1, max(x0,0):x1]
    cv2.imwrite(f"{outdir}/seat{idx}_stack.png", crop)
    exp = EXPECT.get(idx)
    hit = "OK" if r["stack"]==exp else "MISS"
    if r["stack"]==exp: ok+=1
    print(f"seat{idx}: stack={r['stack']} raw='{r['stack_raw']}' exp={exp} {hit}  bet={r['bet']} '{r['bet_raw']}'")
print(f"STACKS {ok}/{len(EXPECT)}")

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import cv2, numpy as np
from landmark_anchor import anchor, REF_SEAT_COINS
import read_seats, read_pills

path = sys.argv[1] if len(sys.argv) > 1 else \
    "/Users/jasonli/Documents/GitHub/PokerTrainer/cv_lab/hand01/decision_frames/t0328.00s.png"
img = cv2.imread(path)
a = anchor(img); scale = a["s"]
vis = img.copy()
for idx in range(8):
    bx = a["map_roi"](read_seats._seat_bet_box(idx))
    px = a["map_roi"](read_pills._pill_box(idx))
    sx0,sx1,sy0,sy1 = a["map_roi"](read_seats._seat_stack_box(idx))
    cv2.rectangle(vis,(bx[0],bx[2]),(bx[1],bx[3]),(0,255,255),3)   # bet yellow
    cv2.rectangle(vis,(px[0],px[2]),(px[1],px[3]),(255,0,255),3)   # pill magenta
    cv2.rectangle(vis,(sx0,sy0),(sx1,sy1),(0,255,0),2)            # stack green
    # seat coin center
    cxx = a["map_roi"]((REF_SEAT_COINS[idx][0],REF_SEAT_COINS[idx][0],REF_SEAT_COINS[idx][1],REF_SEAT_COINS[idx][1]))
    cv2.circle(vis,(cxx[0],cxx[2]),6,(0,0,255),-1)
    cv2.putText(vis,str(idx),(cxx[0]+8,cxx[2]),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255),2)

# blue + white masks
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
blue = cv2.inRange(hsv,(100,120,120),(130,255,255))
g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
white = ((hsv[:,:,1]<60)&(g>180)).astype(np.uint8)*255

out = "/Users/jasonli/Documents/GitHub/PokerTrainer/cv_lab/hand01/full_probe"
os.makedirs(out, exist_ok=True)
def small(im):
    h,w=im.shape[:2]; return cv2.resize(im,(w//2,h//2))
cv2.imwrite(f"{out}/overlay.jpg", small(vis), [cv2.IMWRITE_JPEG_QUALITY,80])
cv2.imwrite(f"{out}/mask_blue.jpg", small(cv2.cvtColor(blue,cv2.COLOR_GRAY2BGR)))
cv2.imwrite(f"{out}/mask_white.jpg", small(cv2.cvtColor(white,cv2.COLOR_GRAY2BGR)))
print("wrote overlay, mask_blue, mask_white to", out)

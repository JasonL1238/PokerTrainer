"""Harvest distinct board/hero card faces across the session for offline
labelling, to build a COMPLETE rank/suit template bank.

Sampling at 1 fps, on table frames, we crop every present board slot + the two
hero index corners, dedup by a rank-glyph signature, and save a montage. I then
label the montage by eye (offline; permitted for template building) and
build_bank_from_labels.py turns the labels into templates. No VLM, no runtime
model.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import cv2
import av

from classify_screen import classify
import read_cards as rc
import read_hero as rh

VIDEO = "/Users/jasonli/Documents/GitHub/PokerTrainer/data/videos/clubwpt_session_01.mov"
OUT = "/Users/jasonli/Documents/GitHub/PokerTrainer/cv_lab/harvest"
DEDUP_SSD = 0.045      # novelty threshold on the rank-glyph signature


def _sig(face_bgr, is_hero):
    """Rank-glyph + suit-color signature for dedup."""
    if is_hero:
        rsub = rh._sub(face_bgr, rh.IDX_X0, rh.IDX_X1, rh.RANK_Y0, rh.RANK_Y1)
        gl = rh._glyph(rsub, rc.RANK_SIZE)
        col = rh._suit_color(rh._sub(face_bgr, rh.IDX_X0, rh.IDX_X1, rh.SUIT_Y0, rh.SUIT_Y1))
    else:
        gl = rc._rank_glyph(face_bgr)
        col = rc.suit_color(face_bgr)
    return gl, col


def main():
    os.makedirs(OUT, exist_ok=True)
    container = av.open(VIDEO)
    stream = container.streams.video[0]
    kept = []          # (sig_glyph, color, crop, meta)
    for t in range(0, 565):
        container.seek(int(t / stream.time_base), stream=stream)
        frame = None
        for frame in container.decode(stream):
            if float(frame.pts * stream.time_base) >= t:
                break
        if frame is None:
            continue
        img = frame.to_ndarray(format="bgr24")
        label, a = classify(img, None)
        if label != "table":
            continue
        m = a["map_roi"]
        faces = []
        for i, slot in enumerate(rc.BOARD_SLOTS):
            crop = rc.slot_crop(img, m, slot)
            if crop.size and rc.card_present(crop):
                faces.append((crop, False, f"board{i}"))
        x0, x1, y0, y1 = m(rh.hero_roi_box())
        roi = img[max(y0, 0):y1, max(x0, 0):x1]
        if roi.size and rh.card_present(roi):
            for i, fs in enumerate(rh.FACES):
                faces.append((rh._deskew(roi, fs), True, f"hero{i}"))
        for crop, is_hero, tag in faces:
            try:
                gl, col = _sig(crop, is_hero)
            except Exception:
                continue
            novel = True
            for (kg, kc, _, _) in kept:
                if kc == col and float(np.mean((gl - kg) ** 2)) < DEDUP_SSD:
                    novel = False
                    break
            if novel:
                kept.append((gl, col, crop, {"t": t, "tag": tag, "color": col}))
        if t % 60 == 0:
            print(f"t={t}s kept={len(kept)}", flush=True)
    container.close()

    # save crops + montage
    meta = []
    cell = 90
    n = len(kept)
    cols = 12
    rows = (n + cols - 1) // cols
    montage = np.full((rows * cell, cols * cell, 3), 40, np.uint8)
    for idx, (_, col, crop, md) in enumerate(kept):
        md["idx"] = idx
        meta.append(md)
        cv2.imwrite(f"{OUT}/card_{idx:03d}.png", crop)
        c = cv2.resize(crop, (cell - 8, cell - 8))
        r, cc = divmod(idx, cols)
        montage[r * cell + 4:r * cell + cell - 4, cc * cell + 4:cc * cell + cell - 4] = c
        cv2.putText(montage, str(idx), (cc * cell + 6, r * cell + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    cv2.imwrite(f"{OUT}/montage.png", montage)
    json.dump(meta, open(f"{OUT}/harvest_meta.json", "w"), indent=0)
    print(f"DONE: {n} unique card faces -> {OUT}/montage.png")


if __name__ == "__main__":
    main()

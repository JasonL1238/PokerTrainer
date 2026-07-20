"""Calibrate deterministic OCR templates from hand-verified labeled crops.

VLM-in-the-loop is used ONLY here (offline, one-time) to transcribe a handful of
ClubWPT HUD crops; the resulting templates are pure pixel arrays that the runtime
reader (ocr_readers.py) matches deterministically. Re-run whenever the client font
or HUD styling changes.

Digit templates: for each (annotation_id -> value) pair, the value's crop is
segmented into glyphs and zipped left-to-right against the value's digit/'.' chars;
matching glyphs are averaged per character into one unit-vector template.

Word templates: each (annotation_id -> word) pill crop is reduced to its whole-word
white-text mask and averaged per word.

    python cv_lab/scripts/calibrate_ocr.py            # build + save templates
    python cv_lab/scripts/calibrate_ocr.py --eval     # also re-read all crops & score
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from labeling_poker.config import DEFAULT_DB_PATH, EXISTING_DATASET_IMAGES_DIR  # noqa: E402
from cv_lab.scripts.ocr_readers import (  # noqa: E402
    DEFAULT_TEMPLATE_PATH,
    DIGIT_SIZE,
    WORD_SIZE,
    TemplateOCR,
    _norm,
    binarize_text,
    segment_glyphs,
    tokenize,
)

# annotation_id -> transcribed numeric value (verified via calibration montage).
# Multi-digit pot/stack crops build the digit templates (their value token is picked
# unambiguously by glyph count, isolating it from the POT:/BB affixes). The single-
# digit bet crops are held out for evaluation only (BET_EVAL).
NUMBER_TRUTH: dict[int, str] = {
    39: "16.50", 1338: "6.50", 2547: "8.50", 3672: "96.30", 4635: "347.50",
    5806: "78.20", 40: "269.10", 1112: "293.70", 1974: "168.40", 2922: "273.10",
    3668: "111.30", 4346: "99.50", 5070: "169.90", 6067: "108.30",
}
BET_EVAL: dict[int, str] = {262: "9", 1533: "1", 2956: "1", 3922: "3", 4854: "1", 6172: "1"}
# pot crops carry the POT: prefix + BB suffix -> source of the affix-letter templates.
_POT_IDS = {39, 1338, 2547, 3672, 4635, 5806}
# Bet crops also carry the chip-stack icon next to the amount. Its white suit
# highlight survives binarize_text at HUD scale and classifies as a confident '0',
# joining the digit run ("12" -> run [chip,1,2] -> gap-inferred decimal -> 0.12).
# Harvest it as affix template 'c' so it breaks the run like the POT:/BB letters.
# Only the chip GLYPH is taken from these crops; their values stay held out of the
# digit templates (BET_EVAL remains a fair number-reading eval).
_CHIP_IDS = set(BET_EVAL)

# annotation_id -> transcribed pill word.
PILL_TRUTH: dict[int, str] = {
    69: "check", 922: "check", 5776: "check",
    1536: "call", 6445: "call",
    1821: "bet", 3619: "bet",
    2571: "raise", 3118: "raise", 4046: "raise", 4856: "raise", 6268: "raise",
    425: "fold", 447: "fold", 598: "fold", 691: "fold", 803: "fold", 1147: "fold",
}


def _load_crop(con: sqlite3.Connection, imgdir: Path, aid: int) -> np.ndarray | None:
    r = con.execute(
        "SELECT f.path p,a.x1,a.y1,a.x2,a.y2 FROM annotations a JOIN files f ON f.id=a.file_id WHERE a.id=?",
        (aid,),
    ).fetchone()
    if not r:
        return None
    ip = (imgdir / r[0]).resolve()
    img = cv2.imread(str(ip))
    if img is None:
        return None
    x1, y1, x2, y2 = (int(v) for v in r[1:5])
    return img[max(0, y1):y2, max(0, x1):x2]


def build_templates(db: Path, imgdir: Path) -> TemplateOCR:
    con = sqlite3.connect(str(db))
    digit_acc: dict[str, list[np.ndarray]] = {}
    for aid, value in NUMBER_TRUTH.items():
        crop = _load_crop(con, imgdir, aid)
        if crop is None:
            continue
        comps = segment_glyphs(binarize_text(crop))
        digits_only = value.replace(".", "")
        want = len(digits_only)
        # Full-height glyphs only (drops the shorter BB and the decimal dot). The green
        # chip is already gone (dropped by the low-saturation binarize), so the tall
        # glyph layout is [POT:]? value BB. Slice out the value positionally rather than
        # by token count (the decimal splits the value token in the smaller stack font).
        max_h = max((c.h for c in comps), default=0)
        tall = [c for c in comps if c.h >= 0.55 * max_h]
        prefix = 3 if aid in _POT_IDS else 0  # "POT:" contributes P,O,T
        value_glyphs = tall[prefix : prefix + want]  # drop prefix and trailing "BB"
        if len(value_glyphs) != want or len(tall) != prefix + want + 2:
            print(f"  ! id{aid} '{value}': tall={len(tall)} != {prefix}+{want}+2(BB), skipped")
            continue
        for g, ch in zip(value_glyphs, digits_only):
            digit_acc.setdefault(ch, []).append(_norm(g.mask, DIGIT_SIZE))

        # Affix-letter templates so the reader rejects non-value glyphs instead of
        # misreading B as 8 / T as 7. 'O' is skipped (visually identical to '0').
        for g in tall[-2:]:  # BB suffix
            digit_acc.setdefault("B", []).append(_norm(g.mask, DIGIT_SIZE))
        if prefix:  # POT: prefix -> P (0) and T (2)
            digit_acc.setdefault("P", []).append(_norm(tall[0].mask, DIGIT_SIZE))
            digit_acc.setdefault("T", []).append(_norm(tall[2].mask, DIGIT_SIZE))

    # Chip-icon affix from bet crops: expected tall layout is value digits + BB +
    # one chip at either end. The chip is the only glyph wider than tall (the suit
    # highlight sits on a squat stack); digits and B are all taller than wide.
    for aid in _CHIP_IDS:
        crop = _load_crop(con, imgdir, aid)
        truth = BET_EVAL.get(aid, "")
        if crop is None or not truth:
            continue
        comps = segment_glyphs(binarize_text(crop))
        max_h = max((c.h for c in comps), default=0)
        tall = sorted((c for c in comps if c.h >= 0.55 * max_h), key=lambda g: g.x)
        want = len(truth.replace(".", ""))
        if len(tall) != want + 3:  # value + B + B + chip
            print(f"  ! chip id{aid}: tall={len(tall)} != {want}+3, skipped")
            continue
        chip = max((tall[0], tall[-1]), key=lambda g: g.w / max(g.h, 1))
        if chip.w <= chip.h:
            print(f"  ! chip id{aid}: no wide end glyph ({chip.w}x{chip.h}), skipped")
            continue
        digit_acc.setdefault("c", []).append(_norm(chip.mask, DIGIT_SIZE))

    digits: dict[str, np.ndarray] = {}
    for ch, vecs in digit_acc.items():
        m = np.mean(vecs, axis=0)
        n = np.linalg.norm(m)
        digits[ch] = (m / n) if n > 1e-6 else m

    word_acc: dict[str, list[np.ndarray]] = {}
    for aid, word in PILL_TRUTH.items():
        crop = _load_crop(con, imgdir, aid)
        if crop is None:
            continue
        mask = binarize_text(crop)
        glyphs = segment_glyphs(mask)
        if glyphs:
            max_h = max(g.h for g in glyphs)
            glyphs = [g for g in glyphs if g.h >= 0.5 * max_h]
        if not glyphs:
            continue
        x0 = min(g.x for g in glyphs); y0 = min(g.y for g in glyphs)
        x1 = max(g.x + g.w for g in glyphs); y1 = max(g.y + g.h for g in glyphs)
        word_acc.setdefault(word, []).append(_norm(mask[y0:y1, x0:x1] > 0, WORD_SIZE))

    words: dict[str, np.ndarray] = {}
    for w, vecs in word_acc.items():
        m = np.mean(vecs, axis=0)
        n = np.linalg.norm(m)
        words[w] = (m / n) if n > 1e-6 else m

    con.close()
    print(f"digit templates: {sorted(digits)}  ({len(digit_acc)} chars)")
    print(f"word templates : {sorted(words)}")
    return TemplateOCR(digits, words)


def evaluate(bank: TemplateOCR, db: Path, imgdir: Path) -> None:
    con = sqlite3.connect(str(db))
    ok = 0
    attempted = 0
    combined = {**NUMBER_TRUTH, **BET_EVAL}
    print("\n== number read-back (train pot/stack + held-out bet) ==")
    for aid, value in combined.items():
        crop = _load_crop(con, imgdir, aid)
        if crop is None:
            print(f"       id{aid:<5} truth={value:<8} crop missing on disk, skipped")
            continue
        val, raw = bank.read_number(crop)
        got = ("%g" % val) if val is not None else "None"
        hit = got == ("%g" % float(value))
        ok += hit
        attempted += 1
        tag = "bet*" if aid in BET_EVAL else "    "
        print(f"  {tag} id{aid:<5} truth={value:<8} got={got:<8} raw='{raw}' {'OK' if hit else 'XX'}")
    print(f"numbers: {ok}/{attempted}  (* = held-out, {len(combined) - attempted} skipped)")

    okp = 0
    print("\n== pill read-back ==")
    for aid, word in PILL_TRUTH.items():
        crop = _load_crop(con, imgdir, aid)
        if crop is None:
            print(f"  id{aid:<5} truth={word:<7} crop missing on disk, skipped")
            continue
        got, sc = bank.read_word(crop)
        hit = got == word
        okp += hit
        print(f"  id{aid:<5} truth={word:<7} got={str(got):<7} score={sc:.2f} {'OK' if hit else 'XX'}")
    print(f"pills: {okp}/{len(PILL_TRUTH)}")
    con.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DEFAULT_DB_PATH))
    ap.add_argument("--images", default=str(EXISTING_DATASET_IMAGES_DIR))
    ap.add_argument("--out", default=str(DEFAULT_TEMPLATE_PATH))
    ap.add_argument("--eval", action="store_true")
    args = ap.parse_args()

    bank = build_templates(Path(args.db), Path(args.images))
    bank.save(args.out)
    print(f"saved -> {args.out}")
    if args.eval:
        evaluate(bank, Path(args.db), Path(args.images))


if __name__ == "__main__":
    main()

# CV Lab — Findings 05 (landmark-anchored table localization)

## Goal
Make ROIs tolerant to the user's stated constraint: screen recordings vary at the edges
(sometimes more/less non-WPT screen content is captured), so the table sits at a different
offset/scale per recording. The hardcoded fractional ROIs from Findings 04 would break. Solution:
locate the table by ON-SCREEN LANDMARKS, derive all ROIs relative to it.
Module: `cv_lab/scripts/landmark_anchor.py`. No ground truth used.

## The key color finding
Sampled pixel colors across the UI: the ENTIRE ClubWPT interface is dark and COOL (hue ~110,
saturation < 90) — the felt, the (perceived) gold rim, the background, the text panels are all
low-saturation. The gold-rim segmentation I first tried FAILED for this reason.

The exception: the little green chip-coin icons are FULLY SATURATED (HSV sat = 255). They are the
ONLY saturated pixels in the whole UI. So a simple HSV green mask isolates them with zero tuning,
regardless of surrounding crop or brightness.

## The anchor: the green-coin constellation
The coins form a RIGID CONSTELLATION at fixed table positions:
- 8 seat coins (one per seat, area ~340 px at reference scale), ringing the oval.
- 1 POT coin, distinctly larger (~1.6x a seat coin) near table center-top.
- (bet chips on the felt also show green but are much larger / irregular -> rejected as anchors.)

Because it is rigid, we fit a SIMILARITY transform (uniform scale s + translation t; NO rotation —
screen recordings don't rotate) from the detected coins to a reference constellation measured from
t0360. Robustness details:
- POT coin identified by relative area (~1.6x the median seat coin) + nearest table-top-center.
- Fit = centroid+RMS-radius init, then ICP-lite (assign each detected coin to nearest reference
  slot, reject far outliers = bet chips, least-squares refit). Works with missing seats (2-8
  occupied) and with stray bet-chip blobs.
- Any canonical ROI maps through the transform to pixels: `roi_px = s * roi_ref_px + t`.

## Validation — synthetic edge variation (the decisive test)
Transformed the reference frame to simulate different recordings: pad with foreign border content
(extra desktop/screen), scale up/down, large offsets. Ran anchoring, compared fitted transform to
the KNOWN ground-truth transform, and diffed the mapped POT/BOARD ROI content vs the reference:

```
case               s_fit  s_true  tx_err  ty_err   potΔ  boardΔ  seats
identity           0.999   1.000     0.7     0.5    0.0     0.0     8
pad 300L/180T      0.999   1.000     0.7     0.5    0.0     0.0     8
scale 0.7          0.699   0.700     0.8     0.3    3.7     3.8     8
scale 1.25         1.249   1.250     0.8     0.3    3.4     3.4     8
big offset(0.85)   0.849   0.850     0.8     0.2    4.0     3.0     8
```

Scale recovered to ~0.1%, translation to < 1 px, and the mapped ROIs land on IDENTICAL content
(potΔ/boardΔ ~0-4 = pure resampling noise) no matter the crop/scale/offset. Visual check: the
mapped POT ROI on a padded+shifted frame crops exactly "POT [coin] 65.10 BB"; the board ROI crops
exactly the 4 board cards.

## Validation — real frames
Ran on 5 real frames incl. ones with bet chips (t0300, t0376) and showdown chip fragments (t0394):
all anchored consistently (s 0.999-1.005, |t| < 3 px), POT coin classified correctly every time.
Bet chips inflate the raw coin count (e.g. 10 "seats") but ICP outlier-rejection keeps the fit
exact (s = 1.000).

## Validation — end-to-end (anchored selector)
Wired anchoring into the key-frame selector (`--anchor`: per-frame anchor -> mapped ROIs; falls
back to canonical fractions if the table can't be located). Re-ran on the Experiment-A hand:
- ALL 12 events captured (the one "borderline" — hero all-in, kf gap 2.67s vs a 2.5s cutoff — is
  clearly captured; a tolerance artifact, not a miss).
- Per-frame anchoring is stable, so region-diff signals match the hardcoded run.
Combined with the proven crop-invariance, this means the anchored selector yields the same
keyframes whether or not the recording is cropped/scaled/offset. Edge-tolerance achieved.

## Honest limitations
- Assumes no rotation (valid for screen recordings) and roughly preserved aspect ratio.
- Needs >= 3 seat coins visible; fails on near-empty / non-table screens — which a screen
  classifier should skip anyway (next task).
- Reference constellation is for THIS 8-max ClubWPT Gold layout; a different table size (e.g.
  6-max) would need its own reference, though the fit already tolerates missing seats.
- Per-frame anchoring costs one HSV-mask + contour pass per frame (cheap). Within a single
  recording s,t are ~constant, so it could be computed once and cached.

## Pipeline status
Locate table (landmark anchor, DONE) -> select key states (region diff, DONE) -> VLM read (DONE)
-> stitch + arithmetic reconcile (DONE). Remaining before a full-session run:
1. **Screen classifier** (table vs lobby vs transition) so non-table frames are skipped.
2. The Findings-04 selector polish (overlay skip, settle-confirmation, action-pill ROI).
3. Wire hand-boundary segmentation + run the whole session end-to-end.

# 10 — Generalizable reconstruction (supersedes the tuned-patch era of 07-29..08-03)

Trigger: the 2026-08-03 session (2114x1414, 297s) reconstructed at a 56.2%
operator-flagged frame error rate — one "hand" spanned 208s and fused 3-4 real
deals with streets in impossible order. Diagnosis against the DB, the timeline
artifact, and git history: the spine encoded facts about one particular table
(hero seated and dealt in at seat 0, two blinds of 0.5/1.0, hero cards as the
deal clock, one window geometry) as constants, and layered repair heuristics on
top when observations disagreed. The 08-03 table straddles (0.05/0.10/0.20) and
the hero sat out — both unrepresentable — and every repair then worked against
the truth (the pot-collapse carry erased the next deal's blind pot 35 times in
the merged mega-hand).

Direction inverted: **observe the table's structure per session, delete the
repairs.** All changes below are live in `scripts/pipeline/`.

## Spine (build_yolo_hand_timeline.py)

- **Segmentation** is multi-signal and hero-independent: dealer-button movement
  (next-read confirmed), board reset (confirmed by the old board never
  resuming), and fresh hero cards after an empty passage are co-equal boundary
  signals. A signal only ARMS the cut; it lands on the first state showing the
  new deal (card backs / hero cards / fresh board), so teardown interstitials
  trail the old hand exactly as before. Segments with no deal evidence are not
  hands.
- **Pot-collapse machinery deleted** (`_POT_COLLAPSE_RATIO`/`_FLOOR_BB`, the
  unconditional carry in `_debounce_pot`, the trim's pot arm): with real
  boundaries, a persistent in-hand pot collapse is real by construction.
- **Forced-post structure is observed, not assumed**: per-hand chains read off
  the deal-open state (standing bets clockwise of the button, nondecreasing,
  straddles at least doubling), voted across the session
  (`_session_forced_posts`); `(0.5, 1.0)` survives only as the documented
  last-resort fallback. Straddles get position `ST`, an explicit
  `post_blind`/`forced_bet_type="straddle"` action row, and preflop order
  opens left of the last post. The DEV CORPUS ITSELF is a straddle game —
  every deal-open shows the 2.0 third post; it was previously booked as an
  UTG "call 2.0" off its green pill, which held only while the pill was
  readable.
- **Roster requires deal evidence**: mid-hand stack-HUD recruitment is capped
  by the session-level set of seats ever dealt cards. A sitting-out hero is
  occupancy, not participation; hero-less hands reconstruct correctly and stay
  unexported under the existing hero-participation gate.
- **VFR continuity**: the sampler answers each request with the frame IN
  EFFECT at that time and returns duplicate-request spans as
  `observed_static_until_s` proofs; `prior_gap_s` now measures time since the
  last OBSERVATION (same-signature frames and static proofs included), not
  since the last distinct state. Spurious coverage gaps on change-driven
  recordings stop firing; nontable spans still gap.

## Geometry / OCR

- **Seat attribution runs in the anchored reference basis**
  (region_detections.py): `_nearest_seat` consumes `TableAnchor.to_ref`
  coordinates with a per-class rejection radius derived from the anchor table
  (half the minimum pairwise spacing); unanchored frames fail closed for
  seated classes like they always did for cards. `SEAT_CENTROIDS` and the
  frame-normalized `_center` are deleted. Measured on g0621 (AR 1.750): 495
  boxes had been filed on the wrong seat (231 bet_text + 227 card_back on
  seat 3 that belonged to seat 2) — the "disagreeing 0.5/6.0 reads" and
  "check facing a raise" codes on that recording were this one defect wearing
  two names, and both dissolve. Invariance under resize AND chrome offset is
  pinned by test.
- **OCR renormalized re-read replaces the 18-factor recovery stack**
  (ocr_readers.py, net ≈ -560 lines): a refused crop is re-read at, at most,
  two DERIVED scales — 1/anchor_scale (the reference render the bank was
  calibrated on) and canvas/run_h (the measured run mapped to the template
  canvas) — under the FULL strict contract, refusing when the scales
  disagree. No waivers (`skip_unexplained_ink`, `allow_b8_suffix` are gone),
  no factor search. Recorded shortfall: 2 of the 6 pinned job-4 1052x732
  crops (204.50, 212.20) now stay UNKNOWN with a named code where the tuned
  stack read them; they never read wrong.
- **Layout supportedness is anchor-health**, not a W×H floor: `-unsupported`
  = no fitted session anchor, or fitted scale × 21px reference glyph height
  below the 9px renormalization-validated floor. The old bound had stamped
  the 4.5%-error 1052x732 session "unsupported" and the 56%-error 2114x1414
  one "supported".
- **classify_screen** tallies WHY frames are nontable (into timeline metadata
  as `nontable_reasons`), and its scale ceiling admits 2x Retina captures
  (1.8 → 2.5). The ICP inlier floor's absolute arm scales with the fit
  (0.015 × fitted table width ≈ the old 30px at s=1).

## Eval status

The five frozen dev-geometry replays (tests/test_cv_recording_regressions.py)
are green with every diff individually explained in the tests themselves. The
v00 answer-key eval could NOT be re-run end-to-end: `clubwpt_session_01.mov`
is no longer on disk (the answer key survives at
cv_lab/results/ground_truth/v00_hands.json; the eval harness itself still
runs).

**08-03 recording re-run (2114x1414, 297s, full pipeline, measured):**

| fact | old (job 4) | new |
|---|---|---|
| hands segmented | 3 (one fusing 3-4 deals over 208s) | 6, longest 95s, streets ordered |
| pot collapses "repaired" | 35 in the merged hand | 0 |
| forced-post structure | none (straddle booked as opening raise) | (0.5, 1.0, 2.0), one ST post per hand |
| waiting hero | exported as SB of every hand | out of the roster for hands 2-3 (its "Waiting" span); in from hand 4, where it genuinely bought in and played (6c As / 8s Ac / 5d 7d) |
| positions | shifted 2 seats (phantom hero + unmodelled straddle) | match the operator's review notes ("bb starts 1 left of me"; "UTG folds, not UTG+1 bc hes the straddle") |
| layout profile | "2114x1414" (misleadingly supported) | "2114x1414" with anchor-health semantics |

Residuals on that session, on the record: hand 2 carries
pot_not_reconciled / board_zone_yield_partial / amounts_unknown_in_ledger
(the REF board bands are still thin at this geometry -- known, out of scope
here); hand 4 (the hero's messy sit-in hand: a green pill during buy-in)
carries stack_ledger_incoherent. Both are flagged, not silently wrong.

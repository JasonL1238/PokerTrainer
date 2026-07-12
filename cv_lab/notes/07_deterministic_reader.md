# CV Lab — Findings 07 (deterministic READ: pot + cards)

Per the Findings-06 decision (VLM is validation-only), the production READ step must be classic
CV. This note covers the first two extractors, both validated against the VLM answer key.
No vision model runs at read time. Scripts: `read_pot.py`, `read_cards.py`. Template banks
persisted under `cv_lab/models/`.

## POT number reader (`read_pot.py`) — the arithmetic-reconciler linchpin
Method: anchored pot ROI -> mask green coin -> binarize bright text -> connected components
classified by HEIGHT band (digits h~21 | 'BB' letters h~16 ignored | decimal point h~4 | panel
edge lines ignored) -> recognize each digit by nearest-template (normalized SSD) -> insert '.' by
x-position -> parse float.

Digit templates built once from frames with known pot value (answer key used only to LABEL glyphs;
runtime is pure template match).

Validation (12 frames, known pots 5.50 -> 347.50):
- **Full bank: 12/12 exact.**
- Leave-one-out: 11/12. The single miss (347.50 -> 342.50) is because 347.50 is the ONLY frame
  containing a '7', so LOO removed the sole '7' template -> a data-coverage artifact, NOT a method
  failure. A full session yields many exemplars per digit.
- Digit coverage from pots alone lacks '9' (no pot value had a 9). '9' comes free from the
  stack/bet reader (same font) — TODO with that extractor.

## CARD reader (`read_cards.py`) — also the deterministic spade/club fix
Method per anchored card slot: present? (brightness) -> suit COLOR red/black (HSV red mask in the
suit sub-region) -> suit SHAPE (template-match binarized pip against the two same-color candidates)
-> rank (template-match the rank glyph). Color splits {h,d} from {s,c}; shape then separates within
color. This is the deterministic resolution of the spade-vs-club ambiguity the VLM kept flagging.

Board slots are axis-aligned + evenly spaced (5 canonical ROIs). Hero cards are fanned/tilted and
are deferred to a follow-up.

Validation (cross-frame: build templates from one frame, read boards in OTHER frames):
- Hand-1 board 8d Tc 7c 9h Kd across 6 frames: **24/24 cards exact** (incl. the black clubs Tc/7c
  the VLM was unsure about).
- **Spade-vs-club, explicit:** built suit templates incl. spade (from a K♠) and club, then
  cross-frame read a different board 6h 2c K♠ (+ turn 10♠): 2♣ -> 'c', K♠ -> 's', 10♠ -> 's', all
  correct. **7/7 on the s/c-focused test.** The exact VLM failure mode is deterministically solved.

## Why this beats the VLM here
- Deterministic, reproducible, no per-frame model call.
- Strictly MORE accurate on the one thing the VLM got wrong/uncertain (black-suit disambiguation),
  because shape templates on the fixed card art are unambiguous where a compressed thumbnail is not.
- Self-labeling bootstrap: templates were built using the VLM answer key ONCE, offline; the shipped
  reader carries only the template arrays.

## Remaining deterministic extractors (next)
1. **Stacks / bets** OCR — reuse the pot digit bank (same font); also completes the '9' template.
2. **Action pills** — classify by pill color (gray check/fold, green call/bet, orange raise, white
   BB) + short text.
3. **Active seat** — blue timer-ring detection. **Dealer** — white 'D' disc.
4. **Hero cards** — fanned/tilted; needs per-card rotation/deskew before the same rank/suit match.
5. Then REPLACE the VLM read in the pipeline with these extractors and re-run the Experiment-A hand
   fully deterministically, scoring the stitched history against the VLM answer key.

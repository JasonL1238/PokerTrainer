# CV Lab — Findings 01 (Layout map)

All experimentation lives under `cv_lab/` and is kept fully separate from `poker_tracker/`.

## Source video
- `data/videos/clubwpt_session_01.mov` (moved from project root; original name had a ` `
  narrow-no-break-space that broke path matching).
- **h264, 2138×1402, ~60 fps, 564.9 s (33,945 frames)**.
- OpenCV's bundled FFmpeg **cannot open** this .mov → use **PyAV** for decode. Confirmed working.
- Non-standard resolution confirms it is a screen-recording crop. In THIS video the WPT window
  fills the whole frame (its own dark bg = the recording bg), so no foreign content on edges —
  but the user says edges will vary, so we must NOT rely on absolute pixel coords.

## Screen types seen
- **Table view** (frames 00,01,02,04,05,07,09) — the poker table.
- **Lobby view** (frame 11) — game list / menu. Must be detected and skipped.
- Need a cheap **screen classifier** (table vs lobby vs other) before any extraction.

## Table layout (8-max) — everything the UI hands us for free
The client renders almost every decision-relevant stat as clean text / colored pills. This makes
the problem far more "OCR + template + layout" than "infer from pixel deltas".

Stable landmarks (good anchors for normalization):
- `POT: <n> BB` — center, just above the board.
- `BLINDS 0.05/0.10 (0.05)` — top-right.
- `HANDS 1.16.2.2606300254` — bottom-left (table/hand id string).
- Faded `ClubWPT GOLD` logo — dead center.
- Hamburger menu — top-left.

Per-seat (8 fixed seat anchors around the oval):
- Player **name** + **stack in BB** (e.g. `99 BB`, `188 BB`) — already denominated in BB, no
  chip→BB conversion needed.
- Avatar circle.
- **Action pill** under seat, color-coded: `CHECK`/`FOLD` (gray), `CALL`/`BET` (green),
  `RAISE` (orange), `BB` (white, = posted big blind). ← per-player last action, given directly.
- **Bet chips** on the felt near each seat with `<n> BB`.
- **Active-player timer**: blue ring + countdown number around whoever is to act.
- **Dealer button**: white `D` disc.
- Seats can also show `Waiting` / new player joining (frame 09: `GAU` 200 BB replaced 7HuhHuh2).

Hero (bottom center, name "Mochi RiceBalls"):
- Hole cards shown **face up** (e.g. 5♦6♠, K♠K♥, Q♠J♥).
- Made-hand hint label above cards: `PAIR`, `STRAIGHT`, etc.
- When it's hero's turn, action buttons appear bottom-right (`Check`, `Check/Fold`, etc.).
- Hero name matches the logged-in account name shown in the lobby top-right.

Board cards render at fixed center slots (flop x3 / turn / river), clean card graphics.

## Implication for approach
Because the UI is clean, high-contrast, consistent, and BB-denominated, TWO approaches are viable:
- **A. Classic CV**: landmark-anchor → normalized ROI crops → template-match cards + OCR numbers/pills.
- **B. VLM extraction**: crop the table region → ask a vision model for structured JSON of all stats.

Given the text density and card graphics, the likely winner is a **hybrid**: VLM for the whole-frame
structured read (robust, fast to build, edge-tolerant), with template-matching as a cheap/deterministic
cross-check on cards. Action reconstruction is much easier here than the generic case because the
per-player pills label the action explicitly.

## Open questions / edge cases to watch
- Hand segmentation: is `HANDS <id>` per-hand or per-table-session? (Looked constant across frames —
  verify at hi-res.) If not per-hand, segment hands by hero-cards-appear / board-reset / pot-reset.
- Animations (chips sliding, card deal/flip, pot pushes) → sample stable frames only.
- Mid-action frames where pills/timer are transitioning.

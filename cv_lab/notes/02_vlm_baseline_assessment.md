# CV Lab — Findings 02 (VLM baseline assessment)

## Experiment
4 hand-picked, stable table frames (00, 04, 07, 09). One subagent per frame read the **full-res**
PNG and returned strict JSON of every decision-relevant stat. No ground truth, no hand-tuned CV.
Raw outputs: `cv_lab/results/vlm_baseline_batch01.json`.

## Verdict: strong. What the VLM got right (cross-checked vs my own view of each frame)
- `screen_type`, `blinds` (0.05/0.10 ante 0.05) — 4/4
- `pot_bb` — 4/4 exact (13.50, 15, 65.10, 63.20)
- `board_cards` ranks — 4/4 correct
- Hero hole-card **ranks** + stack + made-hand label (PAIR/STRAIGHT) + action buttons — 4/4
- All 8 seat names + stacks (to the decimal) — effectively 4/4
- Action pills (RAISE/CALL/CHECK/FOLD/BET/BB) — correct where present
- Dealer button seat, active/to-act seat — mostly correct
- **Edge case handled**: frame 09 correctly flagged seat "GAU" as `Waiting` (new/sitting-out player)

This confirms the "just read the screen" approach is viable and high-accuracy on clean frames.

## Confirmed weaknesses (specific, actionable)
1. **Black-suit disambiguation ♠ vs ♣** — frame 07 hero card read as Qc; frame 09 board 9/6 suits
   flagged uncertain. Both suits are black; at compression/low-res the shapes blur. THIS is the #1
   card-reading risk. Fix: deterministic template-match / suit-shape classifier on the fixed card
   slots as a cross-check. (Red vs black is trivial; the ambiguity is only within black.)
2. **Bet-chip → seat assignment** — chips sit on the felt, not rigidly under a seat; agents inferred
   by position and flagged it. Fix: once the table is geometrically normalized, each seat has a fixed
   "bet zone"; assign by zone, not by guessing.
3. **`is_active` occasionally on 2 seats** (frame 09 marked hero + Donkey13). The blue timer ring is
   the single source of truth; detect it explicitly (ring color/shape) rather than trusting prose.
4. **`HANDS 1.16.2.2606300254` is NOT a per-hand id** — verified identical across frames 00 and 04,
   which are different hands. It's a table/session string. => segment hands by hero-cards-change /
   board-reset / pot-reset, NOT by this value.
5. **Name tokenization** ("Ash Lord5699" vs "AshLord5699") — wraps to 2 lines; cosmetic, normalize.

## Honest scope caveats (what this baseline did NOT test)
- These were **stable, clear, hand-picked frames**. The real pipeline must also:
  - **classify/skip** lobby + transition + animation frames,
  - **select** which frames to read (can't run a VLM on all 33,945 frames — sample at state changes /
    decision points; target tens–low-hundreds of key frames per session),
  - **segment** frames into hands, and
  - **stitch** per-frame snapshots into a street-by-street action timeline.
- Per-frame accuracy is necessary but not sufficient; timeline reconstruction is the next real test.

## Recommended pipeline (hybrid, VLM-first)
1. Frame **decode** via PyAV (OpenCV can't open these .mov files).
2. Cheap **screen classifier** (table vs lobby vs transition) — histogram/landmark, no VLM.
3. **Stable-frame + state-change selection** to pick key frames (dedupe near-identical frames).
4. **VLM structured read** per selected frame (this baseline) — the workhorse.
5. **Deterministic cross-checks**: template-match card slots (fix ♠/♣), fixed seat bet-zones,
   timer-ring detection for active seat.
6. **Hand segmentation** (hero-card/board/pot resets) + **action-timeline** stitching.
7. Emit structured hand histories → feed the existing poker_tracker schema later (kept separate for now).

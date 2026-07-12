"""End-to-end deterministic session pipeline (NO VLM at runtime).

decode (PyAV) -> screen classify -> landmark anchor -> deterministic READ
(pot/board/hero/stacks/bets/pills/active/dealer) -> hand segmentation ->
street-by-street stitch + arithmetic reconciliation -> reconstructed hands.

Outputs a timeline + a list of reconstructed hands and a completeness score.
Sampling is sequential-decode every `--stride` frames (default 30 -> ~2 fps).
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import av

import read_table

VIDEO = "/Users/jasonli/Documents/GitHub/PokerTrainer/data/videos/clubwpt_session_01.mov"
MODELS = "/Users/jasonli/Documents/GitHub/PokerTrainer/cv_lab/models"


# ---------- pass 1: decode + read timeline ----------
def scan(video, stride, t_start, t_end):
    read_table.load_models(MODELS)
    container = av.open(video)
    stream = container.streams.video[0]
    tb = stream.time_base
    timeline = []
    i = -1
    for frame in container.decode(stream):
        i += 1
        if i % stride:
            continue
        t = float(frame.pts * tb)
        if t < t_start:
            continue
        if t_end and t > t_end:
            break
        img = frame.to_ndarray(format="bgr24")
        snap = read_table.read_table(img)
        snap["t"] = round(t, 2)
        timeline.append(snap)
        if len(timeline) % 50 == 0:
            print(f"t={t:6.1f}s  frames={len(timeline)}", flush=True)
    container.close()
    return timeline


# ---------- pass 2: hand segmentation ----------
def _mode_hero(snaps):
    """Consensus hero cards over a set of snapshots (as a sorted tuple)."""
    from collections import Counter
    c = Counter()
    for s in snaps:
        h = s.get("hero") or []
        if len(h) == 2:
            c[tuple(sorted(h))] += 1
    return c.most_common(1)[0][0] if c else None


def _denoise_pot(tbl):
    """Drop isolated pot reads (a real pot persists >=2 samples at 2fps); single-
    frame values are transition/award-animation artifacts. Returns a parallel
    list of cleaned pots (None where unreliable)."""
    pots = [s.get("pot") for s in tbl]
    clean = list(pots)
    for i, p in enumerate(pots):
        if p is None:
            continue
        left = i > 0 and pots[i - 1] == p
        right = i < len(pots) - 1 and pots[i + 1] == p
        if not (left or right):
            clean[i] = None
    return clean


def _smooth_bc(tbl):
    """Median-of-3 board counts (kills single-frame board misreads)."""
    bc = [len(s.get("board") or []) for s in tbl]
    sm = list(bc)
    for i in range(1, len(bc) - 1):
        sm[i] = sorted(bc[i - 1:i + 2])[1]
    return sm


def segment(timeline):
    """Split the table timeline into hands at DEAL events. Primary signal: the
    board resetting to 0 after a flop was seen (a clean falling edge). Fallback
    for fold-around (preflop-only) hands: the pot dropping back to blinds while
    the board is empty. Robust to isolated pot/board misreads."""
    tbl = [s for s in timeline if s.get("screen") == "table"]
    clean = _denoise_pot(tbl)
    bc = _smooth_bc(tbl)
    for s, p in zip(tbl, clean):
        s["pot_clean"] = p
    hands, cur = [], []
    reached_flop, hand_max, prev_bc = False, 0.0, 0
    for i, s in enumerate(tbl):
        p = clean[i]
        boundary = False
        if cur:
            board_reset = reached_flop and bc[i] == 0 and prev_bc > 0
            pot_reset = (bc[i] == 0 and p is not None and hand_max > 5
                         and p < 0.5 * hand_max)
            boundary = board_reset or pot_reset
        if boundary:
            hands.append(cur)
            cur, reached_flop, hand_max = [], False, 0.0
        cur.append(s)
        if bc[i] >= 3:
            reached_flop = True
        if p is not None:
            hand_max = max(hand_max, p)
        prev_bc = bc[i]
    if cur:
        hands.append(cur)
    return hands


# ---------- pass 3: reconstruct + reconcile one hand ----------
def _mode(vals):
    from collections import Counter
    vals = [v for v in vals if v is not None]
    return Counter(vals).most_common(1)[0][0] if vals else None


def reconstruct(hand):
    """Build street-by-street state and reconcile pot vs stack deltas, using the
    denoised pot stream and consensus over frames (robust to animation spikes)."""
    from collections import Counter
    hero = _mode_hero(hand)
    dseat = Counter(s.get("dealer_seat") for s in hand if s.get("dealer_seat") is not None)
    dealer = dseat.most_common(1)[0][0] if dseat else None

    # consensus board per street (largest board count seen -> final board)
    board_by_bc = {}
    for s in hand:
        b = s.get("board") or []
        board_by_bc.setdefault(len(b), []).append(tuple(b))
    max_bc = max(board_by_bc) if board_by_bc else 0
    board_final = list(_mode(board_by_bc[max_bc])) if max_bc else []

    # settled final pot = mode of cleaned pots in the last ~20% of the hand
    clean = [s.get("pot_clean") for s in hand]
    tail = [p for p in clean[int(0.8 * len(clean)):] if p is not None]
    final_pot = _mode(tail) if tail else _mode([p for p in clean if p is not None])

    # pot sequence: strictly-increasing plateaus. The pot only grows within a
    # hand (until the showdown award, capped out below), so a value below the
    # running max is a misread/animation dip and is dropped.
    seq, run_max = [], 0.0
    for p in clean:
        if p is None or p < 1.0:                     # sub-blind noise / leading 0
            continue
        if final_pot is not None and p > final_pot * 1.05:   # award over-count
            continue
        if p > run_max + 1e-6:
            seq.append(p)
            run_max = p

    # per-street end pot: last stable cleaned pot while at that board count
    streets = {0: "preflop", 3: "flop", 4: "turn", 5: "river"}
    street_pot = {}
    for s in hand:
        bc = len(s.get("board") or [])
        p = s.get("pot_clean")
        if bc in streets and p is not None and (final_pot is None or p <= final_pot * 1.05):
            street_pot[streets[bc]] = p     # last stable wins (end of street)
    pots = [{"street": st, "pot": street_pot[st]}
            for st in ["preflop", "flop", "turn", "river"] if st in street_pot]

    # stable per-seat stacks (drop isolated single-frame reads) for reconciliation
    stable = {}
    for i in range(8):
        vals = [seat["stack"] for s in hand for seat in (s.get("seats") or [])
                if seat["seat"] == i and seat["stack"] is not None]
        keep = [v for j, v in enumerate(vals)
                if (j > 0 and vals[j - 1] == v) or (j < len(vals) - 1 and vals[j + 1] == v)]
        if keep:
            stable[i] = keep
    # chips each seat put in = start(first stable) - min(stable)
    contributed = sum(max(v[0] - min(v), 0) for v in stable.values())
    # winner = seat that recovered the most (min -> end), i.e. collected the pot
    winner, win_gain = None, 0.0
    for i, v in stable.items():
        gain = v[-1] - min(v)
        if gain > win_gain:
            win_gain, winner = gain, i

    # soft self-check: total contributions ~ final pot (stacks are noisier than
    # pot/cards -- all-ins, sit-outs -- so this raises confidence, not a gate).
    reconciled = (final_pot is not None and contributed > 0
                  and abs(contributed - final_pot) <= max(3.0, 0.25 * final_pot))
    # card consistency self-check: no card can appear twice (hero + board).
    all_cards = (list(hero) if hero else []) + list(board_final)
    cards_ok = len(all_cards) == len(set(all_cards))
    # a hand is COMPLETE when the pot/card evidence is self-consistent:
    saw_flop = len(board_final) >= 3
    complete = (bool(hero) and cards_ok and final_pot is not None
                and len(seq) >= 2 and winner is not None
                and (saw_flop or len(seq) >= 3))
    return {
        "t_start": hand[0]["t"], "t_end": hand[-1]["t"], "n_snaps": len(hand),
        "hero": list(hero) if hero else None, "dealer_seat": dealer,
        "board": board_final, "streets": pots, "pot_sequence": seq,
        "final_pot": final_pot, "contributed_est": round(contributed, 2),
        "reconciled": reconciled, "winner_seat": winner,
        "win_gain": round(win_gain, 2), "complete": complete,
    }


def min_stk_during(hand, seat_i):
    vals = [seat["stack"] for s in hand for seat in (s.get("seats") or [])
            if seat["seat"] == seat_i and seat["stack"] is not None]
    return min(vals) if vals else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=30)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=0.0)
    ap.add_argument("--out", default="cv_lab/results/session_run.json")
    ap.add_argument("--video", default=VIDEO)
    ap.add_argument("--from-timeline", default="",
                    help="reprocess a saved *_timeline.json instead of decoding")
    args = ap.parse_args()

    if args.from_timeline:
        timeline = json.load(open(args.from_timeline))
    else:
        timeline = scan(args.video, args.stride, args.start, args.end)
    hands = segment(timeline)
    recon = [reconstruct(h) for h in hands]
    n_table = sum(1 for s in timeline if s.get("screen") == "table")
    n_complete = sum(1 for r in recon if r["complete"])
    summary = {
        "video": args.video, "stride": args.stride,
        "n_samples": len(timeline), "n_table": n_table,
        "n_hands": len(hands), "n_complete_hands": n_complete,
        "hands": recon,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(summary, open(args.out, "w"), indent=2)
    # also dump the raw timeline for debugging
    json.dump(timeline, open(args.out.replace(".json", "_timeline.json"), "w"), indent=0)
    print(f"\nsamples={len(timeline)} table={n_table} hands={len(hands)} "
          f"complete={n_complete}")
    for r in recon:
        print(f"  [{r['t_start']:.0f}-{r['t_end']:.0f}s] hero={r['hero']} "
              f"board={r['board']} seq={r['pot_sequence']} "
              f"final={r['final_pot']} recon={r['reconciled']} "
              f"win=seat{r['winner_seat']}(+{r['win_gain']}) complete={r['complete']}")


if __name__ == "__main__":
    main()

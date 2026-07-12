"""Offline diagnostics for a saved deterministic CV timeline.

This is post-session analysis only. It reads a saved run_session.py timeline and
prints per-hand extractor stability metrics plus reconstruction failure reasons.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from statistics import mean

sys.path.insert(0, os.path.dirname(__file__))
import run_session


def _card_counts(snaps, key):
    return Counter(tuple(s.get(key) or []) for s in snaps)


def _hero_candidates(snaps):
    counts = _card_counts(snaps, "hero")
    return counts.most_common(5)


def _board_candidates(snaps):
    by_count = defaultdict(Counter)
    for s in snaps:
        board = tuple(s.get("board") or [])
        by_count[len(board)][board] += 1
    return {count: counter.most_common(4) for count, counter in sorted(by_count.items())}


def _duplicate_cards(hero, board):
    cards = list(hero or []) + list(board or [])
    counts = Counter(cards)
    return [card for card, count in counts.items() if count > 1]


def _stable_stack_coverage(snaps):
    coverage = {}
    for seat in range(8):
        vals = [
            row["stack"]
            for s in snaps
            for row in (s.get("seats") or [])
            if row["seat"] == seat and row.get("stack") is not None
        ]
        coverage[seat] = {
            "reads": len(vals),
            "unique": len(set(vals)),
            "first": vals[0] if vals else None,
            "last": vals[-1] if vals else None,
            "min": min(vals) if vals else None,
            "max": max(vals) if vals else None,
        }
    return coverage


def _failure_reasons(recon):
    reasons = []
    hero = recon.get("hero")
    board = recon.get("board") or []
    seq = recon.get("pot_sequence") or []
    if not hero:
        reasons.append("no_hero_consensus")
    dups = _duplicate_cards(hero, board)
    if dups:
        reasons.append(f"duplicate_cards={','.join(dups)}")
    if recon.get("final_pot") is None:
        reasons.append("no_final_pot")
    if len(seq) < 2:
        reasons.append("short_pot_sequence")
    if recon.get("winner_seat") is None:
        reasons.append("no_winner")
    if len(board) < 3 and len(seq) < 3:
        reasons.append("preflop_short_sequence")
    return reasons or ["complete"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("timeline")
    args = ap.parse_args()

    timeline = json.load(open(args.timeline))
    hands = run_session.segment(timeline)
    recons = [run_session.reconstruct(hand) for hand in hands]
    table = [s for s in timeline if s.get("screen") == "table"]
    resid = [s.get("resid") for s in table if s.get("resid") is not None]
    print(f"samples={len(timeline)} table={len(table)} hands={len(hands)} complete={sum(r['complete'] for r in recons)}")
    if resid:
        print(f"anchor_resid mean={mean(resid):.5f} max={max(resid):.5f}")
    print()

    for idx, (hand, recon) in enumerate(zip(hands, recons), start=1):
        print(
            f"hand {idx:02d} {recon['t_start']:.2f}-{recon['t_end']:.2f}s "
            f"n={recon['n_snaps']} complete={recon['complete']} reasons={';'.join(_failure_reasons(recon))}"
        )
        print(f"  hero={recon['hero']} top_hero={_hero_candidates(hand)}")
        print(f"  board={recon['board']} board_candidates={_board_candidates(hand)}")
        print(
            f"  pot_seq={recon['pot_sequence']} final={recon['final_pot']} "
            f"winner={recon['winner_seat']} win_gain={recon['win_gain']} reconciled={recon['reconciled']}"
        )
        noisy = {
            seat: stats
            for seat, stats in _stable_stack_coverage(hand).items()
            if stats["reads"] and stats["unique"] > max(5, stats["reads"] // 5)
        }
        if noisy:
            print(f"  noisy_stack_seats={noisy}")
        print()


if __name__ == "__main__":
    main()

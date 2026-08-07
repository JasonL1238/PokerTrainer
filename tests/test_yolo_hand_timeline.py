"""Reconstruction-spine tests: synthetic 7-class region detections -> full hand.

Uses the region_detections contract with hand-crafted boxes (no detector/OCR) to
drive build_yolo_hand_timeline end to end and assert a complete reconstructed hand:
positions, ordered actions with bet sizes from stack deltas, pot, winner, and the
arithmetic reconciliation. Also confirms the spine output validates cleanly.
"""
import json
from pathlib import Path

from cv_lab.scripts.eval.validate_yolo_card_timeline import validate_timeline
from cv_lab.scripts.pipeline import region_detections as rd
from cv_lab.scripts.pipeline.build_yolo_hand_timeline import build_hand_timeline

CV_FIXTURES = Path(__file__).parent / "fixtures" / "cv"

# 1400x1000 (AR 1.400) sits on the 1.397 reference basis the card zones are
# anchored in: stack_text boxes placed on SEAT_ANCHORS_BY_CLASS fit with residual
# 0.0003, so raw normalized coords map to reference coords 1:1. A square 1000x1000
# frame fits at residual 0.0537, above ANCHOR_MAX_RESID, and would fail closed.
W, H = 1400, 1000


def _det(cls, nx, ny, attr=None, conf=1.0):
    cx, cy = nx * W, ny * H
    return {"cls": cls, "conf": conf, "xyxy": [cx - 15, cy - 20, cx + 15, cy + 20], "attr": attr}


def _stack_anchors(read):
    """The 8 stack_text landmark boxes the table anchor is fitted from.

    Only the seats in ``read`` carry a value; the rest are landmarks with no
    readable amount (attr None -> read_amount None), so they anchor the frame
    without entering the stack ledger.
    """
    return [_det("stack_text", nx, ny, read.get(seat))
            for seat, (nx, ny) in rd.SEAT_ANCHORS_BY_CLASS["stack_text"].items()]


# Hero seat centroid ~ (0.50, 0.86); villain seat 4 centroid ~ (0.50, 0.14).
# Hero hole cards render at reference ry 0.674-0.687.
def _hero_cards():
    return [_det("face_card", 0.46, 0.68, "As"), _det("face_card", 0.52, 0.68, "Kd")]


def _board(cards):
    xs = [0.40, 0.45, 0.50, 0.55, 0.60]
    return [_det("face_card", xs[i], 0.45, c) for i, c in enumerate(cards)]


FLOP = ["2c", "7d", "9h"]
TURN = FLOP + ["Ts"]
RIVER = TURN + ["Jc"]


def _frame(time_s, *, board, s0, s4, pot=None, pill0=None, pill4=None, active=4):
    dets = _hero_cards() + _board(board)
    # Seated boxes sit on their own class's anchor table: seat attribution is
    # anchored with a rejection radius, so a box parked on another class's
    # position (the old avatar centroid) would be refused, exactly as a
    # misplaced box on a real frame should be.
    dets.append(_det("card_back", *rd.SEAT_ANCHORS_BY_CLASS["card_back"][4]))
    dets.append(_det("dealer_button", *rd.SEAT_ANCHORS_BY_CLASS["dealer_button"][4]))
    # stack_text doubles as the table-anchor constellation; only seats 0 and 4
    # carry a readable amount.
    dets.extend(_stack_anchors({0: s0, 4: s4}))
    if pot is not None:
        dets.append(_det("pot_text", 0.50, 0.32, pot))
    if pill0 is not None:
        dets.append(_det("action_pill", *rd.SEAT_ANCHORS_BY_CLASS["action_pill"][0], pill0))
    if pill4 is not None:
        dets.append(_det("action_pill", *rd.SEAT_ANCHORS_BY_CLASS["action_pill"][4], pill4))
    dets.append(_det("active_turn_indicator",
                     *rd.SEAT_ANCHORS_BY_CLASS["active_turn_indicator"][active]))
    return {"image": f"f{time_s}.jpg", "time_s": time_s, "width": W, "height": H, "detections": dets}


def _hand_fixture():
    return [
        _frame(0.0, board=[], s0=100, s4=100, active=4),
        _frame(1.0, board=[], s0=100, s4=97, pot=3, pill4="raise", active=0),      # villain raise 3
        _frame(2.0, board=[], s0=97, s4=97, pot=6, pill0="call", pill4="raise", active=4),  # hero call 3
        _frame(3.0, board=FLOP, s0=97, s4=97, pot=6, pill0="call", pill4="raise", active=0),  # flop dealt
        _frame(4.0, board=FLOP, s0=90, s4=97, pot=13, pill0="bet", active=4),       # hero bet 7
        _frame(5.0, board=FLOP, s0=90, s4=90, pot=20, pill0="bet", pill4="call", active=0),  # villain call 7
        _frame(6.0, board=TURN, s0=90, s4=90, pot=20, pill0="check", pill4="check", active=0),  # turn checks
        _frame(7.0, board=RIVER, s0=90, s4=90, pot=20, pill0="check", pill4="check", active=0),  # river
        _frame(8.0, board=RIVER, s0=110, s4=90, pot=20, active=0),                  # showdown, hero wins
    ]


def _build():
    frames = rd.frames_from_fixture(_hand_fixture())
    return build_hand_timeline(frames)


def test_spine_reconstructs_cards_and_streets():
    timeline = _build()
    assert timeline["summary"]["hands"] == 1
    assert timeline["summary"]["complete_hands"] == 1
    hand = timeline["hands"][0]
    assert hand["hero"] == ["As", "Kd"]
    assert hand["board"] == ["2c", "7d", "9h", "Ts", "Jc"]
    assert [s["street"] for s in hand["streets"]] == ["preflop", "flop", "turn", "river"]


def test_spine_assigns_positions_from_dealer():
    hand = _build()["hands"][0]
    by_seat = {p["seat"]: p for p in hand["players"]}
    assert len(hand["players"]) == 2
    assert by_seat[4]["position"] == "BTN"   # dealer seat
    assert by_seat[0]["position"] == "SB"    # next in ring
    assert by_seat[0]["is_hero"] is True


def test_mid_hand_start_keeps_pre_capture_folders_in_button_position_ring():
    """The flagged job-2 shape: capture begins facing a BB raise after three
    players have already folded, so their card backs and FOLD pills are gone.

    Stable stack HUDs retain the eight occupied seats.  The dealer button at
    seat 4 must therefore yield seat 5=SB, seat 6=BB, seat 7=UTG and hero
    seat 0=UTG+1.  The already-standing actions are ordered around that same
    button-derived ring; they are never numerically seat-sorted.
    """
    from cv_lab.scripts.pipeline.build_yolo_hand_timeline import reconstruct

    states, _events = _states_from(_hand_fixture())
    for state in states:
        state["dealer_seat"] = 4
        state["stacks"].update({
            1: 194.6,
            2: 140.7,
            3: 181.1,
            5: 191.3,
            6: 183.0,
            7: 321.2,
        })
    states[0]["dealt_in"] = [0, 2, 5, 6, 7]
    states[0]["pills"] = {
        0: "call",
        2: "call",
        5: "call",
        6: "raise",
        7: "call",
    }
    states[0]["bets"] = {0: 2.0, 2: 2.0, 5: 2.0, 6: 10.0, 7: 2.0}

    hand = reconstruct(states, 1)
    by_seat = {player["seat"]: player for player in hand["players"]}
    assert set(by_seat) == set(range(8))
    assert {
        seat: by_seat[seat]["position"] for seat in range(8)
    } == {
        4: "BTN",
        5: "SB",
        6: "BB",
        7: "UTG",
        0: "UTG+1",
        1: "LJ",
        2: "HJ",
        3: "CO",
    }

    opening = [
        action for action in hand["actions"]
        if action["source_state_index"] == states[0]["state_index"]
        and action["street"] == "preflop"
    ]
    assert [action["seat"] for action in opening[:5]] == [7, 0, 2, 5, 6]
    assert [action["amount"] for action in opening[:5]] == [2.0, 2.0, 2.0, 2.0, 10.0]
    assert not {
        action["seat"] for action in hand["actions"]
    } & {1, 3, 4}, "pre-capture folders stay positional but receive no invented actions"


def test_a_sitting_out_seat_is_never_recruited_from_its_stack_hud():
    """A stack HUD proves a seat is OCCUPIED, not that it is in the hand. On
    the 2026-08-03 session the hero sat out ('Waiting' badge, no cards all
    session) but showed a stable stack, so the mid-hand-open recruitment made
    them the small blind of every hand and shifted every position by one. The
    session-level dealt-cards set is the cap: a seat never dealt cards anywhere
    in the session cannot be in any hand's roster, while pre-capture folders
    (dealt in neighbouring hands) still recruit exactly as before."""
    from cv_lab.scripts.pipeline.build_yolo_hand_timeline import (
        _player_seats,
        reconstruct,
    )

    states, _events = _states_from(_hand_fixture())
    for state in states:
        state["dealer_seat"] = 4
        state["stacks"].update({1: 194.6, 2: 140.7, 3: 181.1,
                                5: 191.3, 6: 183.0, 7: 321.2})
    states[0]["dealt_in"] = [2, 5, 6, 7]
    states[0]["pills"] = {2: "call", 5: "call", 6: "raise", 7: "call"}
    states[0]["bets"] = {2: 2.0, 5: 2.0, 6: 10.0, 7: 2.0}
    for state in states:
        # Sitting out: hero zone empty all hand, no card back, no pill -- the
        # stack HUD is the only trace of seat 0.
        state["hero_cards"] = []
        state["dealt_in"] = [s for s in state["dealt_in"] if s != 0]
        state["pills"] = {s: p for s, p in state["pills"].items() if s != 0}
        state["bets"].pop(0, None)

    # Without the session cap the hero's stable stack recruits seat 0.
    assert 0 in _player_seats(states, 4)
    # With it, only seats dealt cards somewhere in the session qualify.
    dealt = {1, 2, 3, 4, 5, 6, 7}
    assert 0 not in _player_seats(states, 4, session_dealt_seats=dealt)

    hand = reconstruct(states, 1, session_dealt_seats=dealt)
    seats = {player["seat"] for player in hand["players"]}
    assert 0 not in seats
    assert not any(player["is_hero"] for player in hand["players"])
    # Positions walk the ring over the seven real players; nothing is shifted
    # by a phantom hero between the button and the small blind.
    by_seat = {p["seat"]: p for p in hand["players"]}
    assert by_seat[4]["position"] == "BTN"
    assert by_seat[5]["position"] == "SB"
    assert by_seat[6]["position"] == "BB"


def test_observed_forced_posts_reads_the_chain_and_refuses_touched_opens():
    """The forced-post structure is OBSERVED off the deal-open state, never
    assumed: standing bets on the chain clockwise of the button, nondecreasing,
    each straddle at least doubling the post before it. Anything showing action
    already happened -- a board, a non-POST pill, money beyond the chain, a
    refused read -- makes the open unobservable (None), and the session vote
    decides from the hands whose opens were seen."""
    from cv_lab.scripts.pipeline.build_yolo_hand_timeline import (
        _observed_forced_posts,
    )

    all_seats = tuple(range(8))

    def open_state(bets, pills=None, board=(), dealt=all_seats, unknown=None):
        return {"board_cards": list(board), "dealt_in": list(dealt),
                "bets": bets, "bets_unknown": unknown or {}, "pills": pills or {}}

    players = list(range(8))
    # Dealer 0: chain reads seat 1 (SB), 2 (BB), 3 (straddle). The green POST
    # pill on the straddler is tolerated -- it is the post, not an action.
    straddled = open_state({1: 0.5, 2: 1.0, 3: 2.0}, pills={3: "bet_or_call"})
    assert _observed_forced_posts([straddled], players, 0) == (0.5, 1.0, 2.0)

    plain = open_state({1: 0.5, 2: 1.0})
    assert _observed_forced_posts([plain], players, 0) == (0.5, 1.0)

    # A pill on a non-posting seat means someone already acted.
    acted = open_state({1: 0.5, 2: 1.0}, pills={5: "fold"})
    assert _observed_forced_posts([acted], players, 0) is None

    # A third bet that does not double the BB is not a straddle; standing
    # money beyond the chain means an action, so the open proves nothing.
    minraise = open_state({1: 0.5, 2: 1.0, 3: 1.5})
    assert _observed_forced_posts([minraise], players, 0) is None

    # A refused read on a chain seat cannot certify the chain.
    refused = open_state({1: 0.5, 2: 1.0}, unknown={3: "no_digit_run"})
    assert _observed_forced_posts([refused], players, 0) is None

    # A board showing = mid-hand capture, not an open.
    midhand = open_state({1: 0.5, 2: 1.0}, board=("Ah", "Kd", "2c"))
    assert _observed_forced_posts([midhand], players, 0) is None


def test_preflop_order_opens_left_of_the_last_forced_post():
    """With a straddle the preflop action opens left of the STRADDLE, not left
    of the big blind -- the operator's own review note on the 2026-08-03
    session ('UTG folds. not UTG+1 bc hes the straddle'). Position names gain
    an ST slot and everything after it shifts one seat."""
    from cv_lab.scripts.pipeline.build_yolo_hand_timeline import (
        _positions,
        _street_seat_order,
    )

    straddled = _positions(list(range(8)), 0, n_straddles=1)
    assert straddled[1] == "SB"
    assert straddled[2] == "BB"
    assert straddled[3] == "ST"
    assert straddled[4] == "UTG"
    order = _street_seat_order(straddled, "preflop")
    assert order[0] == 4          # UTG, left of the straddle
    assert order[-1] == 3         # the straddler acts last preflop

    plain = _positions(list(range(8)), 0)
    assert plain[3] == "UTG"
    assert _street_seat_order(plain, "preflop")[0] == 3

    # The straddle count is capped by the ring: 3-handed there is no seat
    # left to straddle, and positions fall back to the plain names.
    three = _positions([0, 1, 2], 0, n_straddles=1)
    assert set(three.values()) == {"BTN", "SB", "BB"}


def test_spine_derives_action_sizes_from_stack_deltas():
    hand = _build()["hands"][0]
    got = {(a["action_type"], a["amount"]) for a in hand["actions"]}
    assert ("raise", 3.0) in got
    assert ("call", 3.0) in got
    assert ("bet", 7.0) in got
    assert ("call", 7.0) in got
    assert sum(1 for a in hand["actions"] if a["action_type"] == "check") >= 2
    # actions never move to an earlier street
    order = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}
    seq = [order[a["street"]] for a in hand["actions"]]
    assert seq == sorted(seq)
    assert all(action["source_image"] for action in hand["actions"])
    assert all(action["source_time_s"] is not None for action in hand["actions"])
    assert {action["derivation"] for action in hand["actions"]} >= {
        "stack_delta",
        "action_pill",
    }


def test_spine_reconciles_pot_and_winner():
    hand = _build()["hands"][0]
    assert hand["pot"] == 20
    assert hand["winner_seat"] == 0
    assert hand["result"] == "Hero wins"
    assert hand["hero_bb_won"] == 10.0
    assert hand["reconciled"] is True
    assert hand["complete"] is True


def test_spine_output_validates_clean():
    report = validate_timeline(_build())
    assert report["summary"]["total_warnings"] == 0
    assert report["summary"]["confidence_score"] == 1.0


# --------------------------------------------------------------------------- #
# Table anchoring: card zones are derived from a per-frame similarity fit, with
# the session-median fit as the only fallback and NO hardcoded window.
# --------------------------------------------------------------------------- #
def test_states_carry_anchor_health():
    from cv_lab.scripts.pipeline.build_yolo_hand_timeline import build_states

    states, _events = build_states(rd.frames_from_fixture(_hand_fixture()))
    assert states
    assert all(s["anchor_ok"] for s in states)
    assert all(s["anchor_source"] == "frame" for s in states)
    assert all(s["unanchored_cards"] == 0 for s in states)


def test_session_anchor_covers_a_frame_whose_own_fit_fails():
    """A frame that lost all but two stack_text boxes cannot fit on its own. The
    session-median transform carries it, so its cards are still zoned -- but the
    state records that the anchor came from the session, not the frame."""
    from cv_lab.scripts.pipeline.build_yolo_hand_timeline import build_states

    fixture = _hand_fixture()
    starved = fixture[4]
    keep = 0
    kept = []
    for det in starved["detections"]:
        if det["cls"] != "stack_text":
            kept.append(det)
        elif keep < 2:
            kept.append(det)
            keep += 1
    starved["detections"] = kept

    frames = rd.frames_from_fixture(fixture)
    assert rd.anchor_for_frame(frames[4]) is None      # the frame alone cannot fit
    assert rd.assign_regions(frames[4])["board"] == []  # ... and fails closed

    states, _events = build_states(frames)
    covered = [s for s in states if s["anchor_source"] == "session"]
    assert covered, "session anchor never used"
    assert all(len(s["board_cards"]) == 3 for s in covered)


def test_no_session_anchor_means_cards_are_dropped_not_guessed():
    """Fail closed end to end: with no fittable frame anywhere, no card is
    assigned a zone rather than falling back to a fixed normalized window."""
    from cv_lab.scripts.pipeline.build_yolo_hand_timeline import build_states

    fixture = _hand_fixture()
    for frame in fixture:
        frame["detections"] = [d for d in frame["detections"] if d["cls"] != "stack_text"]
    states, _events = build_states(rd.frames_from_fixture(fixture))
    assert all(not s["anchor_ok"] for s in states)
    assert all(s["hero_cards"] == [] and s["board_cards"] == [] for s in states)
    assert any(s["unanchored_cards"] > 0 for s in states)


# --------------------------------------------------------------------------- #
# Settlement cut: the real sweep, not the next hand's blind pot.
#
# These run against 15 states retained verbatim from the 07-15 recording, whose
# hand reported a final pot of 1.0 against a true 240.9. Three independent
# defects compounded there; each test below pins one of them.
#   idx  8  t=14.0  pot=24.09  dealer=6  hero=[Ah,9c]  seat4 82.1 -> 323.0  <- real sweep
#   idx 13  t=20.0  pot=1.0    dealer=7  hero=[]       seat2 79.8 -> 797.0  <- next deal
#   idx 14  t=21.0  pot=None   dealer=None hero=[]     stacks={}            <- dropout tail
# --------------------------------------------------------------------------- #
def _g0715_settle_states(pots=None):
    """The retained states, with int seat keys (JSON forces string keys) and
    optionally a corrected pot series."""
    states = json.loads((CV_FIXTURES / "g0715_settle_states.json").read_text(encoding="utf-8"))
    for state in states:
        state["stacks"] = {int(k): v for k, v in state["stacks"].items()}
        state["bets"] = {int(k): v for k, v in state["bets"].items()}
        state["pills"] = {int(k): v for k, v in state["pills"].items()}
    if pots is not None:
        for idx, pot in pots.items():
            states[idx]["pot"] = pot
    return states


def test_settlement_ignores_uninformative_tail_state():
    """The trim loop used to STOP on the first state that failed both tests. The
    final state here has no hero cards, no pot, no dealer and no stacks -- it can
    neither corroborate nor refute anything, yet it shielded the next-deal state
    behind it from ever being examined."""
    from cv_lab.scripts.pipeline.build_yolo_hand_timeline import _trim_trailing_next_deal

    trimmed = _trim_trailing_next_deal(_g0715_settle_states())
    assert len(trimmed) == 13
    assert trimmed[-1]["time_s"] == 18.0


def test_settlement_ignores_seats_not_in_the_hand():
    """Seat 2 was never dealt into this hand (players are 0, 4 and 5) but was
    dealt into the NEXT one; its fresh stack showing up as a jump set the cut."""
    from cv_lab.scripts.pipeline.build_yolo_hand_timeline import _settle_index

    states = _g0715_settle_states()
    assert _settle_index(states) == 13                       # any seat may cut
    assert _settle_index(states, players={0, 4, 5}) != 13     # only this hand's seats


def test_settlement_finds_the_real_sweep_with_corrected_pots():
    """With the OCR decimal fixed (891.0 -> 89.1, 24.09 -> 240.9) the threshold
    0.4 * pot_so_far becomes 96.4 instead of 356.4, so the true +240.9 sweep
    qualifies and the +717.2 phantom no longer does."""
    from cv_lab.scripts.pipeline.build_yolo_hand_timeline import (
        _settle_index,
        _trim_trailing_next_deal,
    )

    corrected = {0: 89.1, 1: 165.0}
    corrected.update({i: 240.9 for i in range(2, 11)})
    states = _trim_trailing_next_deal(_g0715_settle_states(pots=corrected))
    settle = _settle_index(states, players={0, 4, 5})
    assert settle == 8
    pots = [s["pot"] for s in states[: settle + 1] if s["pot"] is not None]
    assert pots[-1] == 240.9


def test_the_next_deals_blind_pot_never_reaches_the_hands_pot_series():
    """The invariant the old collapse-carry protected, now owned by
    segmentation and the tail trim: a post-sweep blind pot (which reads EXACTLY
    1.0 -- one big blind) must not corrupt the hand's pot series. The carry
    itself is gone because it was the wrong owner: on a session whose hand
    boundaries were missed it overwrote the next deal's real blind pot with the
    old hand's pot 35 times in one merged mega-hand, erasing the very evidence
    that a new deal had started. Real g0715 tail: the dealer-marked 1.0 state
    is pruned; the debounce never invents a pot for it."""
    from cv_lab.scripts.pipeline.build_yolo_hand_timeline import (
        _debounce_pot,
        _segment,
        _trim_trailing_next_deal,
    )

    states = _g0715_settle_states()
    (hand,) = _segment(states)          # teardown tail trails, no phantom hand
    trimmed = _trim_trailing_next_deal(hand)
    assert 1.0 not in [s["pot"] for s in trimmed]

    # And the debounce itself is revert-only: it repairs an A -> B -> A blip
    # but never carries the old pot over a reading with no revert evidence.
    blip = [{"pot": 240.9}, {"pot": 1.0}, {"pot": 240.9}]
    _debounce_pot(blip)
    assert [s["pot"] for s in blip] == [240.9, 240.9, 240.9]
    tail = [{"pot": 240.9}, {"pot": 240.9}, {"pot": 1.0}]
    _debounce_pot(tail)
    assert [s["pot"] for s in tail] == [240.9, 240.9, 1.0]


def test_transient_pot_misreads_do_not_escalate_to_a_warning():
    """A hand whose pot misread transiently mid-stream reconstructs the RIGHT
    pot; flagging it as an implausible-amount hand would reject a good hand.
    Seen for real on 07-15 (240.9 misread as 2.0 and 24.0). The old collapse
    carry counted these as repairs; with it gone the revert-only debounce and
    the settle logic absorb them, and nothing escalates."""
    from cv_lab.scripts.pipeline.build_yolo_hand_timeline import reconstruct

    states, _events = _states_from(_hand_fixture())
    accepted = [i for i, s in enumerate(states) if (s["pot"] or 0) > 6.0]
    for i in accepted[1:3]:
        states[i]["pot"] = 1.0            # transient pot misread, twice
    hand = reconstruct(states, 1)
    assert hand["pot"] == 20.0            # the true final pot survives the dips
    assert hand["pot_collapses_repaired"] == 0
    assert "amount_scale_implausible" not in hand["warnings"]
    assert hand["warnings"] == []


def test_debounce_pot_keeps_a_legitimate_small_pot_series():
    """False-positive guard: the collapse arm only applies once the accepted pot
    is above the calibrated floor, so a genuinely small hand is untouched."""
    from cv_lab.scripts.pipeline.build_yolo_hand_timeline import _debounce_pot

    hand = [{"pot": 3.0}, {"pot": 1.5}, {"pot": 1.5}]
    _debounce_pot(hand)
    assert [s["pot"] for s in hand] == [3.0, 1.5, 1.5]


# --------------------------------------------------------------------------- #
# Side pot: detected and explicitly rejected, never silently discarded.
# --------------------------------------------------------------------------- #
def _side_pot_frame(time_s, *, main, side, **kw):
    frame = _frame(time_s, **kw)
    for value, ny in ((main, 0.32), (side, 0.55)):
        if value is not None:
            frame["detections"].append(_det("pot_text", 0.50, ny, value))
    return frame


def test_side_pot_raises_unsupported_warning():
    fixture = [
        _side_pot_frame(0.0, main=20, side=None, board=[], s0=100, s4=100),
        _side_pot_frame(1.0, main=20, side=4.0, board=[], s0=100, s4=96),
        _side_pot_frame(2.0, main=24, side=4.0, board=FLOP, s0=96, s4=96),
    ]
    hand = build_hand_timeline(rd.frames_from_fixture(fixture))["hands"][0]
    assert hand["side_pot"] == 4.0
    assert "side_pot_unsupported" in hand["warnings"]
    # The main pot keeps its exact meaning; the side pot never replaces it.
    assert hand["pot"] is not None and hand["pot"] >= 20


def test_single_pot_hand_raises_no_side_pot_warning():
    hand = _build()["hands"][0]
    assert hand["side_pot"] is None
    assert "side_pot_unsupported" not in hand["warnings"]


# --------------------------------------------------------------------------- #
# Rejection signal B: implausible amount magnitude vs the observed stack scale.
# --------------------------------------------------------------------------- #
_OUTLIER_STACKS = {0: 79.8, 1: 78.7, 2: 79.7, 3: 78.2, 4: 79.2, 5: 322.9, 6: 797.0, 7: 79.3}


def _stack_frame(sources):
    dets = [
        {"cls": "stack_text", "conf": 0.9, "attr": _OUTLIER_STACKS[seat],
         "attr_source": sources.get(seat),
         "xyxy": [nx * W - 40, ny * H - 15, nx * W + 40, ny * H + 15]}
        for seat, (nx, ny) in rd.SEAT_ANCHORS_BY_CLASS["stack_text"].items()
    ]
    return rd.frames_from_fixture(
        [{"image": "f0", "time_s": 0.0, "width": W, "height": H, "detections": dets}]
    )[0]


def test_the_sibling_median_stack_net_no_longer_exists():
    """SUPERSEDES test_stack_outlier_rejected_when_no_decimal_was_read,
    test_stack_outlier_kept_when_the_decimal_was_located,
    test_fixture_path_never_triggers_amount_rejection and
    test_repeated_amount_rejection_warns_on_the_hand.

    `_reject_stack_outliers` dropped any stack read at least 6.0x the frame's
    other stacks when the read had located no decimal point. Its premise -- "no
    field in this corpus legitimately exceeds ~400 BB" -- was measured on five
    recordings and is FALSE on the sixth: clubwpt_session_01 carries 541 provable
    reads at or above 1000 BB (max 1157.10), and instrumented there the net fires
    on 83 frames, drops a legible 1110.0 BB stack against a ~150 sibling median
    every time, and raises `amount_scale_implausible` (a SPINE_FATAL code) on
    three hands.

    THE MEASURED COST OF DELETING IT, recorded rather than assumed: the dropped
    decimal it was built for -- "79.7" read as "797.0" -- is now refused at the
    reader by P5 (`integer_over_decimal_band`), which keys on the numeral's own
    inter-digit spacing. So the read below never reaches the spine at all under
    the shipped reader, and a fixture that hands the spine 797.0 directly is
    asserting a value no reader produces. Exports rose 16 -> 19 on the
    development corpus with no other recording changed.

    What remains as the downward net is `_stack_ledger_violations`, which needs
    no sibling comparison. This test pins the DELETION so the ratio rule cannot
    be reintroduced without re-reading that measurement."""
    import cv_lab.scripts.pipeline.build_yolo_hand_timeline as spine

    for gone in ("_reject_stack_outliers", "_STACK_OUTLIER_RATIO",
                 "_STACK_OUTLIER_MIN_READS", "_DECIMAL_NOT_LOCATED"):
        assert not hasattr(spine, gone), (
            f"{gone} is back; a magnitude ratio cannot separate a broken read "
            "from a deep stack -- see clubwpt_session_01's 1110.0 BB")

    from cv_lab.scripts.pipeline.build_yolo_hand_timeline import _frame_state

    # A 10.03x sibling disparity is now KEPT: the spine no longer second-guesses
    # a value the reader proved.
    state = _frame_state(_stack_frame(dict.fromkeys(_OUTLIER_STACKS, "integer")))
    assert state["stacks"][6] == 797.0
    assert state["stacks"][5] == 322.9
    assert "amounts_rejected" not in state
    assert "stack_outlier_check_skipped" not in state


def test_a_deep_stack_is_no_longer_dropped_for_being_deep():
    """The clubwpt_session_01 shape, in miniature: one seat at 7.4x the sibling
    median, every read a proven integer. Under the deleted net the seat vanished
    from `stacks` on every frame; it must now survive into the hand."""
    from cv_lab.scripts.pipeline.build_yolo_hand_timeline import _frame_state, build_states

    deep = {0: 150.0, 1: 148.0, 2: 152.0, 3: 149.0, 4: 151.0, 5: 147.0,
            6: 1110.0, 7: 150.0}
    frame = _stack_frame(dict.fromkeys(_OUTLIER_STACKS, "integer"))
    for det in frame.detections:
        cx = (det.xyxy[0] + det.xyxy[2]) / 2.0 / W
        cy = (det.xyxy[1] + det.xyxy[3]) / 2.0 / H
        det.attr = deep[rd._nearest_seat(cx, cy, "stack_text")]
    assert _frame_state(frame)["stacks"][6] == 1110.0
    states, _events = build_states(rd.frames_from_fixture(_hand_fixture()))
    assert all("amounts_rejected" not in s for s in states)


# --------------------------------------------------------------------------- #
# Reconstruction-correctness nets (the failures that used to raise nothing).
# --------------------------------------------------------------------------- #
def _states_from(fixture):
    from cv_lab.scripts.pipeline.build_yolo_hand_timeline import build_states

    return build_states(rd.frames_from_fixture(fixture))


def test_board_row_missed_escalates_to_a_hand_warning():
    """Three or more states showing a community row that nothing zoned as board,
    with an empty final board, is the exact 06-21 failure."""
    from cv_lab.scripts.pipeline.build_yolo_hand_timeline import reconstruct

    states, _events = _states_from(_hand_fixture())
    for state in states:
        state["board_cards"] = []
        state["board_row_missed"] = True
    hand = reconstruct(states, 1)
    assert "board_zone_yield_zero" in hand["warnings"]


def test_board_zone_yield_zero_needs_a_row_the_zone_test_actually_lost():
    """The false-positive control, restated on the quantity that governs it.

    This used to assert that ONE state of board_row_missed is not enough -- a
    "mid-deal transient" argument. Measurement does not support it: over all 1309
    card-bearing frames of the five development recordings, deal animations
    included, board_row_missed is raised on ZERO of them. The run length was
    therefore protecting nothing, while gating the net above the 1-2 distinct
    states a real river street supplies (see
    test_board_zone_nets_fire_on_the_evidence_a_river_actually_supplies).

    What must stay true is this: no state, no warning."""
    from cv_lab.scripts.pipeline.build_yolo_hand_timeline import reconstruct

    states, _events = _states_from(_hand_fixture())
    for state in states:
        state["board_cards"] = []
    assert not any(s.get("board_row_missed") for s in states)
    assert "board_zone_yield_zero" not in reconstruct(states, 1)["warnings"]


def test_spine_does_not_raise_cross_field_street_warnings():
    """`streets`, the voted board and every action's street all derive from the
    same board_cards readings, so inside the spine they cannot disagree. The two
    cross-field invariants live in the validator; this pins that they are not
    also emitted here as dead code."""
    from cv_lab.scripts.pipeline.build_yolo_hand_timeline import reconstruct

    states, _events = _states_from(_hand_fixture())
    hand = reconstruct(states, 1)
    assert "actions_collapsed_to_one_street" not in hand["warnings"]
    assert "board_empty_but_streets_advanced" not in hand["warnings"]

    for state in states:
        state["board_cards"] = []
    stripped = reconstruct(states, 1)
    assert stripped["board"] == []
    assert [s["street"] for s in stripped["streets"]] == ["preflop"]


def test_unanchored_cards_warn_that_the_record_is_incomplete():
    from cv_lab.scripts.pipeline.build_yolo_hand_timeline import reconstruct

    states, _events = _states_from(_hand_fixture())
    assert "anchor_unavailable" not in reconstruct(states, 1)["warnings"]
    states[2]["unanchored_cards"] = 3
    hand = reconstruct(states, 1)
    assert "anchor_unavailable" in hand["warnings"]
    assert hand["anchor_missing_states"] == 0     # anchor_ok was still True


def test_true_zero_stack_survives_to_the_player_row():
    """A seat showing "0 BB" (all-in) reads a confident 0.0 and must reach the
    player row as 0.0; a seat never read must reach it as None. Conflating the
    two would erase every all-in seat from the stack ledger."""
    fixture = []
    for i, board in enumerate(([], [], FLOP)):
        frame = _frame(float(i), board=board, s0=100, s4=0.0, pot=6)
        fixture.append(frame)
    hand = build_hand_timeline(rd.frames_from_fixture(fixture))["hands"][0]
    by_seat = {p["seat"]: p for p in hand["players"]}
    assert by_seat[4]["starting_stack"] == 0.0
    assert by_seat[4]["starting_stack"] is not None


# --------------------------------------------------------------------------- #
# Adversarial round 1: the stack ledger has to be arithmetically possible.
# --------------------------------------------------------------------------- #
def test_a_stack_that_grows_mid_hand_is_a_warning():
    """A player's stack cannot rise before the pot is awarded. Two exported hands
    violated it and both scored 1.0 with warnings=none: a truncated OCR read
    (218 -> 21.0) made seat 3's stack go 21.0 -> 210.5 after it called 7.5, and a
    false zero made another go 0.0 -> 69.2 after an invented all-in. This is the
    cheapest net under both, because it is pure arithmetic over data the spine
    already has and never looks at a pixel."""
    from cv_lab.scripts.pipeline.build_yolo_hand_timeline import reconstruct

    states, _events = _states_from(_hand_fixture())
    assert "stack_ledger_incoherent" not in reconstruct(states, 1)["warnings"]

    seat = next(iter(states[0]["stacks"]))
    base = states[0]["stacks"][seat]
    for state in states[1:]:
        if seat in state["stacks"]:
            state["stacks"][seat] = base * 10.0
    hand = reconstruct(states, 1)
    assert "stack_ledger_incoherent" in hand["warnings"]


def test_the_settlement_sweep_is_not_a_stack_ledger_violation():
    """Negative control: the winner's stack rises by the pot at settlement, which
    is the one legal way a stack grows inside a hand."""
    from cv_lab.scripts.pipeline.build_yolo_hand_timeline import reconstruct

    states, _events = _states_from(_hand_fixture())
    hand = reconstruct(states, 1)
    assert hand["winner_seat"] is not None
    assert "stack_ledger_incoherent" not in hand["warnings"]


def test_a_zero_bet_is_unread_not_a_zero_bet():
    """A bet_text crop holding only chip sprites reads a confident 0.0 -- the
    chip's white annulus matches the '0' template. The client never renders a
    0 BB bet, so mirror the pot rule: 0 means the region was not read."""
    from cv_lab.scripts.pipeline.build_yolo_hand_timeline import _frame_state

    frame = _stack_frame({})
    frame.detections.append(rd.Detection(cls="bet_text", conf=0.9,
                                         xyxy=(0.46 * 2054 - 40, 0.579 * 1470 - 15,
                                               0.46 * 2054 + 40, 0.579 * 1470 + 15),
                                         attr=0.0))
    assert _frame_state(frame)["bets"] == {}


def test_partial_board_row_yield_escalates_to_a_hand_warning():
    """All-or-nothing was never the failure mode. A community row that straddles a
    band edge yields most of its cards and drops the rest, and 3, 4 and 5 are all
    legal board counts -- so a 5-card river board exported as a completed 4-card
    turn board at confidence 1.0 with no warning."""
    from cv_lab.scripts.pipeline.build_yolo_hand_timeline import reconstruct

    states, _events = _states_from(_hand_fixture())
    assert "board_zone_yield_partial" not in reconstruct(states, 1)["warnings"]
    for state in states:
        state["board_row_partial"] = True
    assert "board_zone_yield_partial" in reconstruct(states, 1)["warnings"]


def test_board_zone_yield_partial_is_silent_when_no_state_raises_it():
    """Sibling control of the above. The "a card mid-deal is momentarily outside
    the row" argument that justified a 3-state run is not borne out: the flag is
    raised on 0 of 1309 real card-bearing frames, deal animations included."""
    from cv_lab.scripts.pipeline.build_yolo_hand_timeline import reconstruct

    states, _events = _states_from(_hand_fixture())
    assert not any(s.get("board_row_partial") for s in states)
    assert "board_zone_yield_partial" not in reconstruct(states, 1)["warnings"]


def test_a_genuine_short_stack_is_not_rejected_as_an_outlier():
    """The mirrored 'reject reads far BELOW the median' arm would be a defect, not
    a fix. Both stacks below are real on screen in the development corpus (18.90 BB
    against a 199.0 sibling median, and 31.20 against 185.0), so a deflation-ratio
    net at 6.0 discards genuine short stacks. Pinning that here so the symmetry
    argument cannot be re-applied without re-reading the measurement.

    The inflation arm it was the mirror of is now deleted too (see
    test_the_sibling_median_stack_net_no_longer_exists), which strictly widens
    what this test asserts: NO sibling-magnitude comparison rejects a stack read
    in either direction."""
    from cv_lab.scripts.pipeline.build_yolo_hand_timeline import _frame_state

    stacks = {0: 194.1, 1: 319.7, 2: 210.0, 3: 199.5, 4: 18.9, 5: 216.4,
              6: 205.0, 7: 190.8}
    frame = _stack_frame(dict.fromkeys(_OUTLIER_STACKS, "integer"))
    for det in frame.detections:
        cx = (det.xyxy[0] + det.xyxy[2]) / 2.0 / W
        cy = (det.xyxy[1] + det.xyxy[3]) / 2.0 / H
        det.attr = stacks[rd._nearest_seat(cx, cy, "stack_text")]
    assert _frame_state(frame)["stacks"][4] == 18.9


def test_an_uncalled_shove_returned_before_the_sweep_is_not_a_violation():
    """Negative control 1. A rise IS legal: the client returns the uncalled part
    of an over-shove before settlement, so an all-in seat reads 0 and then reads
    the returned remainder. Measured twice on the baseline recording (0.0 -> 18.9
    on hand 2 and 0.0 -> 128.4 on hand 5). A "the stack must never rise" rule
    flags both; "never above its own starting stack" flags neither."""
    from cv_lab.scripts.pipeline.build_yolo_hand_timeline import _stack_ledger_violations

    window = [
        {"time_s": 0.0, "stacks": {0: 197.5}, "dealt_in": [0]},
        {"time_s": 1.0, "stacks": {0: 167.0}, "dealt_in": [0]},
        {"time_s": 2.0, "stacks": {0: 0.0}, "dealt_in": [0]},
        {"time_s": 3.0, "stacks": {0: 18.9}, "dealt_in": [0]},
    ]
    assert _stack_ledger_violations(window, [0]) == []


def test_a_folded_seat_topping_back_up_is_not_a_violation():
    """Negative control 2. A seat that has folded is out of the ledger, and the
    client refills it to the buy-in without waiting for the hand to end -- on the
    baseline recording seat 6 folds on the turn and goes 164.0 -> 200.0 while the
    river is still being played."""
    from cv_lab.scripts.pipeline.build_yolo_hand_timeline import _stack_ledger_violations

    window = [
        {"time_s": 0.0, "stacks": {6: 183.0}, "dealt_in": [0, 6]},
        {"time_s": 1.0, "stacks": {6: 164.0}, "dealt_in": [0, 6]},
        {"time_s": 2.0, "stacks": {6: 164.0}, "dealt_in": [0]},   # folded
        {"time_s": 3.0, "stacks": {6: 200.0}, "dealt_in": [0]},   # auto top-up
    ]
    assert _stack_ledger_violations(window, [6]) == []


def test_a_seat_holding_more_than_it_started_with_is_a_violation():
    """The positive case, in the exact shape the truncated OCR read produced:
    seat 3's first stack of the hand read 21.0 (the screen said "218 BB") and two
    actions later it held 210.5."""
    from cv_lab.scripts.pipeline.build_yolo_hand_timeline import _stack_ledger_violations

    window = [
        {"time_s": 0.0, "stacks": {3: 21.0}, "dealt_in": [3]},
        {"time_s": 1.0, "stacks": {3: 210.5}, "dealt_in": [3]},
    ]
    hits = _stack_ledger_violations(window, [3])
    assert [h["seat"] for h in hits] == [3]


# --------------------------------------------------------------------------- #
# Terminal-event and winner correctness. These are the fields the app's
# completion evidence is derived from: derive_completion_status only refuses a
# hand when terminal_event is "" or "unobserved", so a fabricated winner that
# manufactures a "showdown" promotes a truncated hand to "complete".
# --------------------------------------------------------------------------- #
def _truncated_last_hand_fixture():
    """The last hand of a recording that simply stops: hero folds preflop, one
    villain raises, nothing is ever swept and no flop is dealt. The villain's
    stack then JITTERS by 0.5 BB between the last two samples -- the entire
    evidence base for the "Villain wins" this used to publish.

    Measured shape, from the 1272x896 development recording (video duration
    197.28 s, last sampled state t=196.0): seat 3's settled series ran
    [190.5 x5, 182.0, 182.5] on a 17.5 BB pot, i.e. a "gain" of 0.03x the pot.
    """
    return [
        _frame(0.0, board=[], s0=100.0, s4=100.0, pot=7.5, active=4),
        _frame(1.0, board=[], s0=100.0, s4=100.0, pot=7.5, pill0="fold", active=4),
        _frame(2.0, board=[], s0=100.0, s4=91.5, pot=17.5, pill0="fold",
               pill4="raise", active=0),
        _frame(3.0, board=[], s0=100.0, s4=92.0, pot=17.5, pill0="fold",
               pill4="raise", active=0),
    ]


def test_stack_jitter_far_below_the_pot_cannot_name_a_winner():
    """A seat that swept the pot gained approximately the pot. Across the 5
    development geometries every real winner's gain/pot lands in [0.96, 1.03];
    this one is 0.03. The phantom guard only rejected gains ABOVE 1.5x the pot,
    so 0.5 BB of OCR noise named a winner, set result='Villain wins', and (via
    terminal_event) promoted a recording-truncated hand to complete."""
    hand = build_hand_timeline(rd.frames_from_fixture(_truncated_last_hand_fixture()))["hands"][0]
    assert hand["winner_seat"] is None, f"win_gain {hand['win_gain']} on pot {hand['pot']}"
    assert hand["result"] != "Villain wins"


def test_a_zero_card_board_is_never_a_showdown():
    """terminal_event was derived from the winner plus len(dealt_in) >= 2 with no
    reference to the board at all -- and hero stays in dealt_in after folding,
    because the client leaves a folded hero's cards on screen greyed out. The
    result was a published 'showdown' on board_cards=''."""
    hand = build_hand_timeline(rd.frames_from_fixture(_truncated_last_hand_fixture()))["hands"][0]
    assert hand["board"] == []
    assert hand["terminal_event"] != "showdown"


def test_showdown_requires_a_complete_board():
    """Across the 5 development geometries, 9 of 21 hands claimed 'showdown' and 8
    of those had fewer than 5 board cards. A showdown is two or more seats seeing
    a completed board; anything else that ends in a sweep is a fold win."""
    from cv_lab.scripts.pipeline.build_yolo_hand_timeline import reconstruct

    states, _events = _states_from(_hand_fixture())
    full = reconstruct(states, 1)
    assert len(full["board"]) == 5
    # Two seats contesting a completed board, neither having folded: a showdown.
    assert full["terminal_event"] == "showdown"
    # Same hand cut off on the turn. The sweep is real, the showdown is not -- and
    # neither is a fold win, because nobody folded. Nobody shows down an incomplete
    # board, so the later streets were simply not observed.
    turn_only = [dict(s) for s in states]
    for state in turn_only:
        state["board_cards"] = state["board_cards"][:4]
    cut = reconstruct(turn_only, 1)
    assert cut["winner_seat"] is not None
    assert cut["terminal_event"] == "unobserved"


def test_bet_text_fallback_does_not_re_emit_a_stack_derived_action():
    """The fallback was guarded by `elif not stack_flat`, i.e. skipped only when
    the stack was PROVABLY unchanged. A stack that reads 190.5 -> 182.0 -> 182.5
    is neither dropped nor flat on the second transition, so the seat's raise --
    already emitted from the stack delta -- was emitted a second time from the
    bet_text delta at a slightly different size ('raise 8.5' then 'raise 8.0'),
    double-counting the money and driving the hand's contributions above its pot.
    A stack that RISES is not a seat putting chips in."""
    fixture = _truncated_last_hand_fixture()
    # The felt's bet chip renders a sample AFTER the stack has already dropped, so
    # the standing raise first becomes readable on the last state -- exactly the
    # dropout the carried-bet high-water mark cannot cover, because there is no
    # earlier reading to carry.
    fixture[3]["detections"].append(_det("bet_text", 0.50, 0.26, 8.0))
    hand = build_hand_timeline(rd.frames_from_fixture(fixture))["hands"][0]
    raises = [a for a in hand["actions"] if a["seat"] == 4 and a["action_type"] == "raise"]
    assert len(raises) == 1, [(a["amount"], a["derivation"]) for a in raises]


def test_contributions_above_the_pot_are_a_warning():
    """With incremental amount semantics every chip in an action is a chip in the
    pot, so the contributions can never EXCEED it. Four of 15 exported hands did,
    all at confidence 1.0 with warnings=none. The shape below is the measured
    cause: the same raise emitted twice, once from the stack delta and once from
    the bet_text delta."""
    from cv_lab.scripts.pipeline.build_yolo_hand_timeline import (
        _contribution_residual,
        reconstruct,
    )

    states, _events = _states_from(_hand_fixture())
    hand = reconstruct(states, 1)
    assert "contributions_exceed_pot" not in hand["warnings"]
    assert _contribution_residual(hand) >= 0

    duplicated = dict(hand)
    duplicated["actions"] = [*hand["actions"], {
        "street": "flop", "action_index": 99, "seat": 4, "player_name": "Seat4",
        "position": "BTN", "action_type": "call", "amount": 7.0,
        "pot_before": None, "stack_before": None, "source_time_s": None,
        "source_image": None, "source_state_index": None,
        "derivation": "bet_text_delta",
    }]
    assert _contribution_residual(duplicated) < 0


def test_an_uncalled_over_shove_is_not_a_conservation_defect():
    """False-positive guard, and the reason the net subtracts a headroom rather
    than reading the sign alone. A shove above what any opponent can match is
    refunded before the sweep, so the pot legitimately excludes it: on the
    baseline recording seat 5 shoves 219.9 into a seat 3 holding 97.5 total, and
    122.4 BB comes straight back. Both development hands with a negative residual
    are of exactly this kind (-13.4 against 18.9 of headroom, -122.9 against
    128.4) and neither is a reconstruction fault."""
    from cv_lab.scripts.pipeline.build_yolo_hand_timeline import _uncalled_shove_headroom

    actions = [
        {"seat": 5, "action_type": "raise", "amount": 8.0},
        {"seat": 3, "action_type": "raise", "amount": 22.0},
        {"seat": 5, "action_type": "all-in", "amount": 219.9},
        {"seat": 3, "action_type": "all-in", "amount": 75.5},
    ]
    assert _uncalled_shove_headroom(actions) == 130.4      # 227.9 - 97.5
    # No all-in anywhere -> no headroom, so a duplicated action has nothing to
    # hide behind. This is the direction the net must keep.
    assert _uncalled_shove_headroom([
        {"seat": 3, "action_type": "raise", "amount": 8.5},
        {"seat": 3, "action_type": "raise", "amount": 8.0},
    ]) == 0.0


def test_result_and_hero_net_must_agree_in_sign():
    """A record that says 'Villain wins' while crediting the hero +65.2 BB
    contradicts itself. Nothing compared the two, so the contradiction shipped at
    confidence 1.0 with warnings=none."""
    from cv_lab.scripts.pipeline.build_yolo_hand_timeline import _result_contradicts_hero_net

    assert _result_contradicts_hero_net("Villain wins", 65.2) is True
    assert _result_contradicts_hero_net("Hero wins", -12.0) is True
    assert _result_contradicts_hero_net("Villain wins", -12.0) is False
    assert _result_contradicts_hero_net("Hero wins", 65.2) is False
    assert _result_contradicts_hero_net("Villain wins", 0.0) is False
    assert _result_contradicts_hero_net("", 65.2) is False


def test_a_folded_hero_cannot_net_positive():
    """The THIRD result value, which the sign check did not cover (round 2).

    `result` comes from a closed vocabulary -- "Hero wins", "Villain wins",
    "Hero folds", "" -- and the check enumerated two of the four, so a hand
    claiming the hero FOLDED while crediting it a positive net could not raise
    result_contradicts_hero_net. A hero who folds forfeits what it committed; the
    case where the hero takes the pot uncontested is winner_seat == 0, i.e.
    "Hero wins", so the fold branch is only reached when the hero surrendered and
    a positive net there is the same self-contradiction the other two catch.

    Latent on the corpus, deliberately pinned anyway: all 5 development
    "Hero folds" hands net -0.5, 0.0 or unknown, so nothing here would have
    caught the omission by running the pipeline."""
    from cv_lab.scripts.pipeline.build_yolo_hand_timeline import (
        _RESULT_FORBIDS_NET_SIGN,
        _result_contradicts_hero_net,
    )

    assert _result_contradicts_hero_net("Hero folds", 65.2) is True
    assert _result_contradicts_hero_net("Hero folds", 0.5) is True
    # The legal directions for a folded hero: a loss, a free fold, an unread net.
    assert _result_contradicts_hero_net("Hero folds", -12.0) is False
    assert _result_contradicts_hero_net("Hero folds", 0.0) is False
    assert _result_contradicts_hero_net("Hero folds", None) is False
    # Every result the spine can PUBLISH is either in the table or is the empty
    # "unresolved" token. A new result value added to the producer without a rule
    # here would silently get no sign check at all, which is how "Hero folds"
    # went uncovered.
    assert set(_RESULT_FORBIDS_NET_SIGN) == {"Hero wins", "Villain wins", "Hero folds"}


def test_the_spine_names_no_reader_token_it_does_not_own():
    """SUPERSEDES test_the_stack_outlier_nets_vocabulary_tracks_the_readers,
    which SUPERSEDED test_stack_outlier_net_is_not_disabled_by_an_INFERRED_decimal.

    Round 4, adversary B found that `attr_source == "gap"` bought total immunity
    from the stack-magnitude net; round 5 found the frozenset naming those tokens
    would go SILENTLY INERT the moment the reader renamed one. Both defects are
    now structurally impossible: the net is deleted, so the spine holds no
    frozenset of reader tokens to drift out of date.

    What remains is the coupling itself. Any token this file DOES quote from the
    reader must still be a token the reader can produce -- otherwise the same
    silent-inertness returns through a different door. The refusal codes the
    spine raises itself are declared here so a rename in either module fails
    loudly."""
    import cv_lab.scripts.pipeline.build_yolo_hand_timeline as spine
    from cv_lab.scripts.pipeline import region_detections as rdm
    from cv_lab.scripts.pipeline.ocr_readers import DECIMAL_EVIDENCE, REFUSAL_CODES

    assert not any(name.startswith("_DECIMAL_NOT") for name in vars(spine)), (
        "the spine must not re-declare the reader's decimal vocabulary")
    # The region layer's own refusal codes are disjoint from the reader's value
    # evidence and join its refusal namespace without colliding.
    own = {rdm.AMOUNT_UNREADABLE_ATTR, rdm.AMOUNT_STACK_BOXES_DISAGREE,
           rdm.AMOUNT_POT_ZERO_IMPOSSIBLE, rdm.AMOUNT_UNSPECIFIED}
    assert not (own & DECIMAL_EVIDENCE), "a refusal code must never mean a value"
    assert not (own & REFUSAL_CODES), "a layer must not shadow a reader code"
    assert rdm.AMOUNT_READER_UNAVAILABLE in REFUSAL_CODES, (
        "the reader owns this token; the region layer only sets it")


def test_a_seat_that_folded_before_the_first_observed_state_is_still_a_player():
    """Round 4, adversary C: the spine emitted an action for a seat its own
    ``players`` list did not contain, and the app's ingest rolls the entire
    session back on that.

    The player set was derived from card_back evidence alone, but a seat that
    folded before the hand came into view has no card_back -- the client removes
    its cards and leaves only the FOLD pill, which stays on the felt for the rest
    of the street. ``_reconstruct_actions`` reads exactly that pill and emits the
    fold, so the ledger names a seat the roster does not.

    Measured on the 07-11 recording: hand 5 listed seats {0,1,2,3,4,7} with a
    preflop ``seat:5`` fold in its actions. Seat 5 folded to seat 4's raise one
    sample before the hand's first observed state.

    A persistent pill is participation evidence of the same kind as a persistent
    card_back, and is held to the same two-state bar so a single misdetection
    cannot conjure a phantom player."""
    from cv_lab.scripts.pipeline.build_yolo_hand_timeline import reconstruct

    states, _events = _states_from(_hand_fixture())
    for state in states:
        state["pills"] = {**state.get("pills", {}), 5: "fold"}
    hand = reconstruct(states, 1)

    seats = {row["seat"] for row in hand["players"]}
    assert 5 in seats, "a seat whose fold the spine BOOKED is a player of the hand"
    assert {a["seat"] for a in hand["actions"]} <= seats


def test_a_single_frame_pill_does_not_conjure_a_phantom_player():
    """The other half of the rule above: one state of pill evidence is a
    misdetection, not a participant. Same bar the card_back rule already uses."""
    from cv_lab.scripts.pipeline.build_yolo_hand_timeline import reconstruct

    states, _events = _states_from(_hand_fixture())
    states[len(states) // 2]["pills"] = {**states[len(states) // 2].get("pills", {}),
                                         6: "fold"}
    hand = reconstruct(states, 1)
    assert 6 not in {row["seat"] for row in hand["players"]}


def test_an_unobserved_starting_stack_is_unknown_not_backfilled():
    """Round 4, adversary B: an unreadable stack did not propagate as UNKNOWN, it
    was silently backfilled from a later sample.

    ``_frame_state`` drops a seat whose stack read failed closed, so "unknown" and
    "no stack_text box on screen" become the same state; ``starting_stack`` is then
    the seat's FIRST surviving reading, which is taken after the seat has already
    put money in. Measured by blanking one seat's reads over a real hand: BTN's
    published starting_stack fell 123.4 -> 115.4 and SB's 218.0 -> 210.5, presented
    as fact, with warnings=none.

    An understated starting stack corrupts effective stack and SPR directly, which
    is the core quantity a post-session study tool computes. PLAN.md is explicit:
    treat absent amounts as unknown, not as a value.

    Measured cost of refusing to guess: 3 of 148 player rows across the five
    development recordings lose a starting_stack they never actually evidenced."""
    from cv_lab.scripts.pipeline.build_yolo_hand_timeline import reconstruct

    states, _events = _states_from(_hand_fixture())
    baseline = {row["seat"]: row["starting_stack"] for row in reconstruct(states, 1)["players"]}
    assert baseline[4] is not None

    states, _events = _states_from(_hand_fixture())
    for state in states[:3]:
        state["stacks"] = {s: v for s, v in state["stacks"].items() if s != 4}
    rows = {row["seat"]: row["starting_stack"] for row in reconstruct(states, 1)["players"]}

    assert rows[4] is None, "an unobserved starting stack must not be backfilled"
    assert rows[4] != baseline[4]
    assert rows[0] == baseline[0], "seats that WERE observed are unaffected"


def test_a_contested_card_identity_is_a_warning_not_a_majority_vote():
    """Round 4, adversary A: the exported g0723a hand 4 published board
    'Js 6h 4c 8d' for a board that is really Js 6h 4c 8-of-HEARTS, at confidence
    1.0 with tags [] and no warning. Pixel ground truth confirmed on the frame.

    The card classifier is not stable on that turn card: over the ten samples that
    show it, it reads 8d eight times and 8h twice. ``_debounce_cards`` is built to
    absorb exactly that kind of suit excursion, and it did -- it replaced the
    minority reading with the accepted one and discarded the disagreement, so
    ``_vote_board`` never saw a contest and nothing downstream could.

    The majority is not evidence when the minority exists: it means the classifier
    could not decide, and here the majority was WRONG. The real board has two
    hearts (a live flush draw); the exported one has one, so every equity, texture
    and coaching conclusion drawn from the hand is against a different board.

    Discriminating measurement, over all five development recordings: a SAME-LENGTH
    disagreement at a fixed board index occurs exactly twice, and both are this one
    card. (Different-length disagreements are board growth / partial reads and are
    not contests -- 8 of those occur and are correctly ignored.)"""
    from cv_lab.scripts.pipeline.build_yolo_hand_timeline import build_states, reconstruct

    fixture = _hand_fixture()
    clean, _ev = build_states(rd.frames_from_fixture(fixture))
    assert "board_card_identity_split" not in reconstruct(clean, 1)["warnings"]

    # Flip ONE board card's suit for one frame, mid-flop: the shape the classifier
    # actually produces. The debounce still reverts it -- the board stays right --
    # but the contest must be reported.
    for row in fixture:
        cards = [d for d in row["detections"] if d["cls"] == "face_card" and d["attr"] == "9h"]
        if cards and row["time_s"] == 4.0:
            cards[0]["attr"] = "9d"
    states, _ev = build_states(rd.frames_from_fixture(fixture))
    hand = reconstruct(states, 1)
    assert hand["board"] == RIVER, "the debounce still keeps the majority board"
    assert "board_card_identity_split" in hand["warnings"]


def test_board_zone_nets_fire_on_the_evidence_a_river_actually_supplies():
    """Round 4, adversary A: the >= 3 state gate on board_zone_yield_zero /
    board_zone_yield_partial asked for more evidence than a real street produces.

    State collapse (the _signature dedup) compresses a hand to its distinct
    states, and measured over the five development recordings exported hands hold
    their FINAL board for as few as 1 or 2 distinct states (g0711 hand 6: 1;
    g0723a hand 5: 2), with several holding an intermediate street for exactly 1.
    So the two nets were gated above the evidence the river supplies -- they could
    not fire on the last street at all, independently of anything else.

    A state's board_row_partial / board_row_missed is a GEOMETRIC fact about the
    frame (a board-shaped row is on screen and the zone test did not yield it),
    not a numeric read, so it needs no run to be believed. Measured cost of firing
    on the first state: 0 of 470 states across the five development recordings
    raise either flag, so nothing measured changes."""
    from cv_lab.scripts.pipeline.build_yolo_hand_timeline import reconstruct

    states, _events = _states_from(_hand_fixture())
    for state in states:
        state["board_cards"] = []
    states[0]["board_row_missed"] = True
    assert "board_zone_yield_zero" in reconstruct(states, 1)["warnings"]

    states, _events = _states_from(_hand_fixture())
    states[-1]["board_row_partial"] = True
    assert "board_zone_yield_partial" in reconstruct(states, 1)["warnings"]


def test_board_zone_yield_zero_is_not_silenced_by_an_earlier_street():
    """Same finding, second arm: board_zone_yield_zero additionally required
    ``not board``, so a hand whose FLOP was captured and whose turn and river rows
    were then all missed could not raise it at all -- the very shape a mid-hand
    zoning drift produces."""
    from cv_lab.scripts.pipeline.build_yolo_hand_timeline import reconstruct

    states, _events = _states_from(_hand_fixture())
    for state in states:
        state["board_row_missed"] = True
    hand = reconstruct(states, 1)
    assert hand["board"], "the flop was captured -- this is not an empty board"
    assert "board_zone_yield_zero" in hand["warnings"]


# --------------------------------------------------------------------------- #
# Phase 6: reconstruction under UNKNOWN.
#
# The reader now REFUSES rather than guessing, so these paths are exercised far
# harder than before. Each test below is a place where a refusal used to be the
# TRIGGER for a weaker estimator or a backfill from a neighbouring state.
# --------------------------------------------------------------------------- #
def _refuse(fixture, cls, seat_xy, code, times):
    """Turn one class's read at `seat_xy` into a NAMED refusal on `times`."""
    for frame in fixture:
        if frame["time_s"] not in times:
            continue
        for det in frame["detections"]:
            if det["cls"] != cls:
                continue
            cx = (det["xyxy"][0] + det["xyxy"][2]) / 2.0 / W
            cy = (det["xyxy"][1] + det["xyxy"][3]) / 2.0 / H
            if abs(cx - seat_xy[0]) < 0.02 and abs(cy - seat_xy[1]) < 0.02:
                det["attr"] = None
                det["attr_source"] = code
    return fixture


def test_an_unreadable_stack_does_not_duplicate_an_already_booked_raise():
    """THE DEFECT THIS PHASE EXPOSED, and the reason the split matters.

    ``elif before is None or after is None`` made "the stack is unreadable" the
    TRIGGER for the bet_text-delta estimator, so a refusal did not reduce what the
    spine claimed -- it changed which evidence the claim rested on, and the
    fallback re-emitted a bet that was merely still standing on the felt.
    Measured on the real g0723b hand 1: blanking the hero's stack reads booked
    `preflop 0 raise 24.0` TWICE, 92.3 BB of action against a 73.8 BB pot, and the
    hand went warnings=[] -> contributions_exceed_pot. The spine got MORE wrong,
    not merely less complete, as reads became unknown.
    """
    hero = rd.SEAT_ANCHORS_BY_CLASS["stack_text"][0]
    fixture = _refuse(_hand_fixture(), "stack_text", hero, "suffix_not_bb",
                      {2.0, 3.0, 4.0, 5.0})
    hand = build_hand_timeline(rd.frames_from_fixture(fixture))["hands"][0]
    hero_money = [(a["street"], a["action_type"], a["amount"]) for a in hand["actions"]
                  if a["seat"] == 0 and a["action_type"] in {"bet", "raise", "call"}]
    assert len(hero_money) == len(set(hero_money)), f"duplicated action: {hero_money}"
    assert "contributions_exceed_pot" not in hand["warnings"]
    # ...and no amount is invented for the transitions nobody could measure.
    assert all(a["amount"] is None or a["derivation"] != "amount_unknown"
               for a in hand["actions"])


def test_a_money_action_with_an_unknown_amount_is_emitted_not_dropped():
    """A pill PROVES the act happened even when nothing can size it. The action is
    emitted with amount=None and derivation "amount_unknown" -- never dropped
    (the hand would then describe a ledger that never happened) and never given a
    fabricated number from OCR. Ledger inference may later size a call when the
    facing level is known; this fixture refuses BOTH stacks on the call so the
    facing raise is also unsized and the hole must survive."""
    hero = rd.SEAT_ANCHORS_BY_CLASS["stack_text"][0]
    villain = rd.SEAT_ANCHORS_BY_CLASS["stack_text"][4]
    fixture = _refuse(_hand_fixture(), "stack_text", villain, "run_clipped", {5.0})
    fixture = _refuse(fixture, "stack_text", hero, "run_clipped", {4.0})
    hand = build_hand_timeline(rd.frames_from_fixture(fixture))["hands"][0]
    unknown = [
        a for a in hand["actions"]
        if a["amount"] is None and a["action_type"] in {"bet", "raise", "call", "all-in"}
    ]
    assert unknown, "unsized money actions must survive when nothing can pin them"
    assert hand["unknown_money_actions"] == len(unknown)
    assert "amounts_unknown_in_ledger" in hand["warnings"]


def test_ledger_infer_fills_call_and_open_from_later_sized_actions():
    """Min-bar reconstruction: a call amount is facing - prior; an unsized open
    is back-solved from a later sized call under a known re-raise."""
    from cv_lab.scripts.pipeline.build_yolo_hand_timeline import _backfill_ledger_amounts

    positions = {0: "UTG", 1: "UTG+1", 6: "SB", 7: "BB"}
    actions = [
        {"street": "preflop", "action_type": "fold", "seat": 0, "amount": None,
         "derivation": "hero_dim"},
        {"street": "preflop", "action_type": "raise", "seat": 1, "amount": None,
         "derivation": "action_pill"},
        {"street": "preflop", "action_type": "call", "seat": 6, "amount": None,
         "derivation": "action_pill"},
        {"street": "preflop", "action_type": "raise", "seat": 7, "amount": 12.0,
         "derivation": "stack_delta"},
        {"street": "preflop", "action_type": "call", "seat": 1, "amount": 10.0,
         "derivation": "stack_delta"},
        {"street": "preflop", "action_type": "call", "seat": 6, "amount": 10.0,
         "derivation": "stack_delta"},
    ]
    filled = _backfill_ledger_amounts(actions, positions)
    by_key = {(a["seat"], a["action_type"], a.get("amount")): a for a in filled}
    assert by_key[(1, "raise", 3.0)]["derivation"].endswith("ledger_infer")
    assert by_key[(6, "call", 2.5)]["derivation"].endswith("ledger_infer")
    assert filled[3]["amount"] == 12.0


def test_ledger_infer_does_not_limp_fill_while_an_open_is_unsized():
    """Adversary: an unsized raise must dirty facing so the next call is not
    filled as a limp to the big blind."""
    from cv_lab.scripts.pipeline.build_yolo_hand_timeline import _backfill_ledger_amounts

    positions = {1: "UTG+1", 6: "SB", 7: "BB"}
    actions = [
        {"street": "preflop", "action_type": "raise", "seat": 1, "amount": None,
         "derivation": "action_pill"},
        {"street": "preflop", "action_type": "call", "seat": 6, "amount": None,
         "derivation": "action_pill"},
    ]
    filled = _backfill_ledger_amounts(actions, positions)
    assert filled[0]["amount"] is None
    assert filled[1]["amount"] is None


def test_ledger_infer_refuses_conflicting_caller_levels():
    """Adversary: two later calls that imply different raise-to levels must not
    invent a compromise open."""
    from cv_lab.scripts.pipeline.build_yolo_hand_timeline import _backfill_ledger_amounts

    positions = {1: "UTG+1", 5: "BTN", 6: "SB", 7: "BB"}
    actions = [
        {"street": "preflop", "action_type": "raise", "seat": 1, "amount": None,
         "derivation": "action_pill"},
        {"street": "preflop", "action_type": "call", "seat": 6, "amount": 2.5,
         "derivation": "stack_delta"},
        {"street": "preflop", "action_type": "call", "seat": 5, "amount": 5.0,
         "derivation": "stack_delta"},
    ]
    filled = _backfill_ledger_amounts(actions, positions)
    assert filled[0]["amount"] is None


def test_a_hand_with_an_unmeasured_transition_is_not_complete():
    """`complete` required a pot, actions and a resolved outcome, and a hand can
    have all three while a money movement inside it went unmeasured. PLAN.md:
    "Mark an incomplete sequence non-authoritative".

    Refuse BOTH the facing bet and the call so ledger_infer cannot size the hole
    from a known facing level -- the remaining unmeasured money must keep the
    hand incomplete.
    """
    hero = rd.SEAT_ANCHORS_BY_CLASS["stack_text"][0]
    villain = rd.SEAT_ANCHORS_BY_CLASS["stack_text"][4]
    clean = build_hand_timeline(rd.frames_from_fixture(_hand_fixture()))["hands"][0]
    assert clean["complete"] is True and clean["unmeasured_transitions"] == 0

    fixture = _refuse(_hand_fixture(), "stack_text", villain, "run_clipped", {5.0})
    fixture = _refuse(fixture, "stack_text", hero, "run_clipped", {4.0})
    hand = build_hand_timeline(rd.frames_from_fixture(fixture))["hands"][0]
    assert hand["unmeasured_transitions"] >= 1
    assert hand["complete"] is False
    assert hand["pot"] is not None and hand["actions"], (
        "the hand is complete by every OLD criterion, which is the point")


def test_a_refusal_on_an_idle_seat_is_a_soft_signal_not_a_ledger_hole():
    """The counterweight, stated so the rule is not read as "any refusal rejects
    the hand". An unmeasured transition means money is KNOWN to have moved and
    its size is unmeasurable. A refusal on a seat with no pill and nothing on the
    felt costs the hand only the read itself, which `amounts_unknown` already
    carries as a confidence cap. Conflating the two lets one unreadable crop on
    an idle seat reject a sound hand."""
    villain = rd.SEAT_ANCHORS_BY_CLASS["stack_text"][4]
    fixture = _refuse(_hand_fixture(), "stack_text", villain, "run_clipped", {7.0})
    hand = build_hand_timeline(rd.frames_from_fixture(fixture))["hands"][0]
    assert hand["amounts_unknown"] >= 1
    assert hand["amounts_unknown_by_code"].get("run_clipped") >= 1
    assert hand["unmeasured_transitions"] == 0
    assert "amounts_unknown_in_ledger" not in hand["warnings"]


def test_an_estimator_with_an_unknown_input_abstains_from_the_pot_consensus():
    """`contrib_pot` sums stack differences across the settled window, so a
    refused read inside it makes the sum wrong by an unmeasured amount while
    looking exactly like a clean one -- and it would then be one of the two
    "independent estimates that agree" that `reconciled` means. An estimator with
    an unknown input is REMOVED from the candidate list, not down-weighted."""
    from cv_lab.scripts.pipeline.build_yolo_hand_timeline import reconstruct

    states, _events = _states_from(_hand_fixture())
    clean = reconstruct(states, 1)
    assert clean["reconciled"] is True and clean["pot"] == 20

    # One refused read on the seat that swept the pot removes BOTH stack-derived
    # estimates: `contrib` sums a holed series, and `win` differences one. Only
    # the pot text survives, and one estimate is not a consensus.
    holed, _events = _states_from(_hand_fixture())
    for state in holed:
        if 0 in state["stacks"]:
            del state["stacks"][0]
            state["stacks_unknown"] = {0: "unexplained_ink_in_numeral"}
            break
    hand = reconstruct(holed, 1)
    assert hand["reconciled"] is False, "a holed estimate must not vote"
    assert "pot_not_reconciled" in hand["warnings"]
    assert hand["pot"] == 20, "the pot TEXT is untouched evidence and still stands"

    # ...and a refusal on a seat that is NOT the winner removes only `contrib`,
    # so text and win still corroborate each other and the hand reconciles. The
    # rule removes the estimator, not the hand.
    partial, _events = _states_from(_hand_fixture())
    for state in partial:
        if 4 in state["stacks"]:
            del state["stacks"][4]
            state["stacks_unknown"] = {4: "unexplained_ink_in_numeral"}
            break
    assert reconstruct(partial, 1)["reconciled"] is True


def test_a_refused_hero_stack_makes_the_hero_net_unknown_not_wrong():
    """`series[0]` is the first SURVIVING reading, so a refusal before it moves the
    baseline past chips the hero had already committed -- the defect note 12 fixed
    for `starting_stack`, on the field a reviewer reads as the bottom line."""
    from cv_lab.scripts.pipeline.build_yolo_hand_timeline import reconstruct

    holed, _events = _states_from(_hand_fixture())
    for state in holed[:2]:
        state["stacks"].pop(0, None)
        state["stacks_unknown"] = {0: "below_calibrated_render_size"}
    hand = reconstruct(holed, 1)
    assert hand["hero_bb_won"] is None
    assert next(p for p in hand["players"] if p["seat"] == 0)["starting_stack"] is None
    assert next(p for p in hand["players"] if p["seat"] == 0)[
        "starting_stack_unknown"] == "below_calibrated_render_size"


def test_a_refused_pot_is_not_substituted_from_an_earlier_state():
    """`cur["pot"] if not None else (pot_so_far or 0.0)` handed the settlement
    scan a STALE pot carried from an earlier state whenever this state's pot was
    unread -- a backfill from a neighbouring state, which is the one thing an
    UNKNOWN must never receive."""
    from cv_lab.scripts.pipeline.build_yolo_hand_timeline import _settle_index

    states, _events = _states_from(_hand_fixture())
    assert _settle_index(states, {0, 4}) == len(states) - 1
    for state in states:
        state["pot"] = None
        state["pot_unknown"] = "pot_zero_impossible"
    assert all(not s.get("settle_scan_skipped") for s in states)
    _settle_index(states, {0, 4})
    assert any(s.get("settle_scan_skipped") for s in states), (
        "every transition is unmeasurable, and each one must say so")


def test_a_refused_bet_is_not_carried_forward_as_a_high_water_mark():
    """The per-street high-water bet_text exists to stop a re-displayed raise
    reading as a fresh one across a dropout. Carrying it ACROSS a refusal is a
    different thing: the mark stands in for a number the pixels declined to
    supply, so the base is unknown and the delta is unmeasurable rather than
    zero-based."""
    fixture = _hand_fixture()
    hero_bet = (0.467, 0.579)   # bet_text seat 0 anchor
    for frame in fixture:
        if frame["time_s"] in {4.0, 5.0}:
            frame["detections"].append(
                _det("bet_text", *hero_bet, 7.0))
    with_bet = build_hand_timeline(rd.frames_from_fixture(fixture))["hands"][0]
    assert with_bet["unmeasured_transitions"] == 0

    for frame in fixture:
        if frame["time_s"] != 5.0:
            continue
        for det in frame["detections"]:
            if det["cls"] == "bet_text":
                det["attr"] = None
                det["attr_source"] = "suffix_not_bb"
    hero = rd.SEAT_ANCHORS_BY_CLASS["stack_text"][0]
    _refuse(fixture, "stack_text", hero, "suffix_not_bb", {6.0})
    hand = build_hand_timeline(rd.frames_from_fixture(fixture))["hands"][0]
    hero_money = [(a["street"], a["action_type"], a["amount"]) for a in hand["actions"]
                  if a["seat"] == 0 and a["action_type"] in {"bet", "raise", "call"}]
    assert len(hero_money) == len(set(hero_money)), (
        f"a bet re-displayed over a refusal was booked twice: {hero_money}")


# --------------------------------------------------------------------------- #
# Round-1 repair: the hand's FIRST state and the money already standing on it.
#
# The client pre-debits, so a hand entering view mid-preflop shows stacks that
# already exclude the standing bets. Two defects lived here (adversary A #7 and
# adversary B #1): `starting_stack` published the raw pre-debited read while the
# ledger booked the standing money as an action (31 understated player rows, one
# arithmetically impossible all-in), and a REFUSED first-state bet fell through
# to a forward scan that published a later state's number as the action amount
# with every unknown channel reading clean.
# --------------------------------------------------------------------------- #
def _standing_bet_fixture(first_bet):
    """A hand that enters view mid-preflop with the hero's raise of 4.0 already
    standing on the felt (stack pre-debited: 96 of a 100 stack). `first_bet`
    controls the FIRST state's bet_text read: ("value", 4.0) renders it,
    ("unknown", code) is a named refusal, ("absent", None) drops the box (the
    real flicker: absent on the very first state, present on the next ones)."""
    hero_bet = (0.467, 0.579)          # bet_text seat 0 anchor
    frames = []
    f0 = _frame(0.0, board=[], s0=96, s4=100, pot=1, pill0="raise", active=4)
    if first_bet[0] == "value":
        f0["detections"].append(_det("bet_text", *hero_bet, first_bet[1]))
    elif first_bet[0] == "unknown":
        det = _det("bet_text", *hero_bet, None)
        det["attr_source"] = first_bet[1]
        f0["detections"].append(det)
    frames.append(f0)
    f1 = _frame(1.0, board=[], s0=96, s4=100, pot=1, pill0="raise", active=4)
    f1["detections"].append(_det("bet_text", *hero_bet, 4.0))
    frames.append(f1)
    frames += [
        _frame(2.0, board=[], s0=96, s4=96, pot=9, pill0="raise", pill4="call",
               active=0),                                    # villain calls 4
        _frame(3.0, board=FLOP, s0=96, s4=96, pot=9, active=0),
        _frame(4.0, board=FLOP, s0=96, s4=96, pot=9, pill0="check", pill4="check",
               active=0),
        _frame(5.0, board=TURN, s0=96, s4=96, pot=9, pill0="check", pill4="check",
               active=0),
        _frame(6.0, board=RIVER, s0=96, s4=96, pot=9, pill0="check", pill4="check",
               active=0),
        _frame(7.0, board=RIVER, s0=96, s4=105, pot=9, active=0),  # villain sweeps
    ]
    return frames


def test_starting_stack_includes_the_money_already_on_the_felt():
    """`starting_stack` = the first observed read PLUS the chips standing in
    front of the seat -- the raw pre-debited read understated 31 player rows by
    exactly the seat's first-state bet, and on one all-in seat published a
    starting stack its own ledger overdraws (97.5 starting, 99.5 contributed, at
    confidence 1.0 with warnings=[])."""
    hand = build_hand_timeline(
        rd.frames_from_fixture(_standing_bet_fixture(("value", 4.0))))["hands"][0]
    by_seat = {p["seat"]: p for p in hand["players"]}
    assert by_seat[0]["starting_stack"] == 100.0, (
        "the standing 4.0 left the displayed stack before the first sample")
    assert by_seat[4]["starting_stack"] == 100.0
    assert ("raise", 4.0) in {(a["action_type"], a["amount"]) for a in hand["actions"]}
    # Chip conservation now holds: nothing the seat contributed exceeds its
    # published starting stack.
    contributed = sum(a["amount"] or 0.0 for a in hand["actions"] if a["seat"] == 0)
    assert contributed <= by_seat[0]["starting_stack"]


def test_bet_text_flicker_on_the_first_state_is_still_filled_from_the_window():
    """Negative control, unchanged behaviour: an ABSENT first-state bet box (the
    box flickers out for one sample) is not a refusal, and the standing bet is
    proven by any read inside the constant-stack window."""
    hand = build_hand_timeline(
        rd.frames_from_fixture(_standing_bet_fixture(("absent", None))))["hands"][0]
    assert ("raise", 4.0) in {(a["action_type"], a["amount"]) for a in hand["actions"]}
    assert hand["unknown_money_actions"] == 0
    assert "amounts_unknown_in_ledger" not in hand["warnings"]


def test_a_refused_first_state_bet_is_not_ocr_backfilled_from_later_frame():
    """THE B1 REGRESSION (updated channel split).

    A refused first-state bet must NOT be filled from a later OCR read of the
    same box via ``_committed_at_start``. That silent OCR backfill is what B1
    killed. Independently, ledger inference MAY size the raise from a later
    measured call (stack delta) -- that is poker arithmetic, not crop backfill,
    and is marked ``ledger_infer``.
    """
    hand = build_hand_timeline(
        rd.frames_from_fixture(_standing_bet_fixture(("unknown", "suffix_not_bb"))))["hands"][0]
    hero_raises = [a for a in hand["actions"]
                   if a["seat"] == 0 and a["action_type"] == "raise"]
    assert hero_raises, hero_raises
    assert all("ledger_infer" in str(a.get("derivation")) for a in hero_raises), hero_raises
    assert all(a["amount"] is not None for a in hero_raises), hero_raises
    # Must not claim a bare action_pill amount with no infer mark (OCR backfill).
    assert not any(
        a.get("derivation") == "action_pill" and a.get("amount") is not None
        for a in hero_raises
    )
    by_seat = {p["seat"]: p for p in hand["players"]}
    assert by_seat[0]["starting_stack"] == 100.0, (
        "standing chips resolve from the constancy window and/or ledger infer")


def test_committed_scan_stops_at_a_refused_stack_read():
    """The constancy proof RESTS on the run of equal stack reads, and a refused
    stack read breaks it: scanning past one booked a raise at 24.0 whose true
    first-state size was 4.0 (the debit hid inside the refusal). Nothing read
    after the break is in evidence."""
    hero_stack = rd.SEAT_ANCHORS_BY_CLASS["stack_text"][0]
    frames = _standing_bet_fixture(("absent", None))
    # t=1: the hero's STACK read is refused while the bet shows 24.0 -- the shape
    # a raise-plus-occlusion produces; t=2 confirms the debit (96 -> 76).
    for det in frames[1]["detections"]:
        cx = (det["xyxy"][0] + det["xyxy"][2]) / 2.0 / W
        cy = (det["xyxy"][1] + det["xyxy"][3]) / 2.0 / H
        if det["cls"] == "stack_text" and abs(cx - hero_stack[0]) < 0.02 \
                and abs(cy - hero_stack[1]) < 0.02:
            det["attr"] = None
            det["attr_source"] = "unexplained_ink_in_numeral"
        if det["cls"] == "bet_text":
            det["attr"] = 24.0
    for frame in frames[2:]:
        for det in frame["detections"]:
            cx = (det["xyxy"][0] + det["xyxy"][2]) / 2.0 / W
            cy = (det["xyxy"][1] + det["xyxy"][3]) / 2.0 / H
            if det["cls"] == "stack_text" and abs(cx - hero_stack[0]) < 0.02 \
                    and abs(cy - hero_stack[1]) < 0.02 and det["attr"] is not None:
                det["attr"] = det["attr"] - 20.0          # the debit landed
    hand = build_hand_timeline(rd.frames_from_fixture(frames))["hands"][0]
    assert not any(a["amount"] == 24.0 and a["derivation"] == "action_pill"
                   for a in hand["actions"]), (
        "a bet read past a refused stack was published as the pre-observed amount")


def test_reconciled_abstains_when_the_ledger_carries_an_unknown():
    """THE B2 REGRESSION. `contrib_holed` was set only from stack-series holes,
    so an unknown MONEY ACTION (amount None, no stack hole anywhere) left the
    contribution estimator voting in the pot consensus with a null entry in the
    very ledger it sums -- 6 of the 10 development hands carrying a ledger
    unknown published `reconciled: true`. The estimator now starts holed whenever
    amounts_unknown_in_ledger holds."""
    hero_bet = (0.467, 0.579)
    frames = [
        _frame(0.0, board=[], s0=96, s4=100, pot=5, pill0="raise", active=4),
        _frame(1.0, board=[], s0=96, s4=100, pot=5, pill0="raise", active=4),
        _frame(2.0, board=[], s0=96, s4=100, pot=5, pill0="raise", pill4="fold",
               active=0),
        _frame(3.0, board=[], s0=96, s4=100, pot=5, active=0),
    ]
    for frame in frames[:2]:
        det = _det("bet_text", *hero_bet, None)
        det["attr_source"] = "suffix_not_bb"
        frame["detections"].append(det)
    hand = build_hand_timeline(rd.frames_from_fixture(frames))["hands"][0]
    # The ledger carries an unknown-size raise; text (5.0) and contrib
    # (initial 5.0 + no observed drops = 5.0) would otherwise be two agreeing
    # votes and the hand would claim reconciliation over a hole.
    assert hand["unknown_money_actions"] >= 1
    assert hand["reconciled"] is False
    assert "pot_not_reconciled" in hand["warnings"]
    assert "amounts_unknown_in_ledger" in hand["warnings"]


def test_a_refused_starting_stack_raises_the_fatal_code():
    """THE B4 REGRESSION (spine half; the export half lives in
    test_yolo_card_app_export). A refused starting stack means accounting can
    never become authoritative for the hand, and the fact used to travel only as
    unread evidence extra."""
    seat4 = rd.SEAT_ANCHORS_BY_CLASS["stack_text"][4]
    fixture = _refuse(_hand_fixture(), "stack_text", seat4, "no_digit_run", {0.0})
    hand = build_hand_timeline(rd.frames_from_fixture(fixture))["hands"][0]
    by_seat = {p["seat"]: p for p in hand["players"]}
    assert by_seat[4]["starting_stack"] is None
    assert by_seat[4]["starting_stack_unknown"] == "no_digit_run"
    assert "starting_stack_unknown" in hand["warnings"]
    report = validate_timeline(build_hand_timeline(rd.frames_from_fixture(fixture)))
    codes = {w.get("code") for h in report["hands"] for w in h.get("warnings", [])}
    assert "starting_stack_unknown" in codes, (
        "the validator must mirror the fatal code or the export gate never sees it")


def test_a_refused_side_pot_read_is_still_an_unsupported_side_pot():
    """THE B7 REGRESSION. A refused side-pot read is a DETECTED second pot whose
    amount is unknown -- not the same fact as 'no side pot' -- and
    side_pot_unsupported (SPINE_FATAL) must fire on it. `side_pot_unknown` was
    plumbed end-to-end and consumed by nothing, so a refusal on the side band
    made the hand indistinguishable from a single-pot hand."""
    from cv_lab.scripts.pipeline.build_yolo_hand_timeline import reconstruct

    states, _events = _states_from(_hand_fixture())
    states[3]["side_pot_unknown"] = "separator_unreconciled"
    hand = reconstruct(states, 1)
    assert "side_pot_unsupported" in hand["warnings"]


def test_amounts_unknown_count_matches_its_own_breakdown():
    """THE B8 REGRESSION. `_bump` added pot_zero_impossible / bet_zero_impossible
    to the by-code map while `amounts_unknown` was taken straight from the view,
    so the exporter's 'per-reason breakdown of the count above' could sum PAST
    the count (cwpt01 hand 6: count 7, breakdown 8) and a pot dropout never
    reached the unread-amount confidence cap."""
    fixture = _hand_fixture()
    for det in fixture[4]["detections"]:
        if det["cls"] == "pot_text":
            det["attr"] = 0.0                     # a dropout: never a real pot
    seat4 = rd.SEAT_ANCHORS_BY_CLASS["stack_text"][4]
    _refuse(fixture, "stack_text", seat4, "run_clipped", {7.0})
    hand = build_hand_timeline(rd.frames_from_fixture(fixture))["hands"][0]
    assert hand["amounts_unknown_by_code"].get("pot_zero_impossible", 0) >= 1
    assert hand["amounts_unknown"] == sum(hand["amounts_unknown_by_code"].values()), (
        f'{hand["amounts_unknown"]} != {hand["amounts_unknown_by_code"]}')


# --------------------------------------------------------------------------- #
# Round-2 regressions: the pre-observed scan's pill-less arm, and the
# committed-at-start constancy window's own honesty.
# --------------------------------------------------------------------------- #
def _pill_less_standing_call_fixture(arm):
    """A hand entering view mid-preflop with a VILLAIN call of 12.0 already
    standing on the felt (stack pre-debited 100 -> 88) and the pill long
    expired -- pills flash for under a second, so the pill-less path is the
    common shape, not the corner. `arm` controls the FIRST state's bet_text for
    seat 4: "value" renders 12.0, "refused" is a named refusal, "absent" drops
    the box (it reappears reading 12.0 on the next state either way)."""
    v_bet = rd.SEAT_ANCHORS_BY_CLASS["bet_text"][4]

    def vbet(val, refused=False):
        det = _det("bet_text", *v_bet, val)
        if refused:
            det["attr_source"] = "suffix_not_bb"
        return det

    f0 = _frame(0.0, board=[], s0=88, s4=88, pot=13, active=4)
    if arm == "value":
        f0["detections"].append(vbet(12.0))
    elif arm == "refused":
        f0["detections"].append(vbet(None, refused=True))
    f1 = _frame(1.0, board=[], s0=88, s4=88, pot=13, active=4)
    f1["detections"].append(vbet(12.0))
    return [
        f0, f1,
        _frame(2.0, board=FLOP, s0=88, s4=88, pot=25, active=0),
        _frame(3.0, board=FLOP, s0=81, s4=88, pot=32, pill0="bet", active=4),
        _frame(4.0, board=FLOP, s0=81, s4=81, pot=39, pill0="bet", pill4="call", active=0),
        _frame(5.0, board=TURN, s0=81, s4=81, pot=39, pill0="check", pill4="check", active=0),
        _frame(6.0, board=RIVER, s0=81, s4=81, pot=39, pill0="check", pill4="check", active=0),
        _frame(7.0, board=RIVER, s0=120, s4=81, pot=39, active=0),
    ]


def test_a_pill_less_standing_call_flickered_off_the_first_state_is_booked():
    """THE ROUND-2 B1 REGRESSION, absent arm. The pre-observed scan iterated
    `set(first["pills"]) | set(first["bets"])`, so a seat whose bet box
    flickered off the very first state -- with its pill long expired -- was
    never iterated at all: no action, no unknown, every channel clean, while
    the constancy proof was simultaneously publishing starting_stack
    100.0 = 88 + 12 off the same standing bet it left out of the ledger. The
    window anchored at the first state's own stack read proves the standing
    amount, and the call is booked."""
    hand = build_hand_timeline(
        rd.frames_from_fixture(_pill_less_standing_call_fixture("absent")))["hands"][0]
    calls = [a for a in hand["actions"]
             if a["seat"] == 4 and a["street"] == "preflop" and a["action_type"] == "call"]
    assert [a["amount"] for a in calls] == [12.0], hand["actions"]
    by_seat = {p["seat"]: p for p in hand["players"]}
    assert by_seat[4]["starting_stack"] == 100.0
    assert hand["unmeasured_transitions"] == 0
    # control: the value arm books the same action from the first state itself
    control = build_hand_timeline(
        rd.frames_from_fixture(_pill_less_standing_call_fixture("value")))["hands"][0]
    assert [a["amount"] for a in control["actions"]
            if a["seat"] == 4 and a["action_type"] == "call" and a["street"] == "preflop"] == [12.0]


def test_a_pill_less_seat_with_a_refused_first_state_bet_is_unknown_money():
    """THE ROUND-2 B1 REGRESSION, refused arm. Same fixture, but the first
    state's read is a NAMED REFUSAL. A refusal is never backfilled from a later
    state (the rule the pill branch already follows), and a refused box cannot
    even prove an action happened -- a blind post produces the same box -- so no
    action may be fabricated. But money of unknown size is standing on the felt:
    dropping it silently exported a hand whose pot contained 12 BB no action
    accounts for, with unknown_money_actions=0, unmeasured_transitions=0,
    warnings=[] and reconciled=True. It is an unmeasured transition, it makes
    the ledger visibly unknown, and the hand does not export as complete."""
    hand = build_hand_timeline(
        rd.frames_from_fixture(_pill_less_standing_call_fixture("refused")))["hands"][0]
    assert not [a for a in hand["actions"] if a["seat"] == 4 and a["street"] == "preflop"], (
        "no action may be fabricated from a refused read")
    assert hand["unmeasured_transitions"] >= 1
    assert "amounts_unknown_in_ledger" in hand["warnings"]
    assert hand["complete"] is False
    # the starting stack itself is still proven by the t=1 read under a
    # constant stack -- the refusal blocks the LEDGER, not the measurement
    by_seat = {p["seat"]: p for p in hand["players"]}
    assert by_seat[4]["starting_stack"] == 100.0


def test_disagreeing_bet_reads_inside_a_constant_stack_window_refuse():
    """A constant stack PROVES the standing bet cannot have changed, so two
    reads of DIFFERENT values inside one committed-at-start window falsify the
    premise the window rests on -- one of the boxes is not this seat's standing
    bet (anchor misattribution; live on g0621 hand 2, where a folded seat whose
    stack never moved read 3.0 preflop and 15.0 on the turn, published
    starting_stack 115.7 for a seat whose true committed chips were 0, and
    would have booked a phantom 'call 15.0'). Taking the max was a choice among
    contradictory evidence; the window now returns UNKNOWN."""
    from cv_lab.scripts.pipeline.build_yolo_hand_timeline import _committed_at_start

    def state(stack, bet):
        return {"stacks": {4: stack} if stack is not None else {},
                "stacks_unknown": {}, "bets": {4: bet} if bet is not None else {},
                "bets_unknown": {}}

    agreeing = [state(100.7, 3.0), state(100.7, None), state(100.7, 3.0)]
    assert _committed_at_start(agreeing, 4) == 3.0
    disagreeing = [state(100.7, 3.0), state(100.7, None), state(100.7, 15.0)]
    assert _committed_at_start(disagreeing, 4) is None, (
        "disagreeing reads under a constant stack are contradictory evidence, "
        "not a population to take the max of")
    # end to end: the phantom call must not reach the ledger and the hand must
    # say why
    frames = _pill_less_standing_call_fixture("absent")
    v_bet = rd.SEAT_ANCHORS_BY_CLASS["bet_text"][4]
    f2 = frames[2]
    det = _det("bet_text", *v_bet, 27.0)      # misattributed box, stack unmoved
    f2["detections"].append(det)
    hand = build_hand_timeline(rd.frames_from_fixture(frames))["hands"][0]
    assert not [a for a in hand["actions"]
                if a["seat"] == 4 and a["amount"] in (12.0, 27.0)], hand["actions"]
    assert hand["unmeasured_transitions"] >= 1
    assert "amounts_unknown_in_ledger" in hand["warnings"]


def test_an_all_refused_committed_window_is_unknown_not_zero():
    """THE ROUND-2 C2 KILL. `return None if saw_refusal else 0.0` survived a
    mutation to `return 0.0` because the only pinning fixture contained a later
    PROVEN read. Here EVERY bet read inside the constant-stack window is a
    named refusal: the OCR standing amount is UNKNOWN. The raw helper must not
    publish 0.0/96.0. End-to-end, a later sized call may still ledger-infer the
    open; without that call the starting stack stays unknown."""
    from cv_lab.scripts.pipeline.build_yolo_hand_timeline import (
        _committed_at_start,
        _observed_starting_stack,
        _observed_starting_stack_unknown,
    )

    def state(stack, bet_refused):
        s = {"stacks": {0: stack}, "stacks_unknown": {},
             "bets": {}, "bets_unknown": {}}
        if bet_refused:
            s["bets_unknown"] = {0: "suffix_not_bb"}
        return s

    window = [state(96.0, True), state(96.0, True), state(96.0, True)]
    assert _committed_at_start(window, 0) is None, (
        "a refusal answered by no read is UNKNOWN, never 0.0")
    assert _observed_starting_stack(window, 0) is None
    assert _observed_starting_stack_unknown(window, 0) == "committed_at_start_unknown"
    # end to end with the later call removed: nothing pins the open.
    frames = _standing_bet_fixture(("unknown", "suffix_not_bb"))
    for det in frames[1]["detections"]:
        if det["cls"] == "bet_text":
            det["attr"] = None
            det["attr_source"] = "suffix_not_bb"
    # Keep villain at 100 forever and strip call pills so ledger_infer cannot pin.
    v_stack = rd.SEAT_ANCHORS_BY_CLASS["stack_text"][4]
    for frame in frames:
        frame["detections"] = [
            d for d in frame["detections"]
            if not (d["cls"] == "action_pill" and d.get("attr") == "call")
        ]
        for det in frame["detections"]:
            if det["cls"] != "stack_text":
                continue
            cx = (det["xyxy"][0] + det["xyxy"][2]) / 2.0 / W
            cy = (det["xyxy"][1] + det["xyxy"][3]) / 2.0 / H
            if abs(cx - v_stack[0]) < 0.02 and abs(cy - v_stack[1]) < 0.02:
                det["attr"] = 100
    hand = build_hand_timeline(rd.frames_from_fixture(frames))["hands"][0]
    by_seat = {p["seat"]: p for p in hand["players"]}
    assert by_seat[0]["starting_stack"] is None
    assert by_seat[0]["starting_stack_unknown"] == "committed_at_start_unknown"
    assert "starting_stack_unknown" in hand["warnings"]


def test_committed_scan_past_a_refused_stack_read_cannot_book_the_later_bet():
    """THE ROUND-2 C3 KILL. The round-1 committed-scan-stop test could not kill
    a scan-past mutant: its refusal state carried no stack read, so the mutant
    `continue`d before reading the bet, and the next state's moved stack ended
    the window anyway. Here the state AFTER the refusal re-reads the START
    value with 24.0 still displayed -- the exact shape where scanning past the
    refusal books 24.0 (or, under the disagreement rule, destroys the proven
    4.0). The stop is what keeps the answer 4.0."""
    from cv_lab.scripts.pipeline.build_yolo_hand_timeline import _committed_at_start

    def state(stack, refused, bet):
        s = {"stacks": {}, "stacks_unknown": {}, "bets": {}, "bets_unknown": {}}
        if stack is not None:
            s["stacks"] = {0: stack}
        if refused:
            s["stacks_unknown"] = {0: "run_clipped"}
        if bet is not None:
            s["bets"] = {0: bet}
        return s

    hand = [
        state(96.0, False, 4.0),    # the true standing raise of 4.0
        state(None, True, 24.0),    # stack REFUSED; felt shows 24.0 (occlusion)
        state(96.0, False, 24.0),   # stack re-reads the start value
        state(76.0, False, None),   # the debit lands
    ]
    assert _committed_at_start(hand, 0) == 4.0, (
        "nothing read past a refused stack read is in evidence")


def test_contrib_estimator_abstains_when_the_first_pot_read_was_refused():
    """THE ROUND-2 B2 REGRESSION. `pots` silently skips refused states, so with
    the hand's first pot read REFUSED, `initial_pot = pots[0]` re-based to a
    later reading that already contained the chips committed in between --
    double-counting them. On the uncalled-refund shape (winner's stack gain
    exceeds the true pot, measured for real on g0723a hands 2 and 5) the
    inflated contrib then AGREED with the inflated win estimate and outvoted
    the correct pot text 2:1: published pot 33.0 for a true 23.0 (43% high),
    reconciled=True, warnings=[], exported. A refused pot read at or before
    the first surviving one is an unknown input, and an estimator with an
    unknown input is removed from the candidate list."""
    def fixture(refused):
        def pot_det(val, refuse):
            det = _det("pot_text", 0.50, 0.32, None if refuse else val)
            if refuse:
                det["attr_source"] = "unexplained_ink_in_numeral"
            return det
        frames = [
            _frame(0.0, board=[], s0=100, s4=100, active=0),
            _frame(1.0, board=[], s0=90, s4=100, pill0="raise", active=4),   # hero raises 10
            _frame(2.0, board=[], s0=90, s4=90, pot=23.0, pill0="raise", pill4="call", active=0),
            _frame(3.0, board=FLOP, s0=90, s4=90, pot=23.0, active=0),
            _frame(4.0, board=TURN, s0=90, s4=90, pot=23.0, pill0="check", pill4="check", active=0),
            _frame(5.0, board=RIVER, s0=90, s4=90, pot=23.0, pill0="check", pill4="check", active=0),
            # hero sweeps: gain 33 on a true 23 pot (the uncalled-refund shape)
            _frame(6.0, board=RIVER, s0=123, s4=90, active=0),
        ]
        frames[0]["detections"].append(pot_det(3.0, refuse=refused))
        frames[1]["detections"].append(pot_det(13.0, refuse=False))
        return frames

    value = build_hand_timeline(rd.frames_from_fixture(fixture(False)))["hands"][0]
    assert value["pot"] == 23.0, "control arm: text 23 + contrib 3+20 agree"
    assert value["reconciled"] is True

    refused = build_hand_timeline(rd.frames_from_fixture(fixture(True)))["hands"][0]
    assert refused["pot"] != 33.0, (
        "a re-based contrib estimate corroborated the inflated win estimate and "
        "outvoted the correct pot text")
    assert refused["pot"] == 23.0
    assert refused["reconciled"] is False, (
        "one voter cannot reconcile: the refusal must be visible, not papered over")
    assert "pot_not_reconciled" in refused["warnings"]


def test_brief_nontable_gap_without_state_change_still_completes():
    """A tab flash that leaves board/stacks unchanged must stitch, not reject."""
    fixture = _hand_fixture()
    # Insert an explicit nontable sample between two identical-ledger states.
    nontable = {
        "image": "overlay.jpg",
        "time_s": 4.5,
        "width": W,
        "height": H,
        "screen": "nontable",
        "detections": [],
    }
    # Place between the flop bet (t=4) and the call (t=5) without changing money.
    fixture.insert(5, nontable)
    hand = build_hand_timeline(rd.frames_from_fixture(fixture))["hands"][0]
    assert hand["coverage_gaps"] == 0
    assert "mid_hand_coverage_gap" not in hand["warnings"]
    assert hand["complete"] is True


def test_mid_hand_coverage_gap_does_not_invent_stack_delta_actions():
    """Tab covers while a raise happens: recover either side, do not invent size."""
    fixture = [
        _frame(0.0, board=[], s0=100, s4=100, active=4),
        _frame(1.0, board=[], s0=100, s4=97, pot=3, pill4="raise", active=0),
        _frame(2.0, board=[], s0=97, s4=97, pot=6, pill0="call", pill4="raise", active=4),
        # Nontable stretch while villain bets the flop off-camera.
        {
            "image": "tab.jpg",
            "time_s": 3.0,
            "width": W,
            "height": H,
            "screen": "nontable",
            "detections": [],
        },
        {
            "image": "tab2.jpg",
            "time_s": 4.0,
            "width": W,
            "height": H,
            "screen": "nontable",
            "detections": [],
        },
        {
            "image": "tab3.jpg",
            "time_s": 5.0,
            "width": W,
            "height": H,
            "screen": "nontable",
            "detections": [],
        },
        # Table returns: flop is out and villain already put 10 more in.
        _frame(6.0, board=FLOP, s0=97, s4=87, pot=16, pill4="bet", active=0),
        _frame(7.0, board=FLOP, s0=87, s4=87, pot=26, pill0="call", pill4="bet", active=4),
        _frame(8.0, board=TURN, s0=87, s4=87, pot=26, pill0="check", pill4="check", active=0),
        _frame(9.0, board=RIVER, s0=87, s4=87, pot=26, pill0="check", pill4="check", active=0),
        _frame(10.0, board=RIVER, s0=113, s4=87, pot=26, active=0),
    ]
    hand = build_hand_timeline(rd.frames_from_fixture(fixture))["hands"][0]
    assert hand["coverage_gaps"] >= 1
    assert "mid_hand_coverage_gap" in hand["warnings"]
    assert hand["complete"] is False
    # No fabricated sized action from the post-gap stack debit alone.
    invented = [
        a for a in hand["actions"]
        if a["derivation"] == "stack_delta"
        and a["street"] == "flop"
        and a["seat"] == 4
        and a.get("amount") == 10.0
    ]
    assert invented == [], f"invented flop raise across coverage gap: {invented}"
    # Visible fresh pill after the gap may record the act with unknown size.
    assert all(
        a.get("amount") is None
        for a in hand["actions"]
        if a.get("derivation") == "amount_unknown"
    )


def test_time_jump_with_board_advance_is_a_coverage_gap_without_nontable_marker():
    """Even without classify_screen markers, a long mid-hand hole with a board
    jump is unobserved coverage -- not a silently complete multi-street hand."""
    fixture = [
        _frame(0.0, board=[], s0=100, s4=100, active=4),
        _frame(1.0, board=[], s0=100, s4=97, pot=3, pill4="raise", active=0),
        _frame(2.0, board=[], s0=97, s4=97, pot=6, pill0="call", pill4="raise", active=4),
        # 8s hole (well above 2.5s threshold at 1s sampling) then flop appears.
        _frame(10.0, board=FLOP, s0=97, s4=97, pot=6, active=0),
        _frame(11.0, board=FLOP, s0=90, s4=97, pot=13, pill0="bet", active=4),
        _frame(12.0, board=FLOP, s0=90, s4=90, pot=20, pill0="bet", pill4="call", active=0),
        _frame(13.0, board=TURN, s0=90, s4=90, pot=20, pill0="check", pill4="check", active=0),
        _frame(14.0, board=RIVER, s0=90, s4=90, pot=20, pill0="check", pill4="check", active=0),
        _frame(15.0, board=RIVER, s0=110, s4=90, pot=20, active=0),
    ]
    hand = build_hand_timeline(rd.frames_from_fixture(fixture))["hands"][0]
    assert hand["coverage_gaps"] >= 1
    assert "mid_hand_coverage_gap" in hand["warnings"]
    assert hand["complete"] is False


def test_hero_dim_at_open_emits_fold_and_blocks_inferred_checks():
    """Operator: hero folded long ago but dim cards kept seat 0 live."""
    from cv_lab.scripts.pipeline.build_yolo_hand_timeline import _reconstruct_actions

    def state(time_s, *, board, dealt, pills=None, bets=None, stacks=None, pot=None):
        return {
            "time_s": float(time_s),
            "image": f"{time_s}.jpg",
            "state_index": 0,
            "board_cards": list(board),
            "dealt_in": list(dealt),
            "pills": dict(pills or {}),
            "bets": dict(bets or {}),
            "stacks": dict(stacks or {}),
            "stacks_unknown": {},
            "bets_unknown": {},
            "hero_cards": ["Ah", "Kd"],
            "hero_dim": True,
            "villain_cards": {},
            "pot": pot,
            "pot_unknown": None,
            "stage": {0: "preflop", 3: "flop"}.get(len(board), "preflop"),
            "sampling_interval_s": 1.0,
            "prior_gap_s": 0.0 if time_s == 0 else 1.0,
        }

    hand = [
        state(0, board=[], dealt=[0, 1, 6, 7], pills={1: "raise", 6: "call"},
              bets={1: 3.0, 6: 3.0}, stacks={0: 200, 1: 200, 6: 200, 7: 200}, pot=7.5),
        state(4, board=[], dealt=[0, 1, 6, 7], pills={7: "raise"},
              bets={7: 13.0, 1: 3.0, 6: 3.0}, stacks={0: 200, 1: 200, 6: 200, 7: 187}, pot=22),
        state(6, board=[], dealt=[0, 1, 6, 7], pills={1: "call"},
              bets={7: 13.0, 1: 13.0, 6: 3.0}, stacks={0: 200, 1: 187, 6: 200, 7: 187}, pot=32),
        state(9, board=FLOP, dealt=[0, 1, 6, 7],
              stacks={0: 200, 1: 187, 6: 187, 7: 187}, pot=42),
    ]
    positions = {0: "UTG", 1: "UTG+1", 6: "SB", 7: "BB"}
    names = {0: "Hero", 1: "S1", 6: "SB", 7: "BB"}
    actions = _reconstruct_actions(hand, positions, names)
    assert any(
        a["seat"] == 0 and a["action_type"] == "fold" and a["derivation"] == "hero_dim"
        for a in actions
    )
    assert not any(a["seat"] == 0 and a["action_type"] == "check" for a in actions)


def test_still_in_seat_after_raise_infers_call_when_street_advances():
    """Operator: because SB is still in you should assume they called."""
    from cv_lab.scripts.pipeline.build_yolo_hand_timeline import _reconstruct_actions

    def state(time_s, *, board, dealt, pills=None, bets=None, stacks=None, pot=None):
        return {
            "time_s": float(time_s),
            "image": f"{time_s}.jpg",
            "state_index": 0,
            "board_cards": list(board),
            "dealt_in": list(dealt),
            "pills": dict(pills or {}),
            "bets": dict(bets or {}),
            "stacks": dict(stacks or {}),
            "stacks_unknown": {},
            "bets_unknown": {},
            "hero_cards": ["Ah", "Kd"],
            "hero_dim": False,
            "villain_cards": {},
            "pot": pot,
            "pot_unknown": None,
            "stage": {0: "preflop", 3: "flop"}.get(len(board), "preflop"),
            "sampling_interval_s": 1.0,
            "prior_gap_s": 0.0 if time_s == 0 else 1.0,
        }

    # No stack reads: money size unknown; still-in must still book the call.
    hand = [
        state(0, board=[], dealt=[1, 6, 7], pills={1: "raise", 6: "call"},
              bets={1: 3.0, 6: 3.0}, pot=7.5),
        state(4, board=[], dealt=[1, 6, 7], pills={7: "raise"},
              bets={7: 13.0, 1: 3.0, 6: 3.0}, pot=22),
        state(6, board=[], dealt=[1, 6, 7], pills={1: "call"},
              bets={7: 13.0, 1: 13.0, 6: 3.0}, pot=32),
        state(9, board=FLOP, dealt=[1, 6, 7], pot=42),
    ]
    actions = _reconstruct_actions(
        hand, {1: "UTG+1", 6: "SB", 7: "BB"}, {1: "S1", 6: "SB", 7: "BB"}
    )
    inferred = [
        a for a in actions
        if a["seat"] == 6 and a["derivation"] == "inferred_still_in"
    ]
    assert len(inferred) == 1
    assert inferred[0]["action_type"] == "call"
    assert inferred[0]["street"] == "preflop"


def test_still_in_call_is_not_inferred_across_coverage_gap():
    """Adversary: board advance after a long hole must not invent still-in calls."""
    from cv_lab.scripts.pipeline.build_yolo_hand_timeline import _reconstruct_actions

    def state(time_s, *, board, dealt, pills=None, bets=None, prior_gap_s=1.0, pot=None):
        return {
            "time_s": float(time_s),
            "image": f"{time_s}.jpg",
            "state_index": 0,
            "board_cards": list(board),
            "dealt_in": list(dealt),
            "pills": dict(pills or {}),
            "bets": dict(bets or {}),
            "stacks": {},
            "stacks_unknown": {},
            "bets_unknown": {},
            "hero_cards": ["Ah", "Kd"],
            "hero_dim": False,
            "villain_cards": {},
            "pot": pot,
            "pot_unknown": None,
            "stage": {0: "preflop", 3: "flop"}.get(len(board), "preflop"),
            "sampling_interval_s": 1.0,
            "prior_gap_s": float(prior_gap_s),
        }

    hand = [
        state(0, board=[], dealt=[1, 6, 7], pills={1: "raise", 6: "call"},
              bets={1: 3.0, 6: 3.0}, prior_gap_s=0.0, pot=7.5),
        state(4, board=[], dealt=[1, 6, 7], pills={7: "raise"},
              bets={7: 13.0, 1: 3.0, 6: 3.0}, pot=22),
        # Long unobserved hole then flop; SB still listed dealt-in at both ends.
        state(20, board=FLOP, dealt=[1, 6, 7], prior_gap_s=16.0, pot=42),
    ]
    actions = _reconstruct_actions(
        hand, {1: "UTG+1", 6: "SB", 7: "BB"}, {1: "S1", 6: "SB", 7: "BB"}
    )
    assert not any(a.get("derivation") == "inferred_still_in" for a in actions)


def test_still_in_does_not_call_for_all_in_seat():
    from cv_lab.scripts.pipeline.build_yolo_hand_timeline import _reconstruct_actions

    def state(time_s, *, board, dealt, pills=None, bets=None, stacks=None, pot=None):
        return {
            "time_s": float(time_s),
            "image": f"{time_s}.jpg",
            "state_index": 0,
            "board_cards": list(board),
            "dealt_in": list(dealt),
            "pills": dict(pills or {}),
            "bets": dict(bets or {}),
            "stacks": dict(stacks or {}),
            "stacks_unknown": {},
            "bets_unknown": {},
            "hero_cards": ["Ah", "Kd"],
            "hero_dim": False,
            "villain_cards": {},
            "pot": pot,
            "pot_unknown": None,
            "stage": {0: "preflop", 3: "flop"}.get(len(board), "preflop"),
            "sampling_interval_s": 1.0,
            "prior_gap_s": 0.0 if time_s == 0 else 1.0,
        }

    hand = [
        state(0, board=[], dealt=[1, 7], pills={1: "raise"}, bets={1: 50.0},
              stacks={1: 0.0, 7: 100.0}, pot=51),
        state(2, board=[], dealt=[1, 7], pills={7: "raise"}, bets={7: 100.0, 1: 50.0},
              stacks={1: 0.0, 7: 50.0}, pot=151),
        state(4, board=FLOP, dealt=[1, 7], stacks={1: 0.0, 7: 50.0}, pot=151),
    ]
    actions = _reconstruct_actions(
        hand, {1: "UTG", 7: "BB"}, {1: "S1", 7: "BB"}
    )
    assert not any(
        a["seat"] == 1 and a["derivation"] == "inferred_still_in" for a in actions
    )


def test_hero_dim_does_not_duplicate_fold_pill():
    from cv_lab.scripts.pipeline.build_yolo_hand_timeline import _reconstruct_actions

    first = {
        "time_s": 0.0,
        "image": "0.jpg",
        "state_index": 0,
        "board_cards": [],
        "dealt_in": [0, 1],
        "pills": {0: "fold", 1: "raise"},
        "bets": {1: 3.0},
        "stacks": {0: 100.0, 1: 100.0},
        "stacks_unknown": {},
        "bets_unknown": {},
        "hero_cards": ["Ah", "Kd"],
        "hero_dim": True,
        "villain_cards": {},
        "pot": 4.5,
        "pot_unknown": None,
        "stage": "preflop",
        "sampling_interval_s": 1.0,
        "prior_gap_s": 0.0,
    }
    second = {
        **first,
        "time_s": 3.0,
        "image": "3.jpg",
        "board_cards": FLOP,
        "dealt_in": [1],
        "pills": {},
        "bets": {},
        "stage": "flop",
        "prior_gap_s": 3.0,
    }
    actions = _reconstruct_actions(
        [first, second], {0: "BTN", 1: "BB"}, {0: "Hero", 1: "BB"}
    )
    hero_folds = [a for a in actions if a["seat"] == 0 and a["action_type"] == "fold"]
    assert len(hero_folds) == 1
    assert hero_folds[0]["derivation"] == "action_pill"


def _action_state(
    time_s,
    *,
    board,
    dealt,
    pills=None,
    bets=None,
    stacks=None,
    pot=None,
    prior_gap_s=None,
    villain_cards=None,
    hero_dim=False,
):
    return {
        "time_s": float(time_s),
        "image": f"{time_s}.jpg",
        "state_index": 0,
        "board_cards": list(board),
        "dealt_in": list(dealt),
        "pills": dict(pills or {}),
        "bets": dict(bets or {}),
        "stacks": dict(stacks or {}),
        "stacks_unknown": {},
        "bets_unknown": {},
        "hero_cards": ["Ah", "Kd"],
        "hero_dim": hero_dim,
        "villain_cards": dict(villain_cards or {}),
        "pot": pot,
        "pot_unknown": None,
        "stage": {0: "preflop", 3: "flop", 4: "turn", 5: "river"}.get(
            len(board), "preflop"
        ),
        "sampling_interval_s": 1.0,
        "prior_gap_s": (
            0.0 if prior_gap_s is None and time_s == 0 else float(prior_gap_s or 1.0)
        ),
    }


def test_mid_street_open_books_standing_flop_bet():
    """Operator: capture begins mid-flop with LJ bet already on the felt."""
    from cv_lab.scripts.pipeline.build_yolo_hand_timeline import _reconstruct_actions

    hand = [
        _action_state(
            0,
            board=FLOP,
            dealt=[0, 3, 5],
            pills={5: "bet", 3: "check"},
            pot=18.1,
        ),
        _action_state(
            1,
            board=FLOP,
            dealt=[0, 3, 5],
            pills={5: "bet", 3: "check"},
            bets={5: 3.6},
            pot=18.1,
        ),
        _action_state(
            3,
            board=FLOP,
            dealt=[0, 3, 5],
            pills={0: "call", 5: "bet"},
            bets={0: 3.6, 5: 3.6},
            stacks={0: 196.5, 3: 362.5, 5: 201.5},
            pot=21.7,
        ),
    ]
    actions = _reconstruct_actions(
        hand,
        {0: "BTN", 3: "UTG", 5: "LJ"},
        {0: "Hero", 3: "UTG", 5: "LJ"},
    )
    lj_bets = [
        a for a in actions
        if a["seat"] == 5 and a["action_type"] == "bet" and a["street"] == "flop"
    ]
    assert lj_bets, actions
    assert lj_bets[0]["amount"] == 3.6
    assert not any(
        a["seat"] == 5 and a["action_type"] == "check" and a["street"] == "flop"
        for a in actions
    )


def test_showdown_reveal_is_not_a_fold():
    """Operator: LJ called and showed cards; card-back gone is not a fold."""
    from cv_lab.scripts.pipeline.build_yolo_hand_timeline import _reconstruct_actions

    hand = [
        _action_state(
            74,
            board=RIVER,
            dealt=[0, 5],
            pills={0: "bet"},
            bets={0: 75.0},
            stacks={0: 167.6, 5: 176.2},
            pot=150.9,
        ),
        _action_state(
            75,
            board=RIVER,
            dealt=[0],
            stacks={0: 167.6, 5: 101.2},
            pot=225.9,
            villain_cards={5: ["Qh", "As"]},
        ),
    ]
    # Seed a prior river bet so street_has_bet is true via the first state's bet.
    hand[0]["pills"] = {0: "bet"}
    actions = _reconstruct_actions(
        hand, {0: "BTN", 5: "LJ"}, {0: "Hero", 5: "LJ"}
    )
    assert not any(
        a["seat"] == 5 and a["action_type"] == "fold" for a in actions
    ), actions


def test_coverage_gap_sizes_bet_from_readable_bet_text():
    """Operator: BB bets 20 BB; gap must not discard a clear bet_text size."""
    from cv_lab.scripts.pipeline.build_yolo_hand_timeline import _reconstruct_actions

    hand = [
        _action_state(
            13,
            board=FLOP,
            dealt=[1, 6, 7],
            pills={6: "check"},
            stacks={1: 202.2, 6: 181.0, 7: 212.2},
            pot=42.0,
            prior_gap_s=1.0,
        ),
        _action_state(
            17,
            board=FLOP,
            dealt=[1, 6, 7],
            pills={6: "check", 7: "bet"},
            bets={7: 20.0},
            stacks={1: 202.2, 6: 181.0, 7: 192.2},
            pot=62.0,
            prior_gap_s=4.0,
        ),
    ]
    actions = _reconstruct_actions(
        hand,
        {1: "UTG+1", 6: "SB", 7: "BB"},
        {1: "S1", 6: "SB", 7: "BB"},
    )
    bb_bets = [
        a for a in actions
        if a["seat"] == 7 and a["action_type"] == "bet" and a["street"] == "flop"
    ]
    assert len(bb_bets) == 1
    assert bb_bets[0]["amount"] == 20.0
    assert bb_bets[0]["derivation"] == "bet_text"


def test_green_post_pill_is_live_post_not_call():
    """Operator: UTG+1 POST is a live missed-blind post, not a call."""
    from cv_lab.scripts.pipeline import region_detections as rd
    from cv_lab.scripts.pipeline.build_yolo_hand_timeline import _reconstruct_actions

    assert rd.read_pill_action(
        type("D", (), {"attr": "post"})(), dealt_in=True
    ) == "post_blind"

    hand = [
        _action_state(
            48,
            board=[],
            dealt=[0, 1, 2, 4],
            pills={2: rd.PILL_BET_OR_CALL},
            stacks={0: 201.8, 1: 199.5, 2: 198.5, 4: 1458.4},
            pot=3.0,
        ),
        _action_state(
            49,
            board=[],
            dealt=[0, 2, 4],
            pills={1: "fold"},
            stacks={0: 201.8, 2: 198.5, 4: 1458.4},
            pot=3.0,
        ),
        _action_state(
            51,
            board=[],
            dealt=[0, 2, 4],
            pills={2: "check"},
            stacks={0: 201.8, 2: 198.5, 4: 1458.4},
            pot=3.0,
        ),
    ]
    actions = _reconstruct_actions(
        hand,
        {0: "BB", 1: "UTG", 2: "UTG+1", 4: "LJ"},
        {0: "Hero", 1: "UTG", 2: "UTG+1", 4: "LJ"},
    )
    posts = [
        a for a in actions
        if a["seat"] == 2 and a["action_type"] == "post_blind"
    ]
    assert len(posts) == 1, actions
    assert posts[0]["is_live_post"] is True
    assert posts[0]["forced_bet_type"] == "big_blind"
    assert not any(
        a["seat"] == 2 and a["action_type"] == "call" and a["street"] == "preflop"
        and a.get("source_time_s") == 48.0
        for a in actions
    )


def test_explicit_post_pill_alias_maps_to_post_blind():
    from cv_lab.scripts.pipeline import region_detections as rd

    for attr in ("post", "POST", "post_blind", "Post"):
        assert rd.read_pill_action(
            type("D", (), {"attr": attr})(), dealt_in=True
        ) == "post_blind"


def _flip_hero_suit(fixture, *, at_or_after, frm="Kd", to="Ks"):
    """Misread ONE hero hole card from ``at_or_after`` to the end of the hand."""
    for row in fixture:
        if row["time_s"] < at_or_after:
            continue
        for det in row["detections"]:
            if det["cls"] == "face_card" and det["attr"] == frm:
                det["attr"] = to
    return fixture


def test_a_hero_slot_that_never_reverts_still_settles_on_the_majority():
    """The shape the sequential debounce cannot reach.

    ``_debounce_cards`` accepts a change its NEXT reading confirms, and undoes an
    excursion only when the accepted value comes back before the sweep. A misread
    that starts partway through a hand and runs to the SHOWDOWN never comes back,
    so it was confirmed and published: measured on a 1344x836 recording, hero
    read Kd Tc for 31 states and Ts for the last 8, and the hand carried two
    different hole-card pairs. Hole cards are dealt once; that is incoherent, not
    merely uncertain.
    """
    from cv_lab.scripts.pipeline.build_yolo_hand_timeline import build_hand_timeline

    # Right from state one the reading is unanimous.
    clean = build_hand_timeline(rd.frames_from_fixture(_hand_fixture()))
    assert clean["hands"][0]["hero"] == ["As", "Kd"]

    # Now flip the second hole card for the LAST THREE frames and never back.
    timeline = build_hand_timeline(
        rd.frames_from_fixture(_flip_hero_suit(_hand_fixture(), at_or_after=6.0))
    )
    hand = timeline["hands"][0]
    assert hand["hero"] == ["As", "Kd"], "six states read Kd, three read Ks"
    heroes = {tuple(s["hero_cards"]) for s in timeline["states"] if s["hero_cards"]}
    assert heroes == {("As", "Kd")}, f"one hand, one pair of hole cards: {heroes}"


def test_the_majority_is_published_but_still_reported_as_contested():
    """Taking the majority is not a claim the majority is right.

    Measured on the 07-23 recording the majority was WRONG -- a turn card read 8d
    eight times and 8h twice with the eight of hearts unmistakably on screen --
    so a hand whose slots were contested keeps its warning and its reduced
    confidence, and the operator still gets it in the review console.
    """
    from cv_lab.scripts.pipeline.build_yolo_hand_timeline import build_hand_timeline

    timeline = build_hand_timeline(
        rd.frames_from_fixture(_flip_hero_suit(_hand_fixture(), at_or_after=6.0))
    )
    assert "hero_card_identity_split" in timeline["hands"][0]["warnings"]


def test_majority_resolution_leaves_board_growth_alone():
    """Only the identity AT a slot is rewritten; each state keeps its own length,
    so a flop does not become a river."""
    from cv_lab.scripts.pipeline.build_yolo_hand_timeline import build_hand_timeline

    timeline = build_hand_timeline(rd.frames_from_fixture(_hand_fixture()))
    lengths = [len(s["board_cards"]) for s in timeline["states"]]
    assert lengths == sorted(lengths), "the board only ever grows"
    assert {tuple(s["board_cards"]) for s in timeline["states"]} == {
        (), tuple(FLOP), tuple(TURN), tuple(RIVER)
    }
    assert timeline["hands"][0]["board"] == RIVER


def test_majority_resolution_never_takes_cards_across_a_hand_boundary():
    """One hand's majority must never reach another hand's states.

    The first attempt at this segmented on "a maximal run of non-empty card
    readings", reasoning that cards leaving the table end the hand. They do not
    always -- the sweep can fall between samples or be collapsed away -- so three
    deals were treated as one, a hand inherited the hole cards of a hand two
    earlier, and that pair then duplicated a card already on its board. A deal
    boundary has exactly one definition in the spine (``_segment``), and this
    resolution rides on it rather than inventing a second.
    """
    from cv_lab.scripts.pipeline.build_yolo_hand_timeline import build_hand_timeline

    first = _hand_fixture()
    # A second deal, with different hole cards, and NO all-empty-board gap in
    # between beyond the one the fixture already carries.
    second = []
    for row in _hand_fixture():
        row = json.loads(json.dumps(row))
        row["time_s"] += 20.0
        row["image"] = f"f{row['time_s']}.jpg"
        for det in row["detections"]:
            if det["cls"] == "face_card" and det["attr"] == "As":
                det["attr"] = "Qh"
            elif det["cls"] == "face_card" and det["attr"] == "Kd":
                det["attr"] = "Qs"
        second.append(row)

    timeline = build_hand_timeline(rd.frames_from_fixture(first + second))
    assert timeline["summary"]["hands"] == 2
    assert timeline["hands"][0]["hero"] == ["As", "Kd"]
    assert timeline["hands"][1]["hero"] == ["Qh", "Qs"], (
        "the longer hand's majority must not leak into the shorter one"
    )
    for state in timeline["states"]:
        shared = set(state["hero_cards"]) & set(state["board_cards"])
        assert not shared, f"resolution dealt a card already on the board: {shared}"

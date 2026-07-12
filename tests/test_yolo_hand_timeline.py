"""Reconstruction-spine tests: synthetic 7-class region detections -> full hand.

Uses the region_detections contract with hand-crafted boxes (no detector/OCR) to
drive build_yolo_hand_timeline end to end and assert a complete reconstructed hand:
positions, ordered actions with bet sizes from stack deltas, pot, winner, and the
arithmetic reconciliation. Also confirms the spine output validates cleanly.
"""
from cv_lab.scripts import region_detections as rd
from cv_lab.scripts.build_yolo_hand_timeline import build_hand_timeline
from cv_lab.scripts.validate_yolo_card_timeline import validate_timeline

W = H = 1000


def _det(cls, nx, ny, attr=None, conf=1.0):
    cx, cy = nx * W, ny * H
    return {"cls": cls, "conf": conf, "xyxy": [cx - 15, cy - 20, cx + 15, cy + 20], "attr": attr}


# Hero seat centroid ~ (0.50, 0.86); villain seat 4 centroid ~ (0.50, 0.14).
def _hero_cards():
    return [_det("face_card", 0.46, 0.85, "As"), _det("face_card", 0.52, 0.85, "Kd")]


def _board(cards):
    xs = [0.40, 0.45, 0.50, 0.55, 0.60]
    return [_det("face_card", xs[i], 0.45, c) for i, c in enumerate(cards)]


FLOP = ["2c", "7d", "9h"]
TURN = FLOP + ["Ts"]
RIVER = TURN + ["Jc"]


def _frame(time_s, *, board, s0, s4, pot=None, pill0=None, pill4=None, active=4):
    dets = _hero_cards() + _board(board)
    dets.append(_det("card_back", 0.50, 0.14))          # villain dealt in (seat 4)
    dets.append(_det("dealer_button", 0.50, 0.14))      # dealer = seat 4
    dets.append(_det("stack_text", 0.50, 0.90, s0))     # hero stack (seat 0)
    dets.append(_det("stack_text", 0.50, 0.18, s4))     # villain stack (seat 4)
    if pot is not None:
        dets.append(_det("pot_text", 0.50, 0.32, pot))
    if pill0 is not None:
        dets.append(_det("action_pill", 0.50, 0.80, pill0))
    if pill4 is not None:
        dets.append(_det("action_pill", 0.50, 0.20, pill4))
    dets.append(_det("active_turn_indicator", *rd.SEAT_CENTROIDS[active]))
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

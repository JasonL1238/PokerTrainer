"""Recording-level regressions for the CV geometry / OCR / pot / confidence fixes.

The unit tests next door pin each mechanism on a single frame or a handful of
hand-crafted states. These replay WHOLE recordings and assert the end-to-end
consequence -- how many hands come out, which boards, what the export gate does --
because every defect in this round was silent at the unit level and only visible
in the finished record.

Input is the retained per-frame detector + card-classifier output from the
5-geometry measurement run, frozen under tests/fixtures/cv/ (see
tests/fixtures/PROVENANCE.json). No video is decoded and no model is loaded, so
these run in the ordinary suite. Because the detector rows are byte-identical
going in, any difference coming out is attributable to the reconstruction code
alone.

The frozen `attr` values are the OCR reads of that run, i.e. from BEFORE the
decimal fix. That is deliberate for the two zoning recordings: it makes them a
clean isolate of the zoning change. The assertions here are correspondingly
card-level (hand count, boards, heroes, streets), which no OCR change can move.
The one place the stale reads are the point -- the 1272x896 stack ledger -- is
tested explicitly as such.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from cv_lab.scripts.eval.validate_yolo_card_timeline import validate_timeline
from cv_lab.scripts.pipeline import landmark_anchor as la
from cv_lab.scripts.pipeline import region_detections as rd
from cv_lab.scripts.pipeline.build_yolo_card_timeline import _zone_for_box
from cv_lab.scripts.pipeline.build_yolo_hand_timeline import (
    _frame_state,
    _session_anchor,
    build_hand_timeline,
)
from cv_lab.scripts.pipeline.export_yolo_card_hands_for_app import (
    _confidence_for_hand,
    timeline_to_session_payload,
)

CV_FIXTURES = Path(__file__).parent / "fixtures" / "cv"

BASELINE = "g0723a_baseline_ar1397_frames.json.gz"   # 2054x1470, AR 1.397, the reference basis
AR1750 = "g0621_ar1750_frames.json.gz"               # 2062x1178, AR 1.750, the destroyed session


def _rows(name: str) -> list[dict]:
    """The raw frame dicts, freshly parsed so a test may edit its own copy."""
    return json.loads(gzip.open(CV_FIXTURES / name, "rb").read())


def _frames(name: str) -> list[rd.Frame]:
    return rd.frames_from_fixture(_rows(name))


@pytest.fixture(scope="module")
def baseline_timeline() -> dict:
    return build_hand_timeline(_frames(BASELINE))


@pytest.fixture(scope="module")
def ar1750_timeline() -> dict:
    return build_hand_timeline(_frames(AR1750))


def _zone_counts(frames: list[rd.Frame]) -> dict[str, int]:
    """Face cards zoned board/hero by the anchored path, and by the legacy
    unanchored window, over the same detections and the same anchor."""
    counts = dict(board=0, hero=0, legacy_board=0, legacy_hero=0)
    for frame in frames:
        anchor = rd.anchor_for_frame(frame)
        view = rd.assign_regions(frame, anchor=anchor)
        counts["board"] += len(view["board"])
        counts["hero"] += len(view["hero"])
        if anchor is None:
            continue
        for det in (d for d in frame.detections if d.cls == "face_card"):
            cx = (det.xyxy[0] + det.xyxy[2]) / 2.0 / frame.width
            cy = (det.xyxy[1] + det.xyxy[3]) / 2.0 / frame.height
            zone = _zone_for_box(cx, cy)
            if zone in ("board", "hero"):
                counts[f"legacy_{zone}"] += 1
    return counts


# --------------------------------------------------------------------------- #
# Defect 1 -- zone anchoring at AR 1.750
# --------------------------------------------------------------------------- #
def test_ar1750_board_zoning_recovers_the_measured_detections() -> None:
    """The headline number. On this recording the community row renders at raw
    normalized cy 0.335-0.338, under the legacy board window's 0.36 floor, so the
    legacy rectangles classified 1 of 1152 face-card detections as board -- a
    ~100% loss of every board in the session, with a correct detector and a
    correct classifier behind it. Anchoring the zones to the table recovers 435.

    Hero must come out bit-identical: the hero window had 0.004 of normalized
    margin left here, so a change that moved it would be luck, not a fix."""
    counts = _zone_counts(_frames(AR1750))
    assert counts["legacy_board"] == 1
    assert counts["board"] == 435
    assert counts["hero"] == counts["legacy_hero"] == 704


def test_ar1750_recording_reconstructs_the_lost_board_and_streets(ar1750_timeline) -> None:
    """The consequence in the finished record. Hand 1 is a four-street showdown;
    with its board destroyed it reconstructed as a single preflop street with 13
    actions on it, and reported no warning of any kind."""
    hands = ar1750_timeline["hands"]
    assert len(hands) == 3

    played = hands[0]
    assert played["board"] == ["8d", "Kd", "Jc", "9h", "Jh"]
    assert played["hero"] == ["Td", "Qs"]
    assert [s["street"] for s in played["streets"]] == ["flop", "turn", "river"]
    assert sorted({a["street"] for a in played["actions"]}) == ["flop", "river", "turn"]

    # Hand 3 is a genuine preflop fold: an empty board there is the correct
    # reading, and nothing added here may start rejecting it.
    folded = hands[2]
    assert folded["board"] == []
    assert [s["street"] for s in folded["streets"]] == ["preflop"]
    # The round-2 "disagreeing bet reads (0.5 then 6.0)" refusal is GONE, and
    # it should be: at aspect 1.750 the frame-normalized seat map filed one
    # player's card backs and 0.5 bet under seat 3 while their anchor-fitted
    # stack read as seat 2 -- the "conflict" was two different players' bets
    # merged onto one seat index. Attribution now runs in the anchored basis
    # (measured on this recording: 231 bet_text and 227 card_back boxes move
    # from seat 3 to seat 2, and 33 bet_text from seat 5 to 6), every class
    # agrees per player, and the hand reads cleanly.
    assert folded["warnings"] == []


def test_ar1750_every_card_bearing_state_anchors(ar1750_timeline) -> None:
    """Anchor health across the recording, including the session-median fallback
    doing real work on frames whose own landmark set is too sparse to fit."""
    states = ar1750_timeline["states"]
    assert all(s["anchor_ok"] for s in states)
    assert {s["anchor_source"] for s in states} == {"frame", "session"}
    assert sum(s["unanchored_cards"] for s in states) == 0


def test_ar1750_export_no_longer_ships_a_played_hand_with_an_empty_board(
    ar1750_timeline,
) -> None:
    """board_cards='' on a hand that reached showdown used to leave the CV export
    silently, tagged nothing, at confidence 0.95.

    The board defect is fixed at the source -- hand 1 carries its five community
    cards in the timeline. Neither hand reaches the export any more, and that is
    a SEPARATE, later-found defect in the same two hands rather than a regression
    here: both contain an action sequence no client can produce (hand 1's seat 5
    calls 75.0 on the river and then folds with nothing raised in between; hand
    3's seat 3 checks facing a 6.0 raise it never matched). Before
    action_sequence_illegal existed they exported at confidence 1.0 with tags [].
    """
    assert ar1750_timeline["hands"][0]["board"] == ["8d", "Kd", "Jc", "9h", "Jh"]
    payload = timeline_to_session_payload(
        ar1750_timeline, timeline_path="t.json", session_name="S")
    # The assertion that used to sit here -- "no exported hand has an empty board
    # AND a non-empty action list" -- was worth nothing twice over. This recording
    # now exports zero hands, so `not any(...)` over an empty list can never fail
    # whatever the zoning code does; and the property is not an invariant anyway,
    # because a genuine preflop fold legitimately has an empty board and a full
    # action list (the 07-11 recording exports exactly such a hand at confidence
    # 1.0). What IS invariant is that every hand that reached a street past
    # preflop carries the board that street implies, and that the one empty board
    # here is a hand that never saw a flop.
    per_hand = [(h["hand_number"], len(h["board"]), [s["street"] for s in h["streets"]],
                 h["terminal_event"]) for h in ar1750_timeline["hands"]]
    # Hand 2's terminal_event was "unobserved" while the green CALL/BET pill
    # ambiguity was forced to "call": that call opened the turn with no prior
    # aggression, so street_has_bet stayed False and the spine DISCARDED seat 6's
    # observed FOLD pill, leaving the hand with no ending. Resolving the same pill
    # structurally (nobody has bet -> it is a bet) restores the fold, and the hand
    # ends the way the recording shows: seat 3 bets 13.1 on the turn, seat 6 folds.
    assert per_hand == [
        (1, 5, ["flop", "turn", "river"], "showdown"),
        (2, 4, ["preflop", "flop", "turn"], "fold_win"),
        (3, 0, ["preflop"], "hero_fold"),
    ]
    for _n, n_board, streets, _terminal in per_hand:
        past_preflop = [s for s in streets if s != "preflop"]
        assert bool(past_preflop) == bool(n_board), (streets, n_board)
    # Hands 1 AND 3 export now (renumbered 1 and 2 on the way out). Hand 3 was
    # held back by action_sequence_illegal + starting_stack_unknown -- both
    # artifacts of the frame-normalized seat map splitting one player's HUD
    # across seats 2 and 3 at this aspect ratio, and both gone in the anchored
    # basis. Hand 2 stays held back for its genuine reconstruction faults
    # (board_regression, street_order_issue), which are what the skip pins.
    assert [h["hand"]["hand_number"] for h in payload["hands"]] == [1, 2]
    assert [sorted(s["codes"]) for s in payload["cv_import_summary"]["skipped"]] == [
        ["board_regression", "street_order_issue"],
    ]


# --------------------------------------------------------------------------- #
# Defect 1 -- fail closed, across a whole recording
# --------------------------------------------------------------------------- #
def test_a_recording_without_landmarks_zones_nothing_at_all() -> None:
    """No silent fallback. With every stack_text box removed, no frame can fit the
    table and the session median has nothing to average, so the zones are simply
    unavailable -- and 1380 face cards must be dropped rather than assigned to a
    guessed rectangle.

    Against the unanchored window the same input yields 648 board + 674 hero cards
    stated with full confidence, which is precisely the failure mode: the old code
    had no way to know it was reading the wrong part of the screen."""
    rows = _rows(BASELINE)
    for row in rows:
        row["detections"] = [d for d in row["detections"] if d["cls"] != "stack_text"]
    frames = rd.frames_from_fixture(rows)
    assert sum(1 for f in frames for d in f.detections if d.cls == "face_card") == 1380

    views = [rd.assign_regions(f) for f in frames]
    assert not any(v["anchor_ok"] for v in views)
    assert sum(len(v["board"]) for v in views) == 0
    assert sum(len(v["hero"]) for v in views) == 0
    assert sum(len(c) for v in views for c in v["villain_cards"].values()) == 0
    assert sum(v["unanchored_cards"] for v in views) == 1380

    timeline = build_hand_timeline(frames)
    assert not any(s["anchor_ok"] for s in timeline["states"])
    assert not any(h["complete_cards"] for h in timeline["hands"])


# --------------------------------------------------------------------------- #
# The strong parts -- AR 1.397 baseline must not move
# --------------------------------------------------------------------------- #
def test_baseline_ar1397_zoning_is_unchanged() -> None:
    """Hero is bit-identical and the board loses exactly 3 of 648. Those 3 are not
    community cards: they sit at reference ry 0.48-0.55, above the measured
    community row (0.437-0.451), and are mid-deal/reveal strays the legacy band
    was wide enough to admit. Anything beyond 3 is a real regression."""
    counts = _zone_counts(_frames(BASELINE))
    assert counts["hero"] == counts["legacy_hero"] == 674
    assert counts["legacy_board"] == 648
    assert counts["board"] == 645


def test_the_development_table_straddles_and_the_spine_says_so(baseline_timeline) -> None:
    """The development recording is itself a straddle game: every deal-open
    state shows a third standing bet of 2.0 on the seat left of the BB, under a
    green pill. The spine used to book that as an UTG *call* off the pill --
    arithmetically coherent, but only while the pill was readable, and it left
    positions and preflop order one seat wrong. The structure is now observed
    per session and every hand carries one typed straddle post, position ST,
    with preflop action opening left of it."""
    assert baseline_timeline["metadata"]["forced_post_structure"] == [0.5, 1.0, 2.0]
    for hand in baseline_timeline["hands"]:
        straddles = [a for a in hand["actions"]
                     if a.get("forced_bet_type") == "straddle"]
        assert len(straddles) == 1, hand["hand_number"]
        (post,) = straddles
        assert post["action_type"] == "post_blind"
        assert post["amount"] == 2.0
        assert post["position"] == "ST"
        assert post["is_live_post"] is True


def test_baseline_ar1397_reconstructs_seven_hands(baseline_timeline) -> None:
    """Golden record for the reference geometry: same hand count, same boards,
    same heroes, same streets, every state anchored on its own frame.

    Three warnings, all TRUE positives on this fixture.

    Hand 7's contributions_exceed_pot: the fixture deliberately preserves the
    pre-fix OCR reads (see PROVENANCE: "the 20 bet_text reads of 0.50 that came
    back 50.0"), so it books "preflop seat 3 call 50.0" into a 34.5 BB pot and the
    action ledger sums 44.5 BB above it with no all-in to refund it. Card-level
    assertions are unaffected, which is the point of freezing the reads.

    Hand 7's board_card_identity_split is the round-4 finding itself: the turn card
    reads 8d on eight samples and 8h on two, and the card on screen is the eight of
    HEARTS. The published board is wrong, and before this signal existed the hand
    exported at confidence 1.0 with tags [].

    Hand 4's hero_card_identity_split is the same signal in the other direction:
    hero's ten reads Tc on ~40 samples and Ts on 6, and Tc is what the frame shows
    (verified at t=210). The majority is right there -- which is why the code is a
    review flag rather than a rejection."""
    assert [
        (h["hand_number"], " ".join(h["hero"]), " ".join(h["board"]),
         [s["street"] for s in h["streets"]], h["complete_cards"], h["warnings"])
        for h in baseline_timeline["hands"]
    ] == [
        (1, "Ac 5c", "5h 4h 3s 2h 7h", ["preflop", "flop", "turn", "river"], True, []),
        (2, "Qs Jh", "3h Kh Td Ac Th", ["preflop", "flop", "turn", "river"], True, []),
        (3, "Jd 8s", "9c Ac 8d Jc", ["preflop", "flop", "turn"], True, []),
        (4, "Tc 4c", "8h 5h Qs", ["preflop", "flop"], True,
         ["hero_card_identity_split"]),
        (5, "8c 6c", "4c 3d 5s 6s Jh", ["preflop", "flop", "turn", "river"], True, []),
        (6, "5h Ad", "7h Ah As", ["preflop", "flop"], True, []),
        (7, "Ts 3c", "Js 6h 4c 8d", ["preflop", "flop", "turn"],
         True, ["board_card_identity_split", "contributions_exceed_pot"]),
    ]
    states = baseline_timeline["states"]
    assert len(states) == 169
    assert {s["anchor_source"] for s in states} == {"frame"}
    assert sum(s["unanchored_cards"] for s in states) == 0


def test_baseline_ar1397_export_reflects_button_order_repair(baseline_timeline) -> None:
    """Four hands out with their boards; three held back, each for a named,
    verified reason. No rejection signal may take a hand off this list without one.

    Hand 1 used to be held back because numeric seat sorting put its already-visible
    BB raise before an earlier-position call, making the later fold appear to face
    no raise. Button-derived ordering now reconstructs the coherent sequence and
    retains all eight occupied seats despite three pre-capture folds.

    The remaining departures are not regressions -- each hand carries a record
    that cannot be true:
      hand 2  river    seat 3 goes all-in for 148.1 and then folds;
      hand 7  preflop  seat 3 "calls 50.0" into a 34.5 BB pot -- the frozen
              pre-fix read of an on-screen 0.50, so the ledger overshoots the pot
              by 44.5 BB with no all-in to refund it.
    Their boards and heroes are unchanged in the timeline (asserted above by
    test_baseline_ar1397_reconstructs_seven_hands); only the export gate moved."""
    payload = timeline_to_session_payload(
        baseline_timeline, timeline_path="t.json", session_name="S")
    assert [h["hand"]["board_cards"] for h in payload["hands"]] == [
        "5h 4h 3s 2h 7h", "8h 5h Qs", "4c 3d 5s 6s Jh", "7h Ah As",
    ]
    summary = payload["cv_import_summary"]
    assert summary["exported_hands"] == 4
    assert [(s["timeline_hand_number"], sorted(s["codes"])) for s in summary["skipped"]] == [
        (2, ["action_sequence_illegal"]),
        (3, ["board_regression", "street_order_issue"]),
        (7, ["contributions_exceed_pot"]),
    ]


# --------------------------------------------------------------------------- #
# Rejection signal A -- a community row nothing zoned as board.
# A signal that fires on everything is worth nothing, so both directions are
# asserted against real recordings, not synthetic shapes.
# --------------------------------------------------------------------------- #
def test_board_row_missed_fires_on_the_measured_ar1750_failure() -> None:
    """Replay of the exact old zoning decisions -- same frames, same row geometry,
    only zone_for_ref swapped for the legacy window -- to show the net is loud on
    the failure it exists for. It fires on 104 of the 352 card-bearing frames."""
    fired = 0
    card_frames = 0
    for frame in _frames(AR1750):
        cards = [d for d in frame.detections if d.cls == "face_card"]
        anchor = rd.anchor_for_frame(frame)
        if not cards or anchor is None:
            continue
        card_frames += 1
        legacy = []
        for det in cards:
            px = (det.xyxy[0] + det.xyxy[2]) / 2.0
            py = (det.xyxy[1] + det.xyxy[3]) / 2.0
            rx, ry = anchor.to_ref(px, py)
            width_ref = (det.xyxy[2] - det.xyxy[0]) / anchor.s / la.REF_DET_W
            legacy.append((rx, ry, _zone_for_box(px / frame.width, py / frame.height),
                           width_ref))
        fired += rd._board_row_missed(legacy)
    assert card_frames == 352
    assert fired == 104


def test_board_row_missed_is_silent_once_the_zones_are_anchored(
    ar1750_timeline, baseline_timeline,
) -> None:
    """The other direction, on the same recording that fires 104 times unanchored,
    plus the healthy reference geometry. A net still firing after the fix would
    mean the fix is incomplete."""
    assert sum(s["board_row_missed"] for s in ar1750_timeline["states"]) == 0
    assert sum(s["board_row_missed"] for s in baseline_timeline["states"]) == 0


# --------------------------------------------------------------------------- #
# Rejection signal B -- stack magnitudes implausible against their own siblings.
# Replayed over the REAL per-frame stack ledgers, pre-fix (1272x896, where a
# dropped decimal inflated 245 reads 100x) and clean (2054x1470).
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def inflated_stack_frames() -> list[list[float]]:
    rows = json.loads(
        (CV_FIXTURES / "g0723b_stack_reads_1272x896.json").read_text(encoding="utf-8"))
    return [r["stacks"] for r in rows]


@pytest.fixture(scope="module")
def clean_stack_frames() -> list[list[float]]:
    out = []
    for row in _rows(BASELINE):
        vals = [d["attr"] for d in row["detections"]
                if d["cls"] == "stack_text" and d.get("attr") is not None]
        if len(vals) >= 4:
            out.append(sorted(vals))
    return out


def _stack_frame(values: dict[int, float], source: str | None = "integer") -> rd.Frame:
    """One synthetic 2054x1470 frame carrying `values` at the real seat anchors."""
    return rd.Frame("x", 0.0, 2054, 1470, [
        rd.Detection("stack_text", 0.9,
                     (x * 2054 - 40, y * 1470 - 20, x * 2054 + 40, y * 1470 + 20),
                     values[seat], attr_source=source)
        for seat, (x, y) in rd.SEAT_ANCHORS_BY_CLASS["stack_text"].items()
        if seat in values
    ])


def test_the_sibling_median_stack_net_is_deleted(inflated_stack_frames) -> None:
    """SUPERSEDES test_stack_outlier_rule_rejects_every_measured_100x_read,
    test_stack_outlier_rule_is_silent_on_the_clean_baseline and
    test_stack_outlier_rule_keeps_reads_whose_decimal_was_located.

    Those three pinned `_reject_stack_outliers`: drop a stack read at least 6.0x
    the frame's other stacks when the read located no decimal point. On this
    recording's PRE-FIX ledger (the fixture below, 197 frames, 245 reads inflated
    100x by a dropped decimal) it removed exactly the broken reads.

    It is deleted, and the deletion is measured in both directions:

    * THE DEFECT IS CLOSED UPSTREAM. Those 245 reads no longer exist. The reader's
      P5 refuses an integer whose widest inter-digit gap reaches the band a
      located decimal occupies (`integer_over_decimal_band`), which is evidence
      from the numeral's own spacing rather than from unrelated seats. What the
      spine receives from this geometry today is UNKNOWN, not 20410.0.
    * THE NET'S PREMISE IS FALSE ON THE SIXTH DEVELOPMENT RECORDING.
      clubwpt_session_01 carries 541 provable reads at or above 1000 BB (max
      1157.10). Instrumented there, the net fires on 83 frames, drops a legible
      1110.0 BB stack against a ~150 sibling median every time, and raises
      `amount_scale_implausible` -- a SPINE_FATAL code -- on three hands.
      Deleting it took the corpus from 16 exports to 19.

    So the fixture is retained as the RECORD of the old ledger, and what it now
    proves is that the spine no longer second-guesses a value it was handed.
    """
    import cv_lab.scripts.pipeline.build_yolo_hand_timeline as spine

    for gone in ("_reject_stack_outliers", "_STACK_OUTLIER_RATIO",
                 "_STACK_OUTLIER_MIN_READS", "_DECIMAL_NOT_LOCATED"):
        assert not hasattr(spine, gone)

    assert len(inflated_stack_frames) == 197
    worst = max(inflated_stack_frames, key=max)
    assert max(worst) >= 20000.0, "fixture no longer holds the pre-fix ledger"
    kept = _frame_state(_stack_frame(dict(enumerate(worst[:8]))))["stacks"]
    assert max(kept.values()) == max(worst[:8]), (
        "the spine must pass through what the reader proved, or refuse at the reader")


def test_the_deep_stack_recording_is_not_rejected_as_implausible(
    clean_stack_frames,
) -> None:
    """The measured clubwpt_session_01 shape: one seat at 1110.0 BB against a
    ~150 BB sibling median, legible on screen, every read a proven integer. The
    deleted net dropped it on 83 frames of that recording and then rejected three
    of its hands. It must survive into the state.

    The clean 2054x1470 ledger is the negative control it always was: no read is
    removed there either, because none is removed anywhere any more."""
    deep = {0: 150.0, 1: 148.0, 2: 152.0, 3: 149.0, 4: 151.0, 5: 147.0,
            6: 1110.0, 7: 150.0}
    assert _frame_state(_stack_frame(deep))["stacks"][6] == 1110.0

    assert len(clean_stack_frames) == 361
    for values in clean_stack_frames:
        state = _frame_state(_stack_frame(dict(enumerate(values[:8]))))
        assert len(state["stacks"]) == len(values[:8])


# --------------------------------------------------------------------------- #
# Defect 4 -- confidence on the real destroyed hand
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def lost_board_hand() -> dict:
    """The actual hand the pre-fix pipeline produced for the AR 1.750 session: a
    four-street showdown reconstructed as 13 preflop actions with no board, no
    warnings, pot 225.9 and 'Hero wins +116 BB'."""
    return json.loads(
        (CV_FIXTURES / "g0621_lost_board_hand.json").read_text(encoding="utf-8"))


def test_the_real_lost_board_hand_is_not_high_confidence(lost_board_hand) -> None:
    """It scored 0.95 as exported and sat in a session scored 0.989, because both
    scores were functions of a warning count and losing a board raises no warning.

    Note the hand's own `streets` list collapsed to ['preflop'] along with its
    board -- both are derived from the same community-card readings -- so the
    street-count check cannot see this, and the action sequence is the only
    independent witness left."""
    assert lost_board_hand["board"] == []
    assert [s["street"] for s in lost_board_hand["streets"]] == ["preflop"]
    assert sorted({a["street"] for a in lost_board_hand["actions"]}) == ["preflop"]
    assert lost_board_hand["warnings"] == []

    report = validate_timeline({"hands": [lost_board_hand], "states": []})
    codes = [w["code"] for w in report["hands"][0]["warnings"]]
    assert "board_empty_but_streets_advanced" in codes
    assert report["hands"][0]["confidence_score"] <= 0.5
    assert _confidence_for_hand(lost_board_hand, codes) <= 0.5


def test_the_real_lost_board_hand_is_rejected_by_the_export_gate(lost_board_hand) -> None:
    payload = timeline_to_session_payload(
        {"hands": [lost_board_hand], "states": []},
        timeline_path="t.json", session_name="S")
    summary = payload["cv_import_summary"]
    assert payload["hands"] == []
    assert summary["exported_hands"] == 0
    assert summary["skipped"][0]["reason"] == "validation_warnings"
    assert "board_empty_but_streets_advanced" in summary["skipped"][0]["codes"]


def test_a_real_preflop_fold_is_not_rejected_for_its_empty_board(ar1750_timeline) -> None:
    """False-positive guard on the same recording: hand 3's board really is empty,
    so NO board-related code may fire on it.

    The hand is now warning-free overall. Its old "seat 3 checks facing a 6.0
    raise" (action_sequence_illegal) and "disagreeing 0.5/6.0 standing-bet
    reads" (starting_stack_unknown) were one defect wearing two codes: at
    aspect 1.750 the frame-normalized seat map filed seat 2's HUD under seat 3,
    so one seat index appeared to both check and raise, and to hold two bets at
    once. Anchored attribution reunites each player's boxes and both codes
    dissolve. The property under test is narrower and unchanged: an empty board
    is a legal value and must not by itself cost the hand anything."""
    folded = ar1750_timeline["hands"][2]
    assert folded["board"] == []
    report = validate_timeline({"hands": [folded], "states": []})
    codes = {w["code"] for w in report["hands"][0]["warnings"]}
    assert not {c for c in codes if "board" in c}
    assert codes == set()


# --------------------------------------------------------------------------- #
# A recording that simply STOPS mid-hand. The last hand of every session takes
# this path, and it is the one place a fabricated terminal event is invisible:
# there is no following hand to contradict it.
# --------------------------------------------------------------------------- #
TRUNCATED = "g0723b_truncated_last_hand_frames.json.gz"   # 1272x896, t=180..196


@pytest.fixture(scope="module")
def truncated_timeline() -> dict:
    return build_hand_timeline(_frames(TRUNCATED))


def test_a_recording_truncated_hand_does_not_invent_a_winner(truncated_timeline) -> None:
    """Real frames from the end of the 1272x896 recording (video duration 197.28 s;
    last sampled state t=196.0). The hand is still preflop: no flop was ever dealt,
    no pot was ever swept, hero has folded and five seats are still contested.

    It exported as hero_cards='3d 2s', board_cards='', pot 17.5, result='Villain
    wins', confidence 1.0, tags=[], completion_status='complete', with
    terminal_event='showdown' on a ZERO-card board. The whole "Villain wins" rested
    on seat 3's stack reading 182.0 at t=195 and 182.5 at t=196 -- 0.5 BB of jitter,
    0.03x the pot -- which the argmax over stack gains was free to call a sweep.

    t_start is 184.0, not 185.0: segmentation cuts at the first state showing
    the new DEAL (all eight card backs, blinds pot 7.5), one state before the
    hero's own cards are read. That deal-open state carries the hand's true
    opening roster and forced posts and belongs to this hand, not to the tail
    of the previous one."""
    hand = truncated_timeline["hands"][-1]
    assert (hand["t_start"], hand["t_end"]) == (184.0, 196.0)
    assert hand["hero"] == ["3d", "2s"]
    assert hand["board"] == []
    assert hand["pot"] == 17.5
    assert hand["winner_seat"] is None, f"win_gain {hand['win_gain']} on pot {hand['pot']}"
    assert hand["result"] == "Hero folds"
    assert hand["terminal_event"] == "hero_fold"
    assert "showdown" not in hand["terminal_event"]


def test_the_truncated_hand_exports_without_a_fabricated_result(
    truncated_timeline,
) -> None:
    """The published record, end to end. The hero's fold IS a complete record of
    the hero's decision, so the hand still exports -- but it must not claim a
    villain won a pot nobody was seen to take.

    The recording opens mid-way through the PREVIOUS hand's river; that
    fragment segments apart as its own truncated hand but does NOT export: its
    teardown state stays attached (the interstitial's mid-animation dealer
    marker is refused by the seat rejection radius, so the dealer-moved trim
    arm has no reading to prune it on), the board it shows then empties inside
    the fragment, and validation's street_order_issue blocks it. One state of
    a hand nobody saw played is exactly the record the gate exists to hold
    back."""
    payload = timeline_to_session_payload(
        truncated_timeline,
        timeline_path="t.json",
        session_name="S",
        include_incomplete=True,
    )
    exported = [h["hand"] for h in payload["hands"]]
    assert len(exported) == 1
    (last,) = exported

    assert last["board_cards"] == ""
    assert last["result"] == "Hero folds"
    assert last["hero_bb_won"] == 0.0
    assert last["completion_evidence"]["terminal_event"] == "hero_fold"
    assert not any(h["result"] == "Villain wins" for h in exported)
    assert [sorted(s["codes"]) for s in payload["cv_import_summary"]["skipped"]] == [
        ["street_order_issue"],
    ]


# --------------------------------------------------------------------------- #
# Round 3: the green CALL/BET pill ambiguity.
# --------------------------------------------------------------------------- #
def test_green_pill_opening_a_street_is_a_bet_and_the_fold_survives(
    baseline_timeline,
) -> None:
    """The ClubWPT client paints CALL and BET on the same green pill, and the word
    template misses on 964 of 2439 action_pill reads across the 5 development
    geometries (39.5%); 412 of those are green. The colour fallback resolved green
    to "call" as "the safe default". It is not a default -- it asserts that
    somebody had already bet on that street.

    Baseline recording, hand 7. The flop went check-check, so seat 3 is first to
    act on the turn and a call is physically impossible; the pill on screen reads
    BET. The forced "call" published `turn: seat:3 call 11.5`, and the knock-on
    was worse than the label: street_has_bet stayed False, so the fold handler --
    which correctly refuses folds on a street with nothing to fold to -- DISCARDED
    seat 2's observed FOLD pill, and the round-completion pass synthesised a CHECK
    for that seat instead. The exported record therefore contained an action that
    never happened and omitted one that did, with warnings=[] and rejection
    codes=[].

    Structure resolves what colour cannot: on a street where nobody has bet, a
    seat putting chips in is BETTING.
    """
    hand = next(h for h in baseline_timeline["hands"] if h["hand_number"] == 7)
    turn = [(a["seat"], a["action_type"], a["amount"]) for a in hand["actions"]
            if a["street"] == "turn"]
    assert turn == [(3, "bet", 11.5), (2, "fold", None)]
    assert "inferred_round_complete" not in {
        a["derivation"] for a in hand["actions"] if a["street"] == "turn"
    }


def test_no_reconstructed_hand_opens_a_street_with_a_call(baseline_timeline) -> None:
    """The class, not the instance. A call needs something to call: on any street
    past preflop (where the blinds are a standing bet) a call before the street's
    first bet/raise/all-in is a reconstruction that contradicts itself.

    Five of the 21 hands reconstructed from the development corpus carried one,
    every one with spine warnings=[] -- g0723a #2 (flop, three seats, and again on
    the turn), g0723a #7, g0621 #1 (flop, two seats), g0621 #2 (turn), g0723b #2
    (flop).
    """
    for hand in baseline_timeline["hands"]:
        aggressed: set[str] = set()
        for action in hand["actions"]:
            street = action["street"]
            if action["action_type"] == "call" and street != "preflop":
                assert street in aggressed, (
                    f"hand {hand['hand_number']}: {street} seat {action['seat']} "
                    f"calls with no prior aggression ({action['derivation']})"
                )
            if action["action_type"] in {"bet", "raise", "all-in"}:
                aggressed.add(street)


def test_green_colour_fallback_does_not_claim_call(monkeypatch) -> None:
    """Unit form of the same rule, at the layer where the claim was made."""
    det = rd.Detection(cls="action_pill", conf=0.9, xyxy=(0, 0, 10, 10), attr="green")
    assert rd.read_pill_action(det, dealt_in=True) == rd.PILL_BET_OR_CALL
    assert rd.read_pill_action(det, dealt_in=True) != "call"
    # A pill whose WORD was read is still definitive.
    for word in ("call", "bet", "raise", "check", "fold"):
        worded = rd.Detection(cls="action_pill", conf=0.9, xyxy=(0, 0, 10, 10), attr=word)
        assert rd.read_pill_action(worded, dealt_in=True) == word


# --------------------------------------------------------------------------- #
# Round 3: facts that used to be indistinguishable from "checked and clean".
# --------------------------------------------------------------------------- #
def test_seat_assignment_is_invariant_under_window_resize_and_chrome_offset() -> None:
    """The generalization the anchored basis buys, pinned. One synthetic table
    rendered three ways -- native, resized 0.8x and 1.3x, and shifted 180x120px
    as if desktop chrome surrounded the client -- must produce IDENTICAL seat
    attributions for every seated class. Under frame-normalized anchors the
    shifted variant silently rotated reads onto neighbouring seats (the
    2026-08-03 failure class); under the anchored basis the fitted transform
    absorbs scale and offset entirely."""
    def table(scale: float, dx: float = 0.0, dy: float = 0.0) -> rd.Frame:
        w, h = round(2054 * scale + 2 * dx), round(1470 * scale + 2 * dy)
        dets = []

        def box(cls, nx, ny, attr=None, half=(40, 20)):
            px, py = nx * 2054 * scale + dx, ny * 1470 * scale + dy
            hx, hy = half[0] * scale, half[1] * scale
            dets.append(rd.Detection(cls, 0.9, (px - hx, py - hy, px + hx, py + hy), attr))

        for seat, (nx, ny) in rd.SEAT_ANCHORS_BY_CLASS["stack_text"].items():
            box("stack_text", nx, ny, 100.0 + seat)
        for seat in (1, 3, 6):
            box("card_back", *rd.SEAT_ANCHORS_BY_CLASS["card_back"][seat])
        box("dealer_button", *rd.SEAT_ANCHORS_BY_CLASS["dealer_button"][5])
        box("active_turn_indicator", *rd.SEAT_ANCHORS_BY_CLASS["active_turn_indicator"][6])
        box("bet_text", *rd.SEAT_ANCHORS_BY_CLASS["bet_text"][3], attr=2.0)
        return rd.Frame("x", 0.0, w, h, dets)

    views = [rd.assign_regions(table(1.0)),
             rd.assign_regions(table(0.8)),
             rd.assign_regions(table(1.3)),
             rd.assign_regions(table(1.0, dx=180.0, dy=120.0))]
    reference = views[0]
    assert reference["dealer_seat"] == 5
    assert reference["active_seat"] == 6
    assert {s for s, info in reference["seats"].items() if info["card_back"]} == {1, 3, 6}
    assert {s: info["stack"] for s, info in reference["seats"].items()
            if info["stack"] is not None} == {s: 100.0 + s for s in range(8)}
    for view in views[1:]:
        assert view["dealer_seat"] == reference["dealer_seat"]
        assert view["active_seat"] == reference["active_seat"]
        assert {s: info["card_back"] for s, info in view["seats"].items()} \
            == {s: info["card_back"] for s, info in reference["seats"].items()}
        assert {s: info["stack"] for s, info in view["seats"].items()} \
            == {s: info["stack"] for s, info in reference["seats"].items()}
        assert view["seat_unassigned"] == 0
        assert view["unanchored_seated"] == 0


def test_a_short_handed_table_no_longer_has_an_unrunnable_check_to_record() -> None:
    """SUPERSEDES test_the_stack_outlier_net_records_when_it_could_not_run.

    That test pinned `stack_outlier_check_skipped`: the sibling-median net needed
    four reads to take a median of, so on a 6-max or heads-up table -- ClubWPT's
    normal case, and never exercised on the development corpus, which renders
    eight stack boxes throughout -- it silently did not run, and the skip had to
    be recorded so it was not mistaken for "checked and clean".

    The net is deleted, so there is no check to skip and nothing to record. What
    replaces it is stronger and needs no siblings at all: a read is either PROVEN
    at the reader or it is UNKNOWN with a named reason, and a three-seat table
    gets exactly the same treatment as an eight-seat one -- given a table
    transform. Three landmarks cannot fit a TRUSTED anchor by themselves
    (ANCHOR_MIN_TRUSTED_POINTS), so the sparse frame reads through the session
    anchor, exactly as the production spine's session-median fallback does."""
    import cv_lab.scripts.pipeline.build_yolo_hand_timeline as spine

    assert not hasattr(spine, "_reject_stack_outliers")

    session = rd.anchor_for_frame(_stack_frame({s: 100.0 for s in range(8)}))
    three = {seat: v for seat, v in enumerate((10000.0, 100.0, 100.0))}
    state = _frame_state(_stack_frame(three), session)
    assert "stack_outlier_check_skipped" not in state
    assert sorted(state["stacks"].values()) == [100.0, 100.0, 10000.0], (
        "a short-handed table is read exactly like a full one")
    # ... and a refusal on the same table is carried with its reason, which is the
    # channel that replaced the net.
    refused = rd.Frame("x", 0.0, 2054, 1470, [
        rd.Detection("stack_text", 0.9, (x * 2054 - 40, y * 1470 - 20,
                                         x * 2054 + 40, y * 1470 + 20),
                     None, attr_source="integer_over_decimal_band")
        for x, y in list(rd.SEAT_ANCHORS_BY_CLASS["stack_text"].values())[:3]
    ])
    unknown = _frame_state(refused, session)["stacks_unknown"]
    assert set(unknown.values()) == {"integer_over_decimal_band"}


def test_two_disagreeing_stack_reads_for_one_seat_are_unknown_not_last_write() -> None:
    """`seat(i)["stack"] = value` inside the detection loop is unconditional, so
    two stack_text boxes mapping to one seat resolved by ITERATION ORDER -- and
    when the later one is unreadable, a good read is destroyed with no counter and
    no flag. Measured over the 1309 card-bearing frames of the 5 development
    geometries: 3 seat slots receive two boxes, and all 3 disagree (g0621 t=105
    seat 4 reads 142.6 against 9.0, a 15.8x split; g0723b t=187/188 seat 1 reads
    295.7 against None).

    Two contradictory readings of one seat are a conflict. PLAN.md is explicit
    that an absent amount is unknown, not a guess, and the same standard applies
    here."""
    x, y = rd.SEAT_ANCHORS_BY_CLASS["stack_text"][4]
    px, py = x * 2054, y * 1470

    def frame(*values):
        return rd.Frame("x", 0.0, 2054, 1470, [
            rd.Detection("stack_text", 0.9,
                         (px - 40 + 3 * i, py - 20, px + 40 + 3 * i, py + 20), v)
            for i, v in enumerate(values)
        ])

    # Two boxes cannot anchor a frame by themselves (seat attribution is
    # anchored and fails closed without a transform), so supply the session
    # anchor the production path would: a fit from a fully-landmarked frame.
    session = rd.anchor_for_frame(_stack_frame({s: 100.0 for s in range(8)}))
    assert session is not None

    agreeing = rd.assign_regions(frame(142.6, 142.6), anchor=session)
    assert agreeing["seats"][4]["stack"] == 142.6
    assert agreeing["stack_conflicts"] == 0

    disagreeing = rd.assign_regions(frame(142.6, 9.0), anchor=session)
    assert disagreeing["seats"][4]["stack"] is None
    assert disagreeing["stack_conflicts"] == 1
    assert disagreeing["amounts_unknown"] == 1

    # The destructive direction: a good read followed by an unreadable one.
    destroyed = rd.assign_regions(frame(295.7, None), anchor=session)
    assert destroyed["seats"][4]["stack"] is None
    assert destroyed["stack_conflicts"] == 1


def test_a_pot_box_dropped_by_the_column_guard_is_counted() -> None:
    """A pot_text outside the table's centre column is discarded, and the discard
    was invisible: pot came back None with every other field unchanged, including
    amounts_unknown. Downstream that raises FEWER signals than a mis-reconciled
    pot, because pot_not_reconciled is itself gated on `final_pot is not None`.

    The guard's own comment concedes the reject branch is unexercised on this
    corpus (0 of 1267 pot_text boxes sit off-column), so a counter is the only way
    a future geometry that does trip it can be told apart from a missing pot."""
    from cv_lab.scripts.pipeline.landmark_anchor import (
        REF_DET_H,
        REF_DET_W,
        REF_POT_RX,
        TableAnchor,
    )
    anchor = TableAnchor(1.0, 0.0, 0.0, 0.0, 8, "frame")

    def view_at(rx: float):
        px, py = rx * REF_DET_W, 0.36 * REF_DET_H
        frame = rd.Frame("x", 0.0, int(REF_DET_W), int(REF_DET_H),
                         [rd.Detection("pot_text", 0.9,
                                       (px - 30, py - 10, px + 30, py + 10), 240.9)])
        return rd.assign_regions(frame, anchor=anchor)

    inside = view_at(REF_POT_RX[0])
    assert (inside["pot"], inside["pot_text_off_column"]) == (240.9, 0)

    outside = view_at(REF_POT_RX[0] - 0.05)
    assert outside["pot"] is None
    assert outside["pot_text_off_column"] == 1, (
        "a discarded pot box must be distinguishable from a pot that was never found"
    )


def test_the_render_geometry_reaches_the_record() -> None:
    """`layout_profile` is read at export from timeline metadata that nothing in
    cv_lab ever wrote, so it was always the empty string: the pipeline had no
    representation at all of the geometry it reconstructed from, and therefore
    none of "this client renders smaller than anything the readers were calibrated
    on". Both the seat anchors and the OCR templates are calibrated artefacts."""
    baseline = build_hand_timeline(_frames(BASELINE))
    assert baseline["metadata"]["layout_profile"] == "2054x1470"

    rows = _rows(BASELINE)
    for row in rows:
        row["width"], row["height"] = 640, 448          # half the smallest calibrated client
        for det in row["detections"]:
            det["xyxy"] = [v / 3.2 for v in det["xyxy"]]
    tiny = build_hand_timeline(rd.frames_from_fixture(rows))
    assert tiny["metadata"]["layout_profile"] == "640x448-unsupported"


def test_hero_net_counts_chips_committed_before_the_first_sample(
    baseline_timeline,
) -> None:
    """The client renders stacks PRE-DEBITED, so `series[-1] - series[0]` cannot see
    anything the hero committed before the hand's first sampled state -- the blinds
    on every hand, and the whole preflop action on a hand the segmenter picks up
    late. It was published as a bare `hero_bb_won` with warnings=[].

    Measured on the 07-23 3.33.54 PM session's first hand: the hero's 24.0 BB raise
    stands in bet_text for 18 consecutive states while the hero's stack never moves,
    and the export reported +52.8 for a hand whose net is +28.8.

    On the baseline recording the same correction is the hero's blind. Hand 1 posts
    2.0, hand 3 posts 1.0, hand 4 posts 0.5 -- amounts a stack delta from the first
    observed state silently omits."""
    from cv_lab.scripts.pipeline.build_yolo_hand_timeline import (
        _committed_at_start,
        _segment,
        build_states,
    )

    states, _events = build_states(_frames(BASELINE))
    hands = list(_segment(states))
    committed = [_committed_at_start(h, 0) for h in hands[:4]]
    assert committed == [2.0, 2.0, 1.0, 0.5], "the standing bet at hand start"

    for hand, blind in zip(baseline_timeline["hands"][:4], committed, strict=False):
        series = [s["stacks"][0] for s in states
                  if hand["t_start"] <= s["time_s"] <= hand["t_end"] and 0 in s["stacks"]]
        if not series:
            continue
        raw_delta = round(series[-1] - series[0], 2)
        if hand["hand_number"] == 2:
            # Hand 2's raw window overshoots its settlement: the tail keeps
            # post-sweep interstitial states (their mid-animation dealer marker
            # is refused by the seat rejection radius instead of snapping, so
            # the dealer-moved trim arm has no reading to fire on), and in
            # them the hero has ALREADY posted the next hand's 0.5 small
            # blind. The spine's net is settlement-bounded by design -- that
            # 0.5 belongs to hand 3 -- so the published net sits 0.5 above the
            # naive whole-window formula: (settled delta 199.9) - (2.0
            # straddle) = 197.9.
            assert hand["hero_bb_won"] == round(raw_delta - blind + 0.5, 2)
            continue
        assert hand["hero_bb_won"] == round(raw_delta - blind, 2), (
            f"hand {hand['hand_number']}: net {hand['hero_bb_won']} ignores the "
            f"{blind} already committed at the first observed state"
        )


def test_the_pill_reader_helper_does_not_claim_call_on_green_either() -> None:
    """`ocr_readers.read_pill_from_image` carries the same colour fallback as
    `region_detections.read_pill_action` and had the same "call is the safe default"
    comment. It has no caller in the reconstruction spine today, which is exactly why
    it needs pinning: a future wiring change must not reintroduce the claim."""
    import numpy as np

    from cv_lab.scripts.pipeline import ocr_readers as ocr

    green = np.zeros((24, 96, 3), np.uint8)
    green[:, :] = (90, 190, 80)          # BGR, the pill's green fill
    resolved = ocr.read_pill_from_image(green, (0, 0, 96, 24), dealt_in=True)
    assert resolved == ocr.PILL_BET_OR_CALL
    assert resolved != "call"


def test_a_contested_card_reaches_the_export_as_a_review_flag(baseline_timeline) -> None:
    """Round 4, adversary A, at the export boundary.

    The 07-23 3.21 PM recording ships a wrong board card -- 'Js 6h 4c 8d' for a
    real 'Js 6h 4c 8h', confirmed on the pixels -- and every hand whose card
    identity the classifier could not settle used to export at confidence 1.0 with
    tags [], indistinguishable from a hand it read unanimously.

    Hand 4 is the case that still exports (hand 7 is held back by chip
    conservation), and it is the honest shape of the fix: the hand is not thrown
    away, it is published with the contest on its face."""
    payload = timeline_to_session_payload(
        baseline_timeline, timeline_path="t.json", session_name="S")
    by_board = {h["hand"]["board_cards"]: h["hand"] for h in payload["hands"]}
    contested = by_board["8h 5h Qs"]
    assert contested["confidence_score"] < 1.0
    assert "LOW_CONFIDENCE" in contested["tags"]
    assert "hero_card_identity_split" in contested["notes"]

    # ... and a hand the classifier WAS unanimous on is untouched.
    clean = by_board["7h Ah As"]
    assert clean["confidence_score"] == 1.0
    assert clean["tags"] == []


def test_board_zone_nets_are_silent_on_every_real_frame_of_both_recordings() -> None:
    """The false-positive bound for the one-state gate, on raw frames rather than
    on the collapsed state list.

    Lowering board_zone_yield_zero / board_zone_yield_partial from a 3-state run to
    a single state is only safe if a healthy frame never raises the flag -- deal
    animations included, since "a card mid-deal is momentarily outside the row" was
    the argument the run length rested on. Over both frozen whole-recording
    fixtures (the 1.397 reference geometry and the 1.750 extreme), with the anchor
    in place, neither flag is raised on a single frame."""
    for fixture in (BASELINE, AR1750):
        missed = partial = 0
        frames = _frames(fixture)
        session = _session_anchor(frames)
        for frame in frames:
            view = rd.assign_regions(frame, anchor=rd.anchor_for_frame(frame) or session)
            missed += view["board_row_missed"]
            partial += view["board_row_partial"]
        assert (missed, partial) == (0, 0), fixture

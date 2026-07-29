"""Tests for the Design-A two-model wiring: Model 1 boxes + Model 2 classifier
feeding the reconstruction spine via region_detections.frame_from_models.

These use a STUB classifier and a synthetic numpy image, so they run without any
trained weights, torch, or cv2 -- only numpy (a spine-adjacent dep) is needed.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "cv_lab", "scripts"))

import cv_lab.scripts.pipeline.region_detections as rd  # noqa: E402


class StubClassifier:
    """Returns a fixed label per call, recording the crop shapes it saw."""

    def __init__(self, labels):
        self._labels = list(labels)
        self.seen_shapes = []

    def classify(self, crop):
        self.seen_shapes.append(tuple(crop.shape))
        label = self._labels.pop(0) if self._labels else None
        return label, 0.99


def _img(h=200, w=300):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_face_card_attr_filled_from_classifier():
    img = _img()
    rows = [
        {"class": "face_card", "conf": 0.9, "x1": 10, "y1": 20, "x2": 40, "y2": 80},
        {"class": "face_card", "conf": 0.8, "x1": 50, "y1": 20, "x2": 80, "y2": 80},
    ]
    clf = StubClassifier(["As", "Kd"])
    frame = rd.frame_from_models(img, 1.0, rows, classifier=clf, image_name="f0")

    assert frame.width == 300 and frame.height == 200
    assert frame.image == "f0"
    labels = [rd.read_card_label(d) for d in frame.detections]
    assert labels == ["As", "Kd"]
    # classifier saw two crops, each non-empty
    assert len(clf.seen_shapes) == 2
    assert all(s[0] > 0 and s[1] > 0 for s in clf.seen_shapes)


def test_detector_naming_scheme_is_canonicalized():
    """Model 2 could emit the detector's own naming; read_card_label canonicalizes."""
    img = _img()
    rows = [{"class": "face_card", "conf": 0.9, "x1": 10, "y1": 10, "x2": 40, "y2": 70}]
    clf = StubClassifier(["10C"])  # ten of clubs in detector form
    frame = rd.frame_from_models(img, 0.0, rows, classifier=clf)
    assert rd.read_card_label(frame.detections[0]) == "Tc"


def test_non_card_classes_pass_through_without_classifier_call():
    img = _img()
    rows = [
        {"class": "pot_text", "conf": 0.9, "x1": 100, "y1": 100, "x2": 160, "y2": 120,
         "attr": "125.5"},
        {"class": "dealer_button", "conf": 0.9, "x1": 5, "y1": 5, "x2": 20, "y2": 20},
    ]
    clf = StubClassifier(["As"])
    frame = rd.frame_from_models(img, 0.0, rows, classifier=clf)
    # classifier untouched (no face_card)
    assert clf.seen_shapes == []
    pot = next(d for d in frame.detections if d.cls == "pot_text")
    assert rd.read_amount(pot) == pytest.approx(125.5)


def test_unknown_classes_dropped():
    img = _img()
    rows = [
        {"class": "player_name_text", "conf": 0.9, "x1": 0, "y1": 0, "x2": 10, "y2": 10},
        {"class": "face_card", "conf": 0.9, "x1": 10, "y1": 10, "x2": 40, "y2": 70},
    ]
    clf = StubClassifier(["Qs"])
    frame = rd.frame_from_models(img, 0.0, rows, classifier=clf)
    assert [d.cls for d in frame.detections] == ["face_card"]


def test_degenerate_box_yields_none_attr():
    img = _img()
    # zero-area box -> crop is None -> attr stays None, no crash
    rows = [{"class": "face_card", "conf": 0.9, "x1": 50, "y1": 50, "x2": 50, "y2": 50}]
    clf = StubClassifier(["As"])
    frame = rd.frame_from_models(img, 0.0, rows, classifier=clf)
    assert frame.detections[0].attr is None
    assert clf.seen_shapes == []  # never called on an empty crop


def test_pad_expands_crop_within_bounds():
    img = _img(h=100, w=100)
    rows = [{"class": "face_card", "conf": 0.9, "x1": 40, "y1": 40, "x2": 60, "y2": 60}]
    clf = StubClassifier(["As"])
    rd.frame_from_models(img, 0.0, rows, classifier=clf, pad=0.5)
    # box is 20x20, pad 0.5 each side -> +10 each side -> 40x40, still in bounds
    assert clf.seen_shapes[0][:2] == (40, 40)


# --------------------------------------------------------------------------- #
# Anchored card zones + hero-seat cross-check
# --------------------------------------------------------------------------- #
# 1400x1000 (AR 1.400) is close enough to the 1.397 reference that a frame whose
# stack_text boxes sit exactly on SEAT_ANCHORS_BY_CLASS fits with resid 0.0003 and
# raw normalized coords map to reference coords 1:1. A 1000x1000 square frame fits
# at resid 0.0537 -- above ANCHOR_MAX_RESID -- and must not be used here.
FIX_W, FIX_H = 1400, 1000


def _stack_anchor_dets(w=FIX_W, h=FIX_H, seats=None):
    """The 8 stack_text landmark boxes the anchor fit needs, on-anchor."""
    anchors = rd.SEAT_ANCHORS_BY_CLASS["stack_text"]
    out = []
    for seat, (nx, ny) in anchors.items():
        if seats is not None and seat not in seats:
            continue
        x, y = nx * w, ny * h
        out.append({"cls": "stack_text", "conf": 0.9, "attr": None,
                    "xyxy": [x - 40, y - 15, x + 40, y + 15]})
    return out


def _fixture_frame(card_centers, *, anchor_seats=None):
    """One frame with face_cards centered at the given normalized (cx, cy),
    plus the stack_text constellation the table anchor is fitted from."""
    w, h = FIX_W, FIX_H
    dets = _stack_anchor_dets(w, h, anchor_seats)
    for i, (cx, cy) in enumerate(card_centers):
        x, y = cx * w, cy * h
        dets.append({"cls": "face_card", "conf": 0.9, "attr": ["As", "Kd"][i % 2],
                     "xyxy": [x - 20, y - 40, x + 20, y + 40]})
    return rd.frames_from_fixture(
        [{"image": "f0", "time_s": 0.0, "width": w, "height": h, "detections": dets}]
    )[0]


def test_hero_zone_on_seat0_anchor_no_mismatch():
    # Cards where hero hole cards actually render: reference ry 0.6737-0.6874,
    # nearest seat 0's card anchor.
    view = rd.assign_regions(_fixture_frame([(0.49, 0.68), (0.51, 0.68)]))
    assert view["hero"] == ["As", "Kd"]
    assert view["hero_seat_mismatch"] is False
    assert view["anchor_ok"] is True


def test_hero_band_rejects_cards_outside_it():
    # Anchoring makes the old "inside the hero window but nearest another seat"
    # scenario unreachable: ref rx 0.405 is outside REF_HERO_RX (0.440-0.544), so
    # the card is zoned "other" instead of being mis-read as a hero hole card.
    view = rd.assign_regions(_fixture_frame([(0.405, 0.65), (0.41, 0.66)]))
    assert view["hero"] == []
    assert view["hero_seat_mismatch"] is False


def test_hero_seat_mismatch_still_computed_and_false_when_anchored():
    # One on-anchor hero card + one card outside the hero band: the second never
    # votes (it is not in the hero zone at all), so no mismatch is raised.
    view = rd.assign_regions(_fixture_frame([(0.50, 0.68), (0.405, 0.65)]))
    assert view["hero"] == ["As"]
    assert view["hero_seat_mismatch"] is False


def test_no_hero_cards_no_mismatch():
    view = rd.assign_regions(_fixture_frame([]))
    assert view["hero_seat_mismatch"] is False


def test_board_row_is_zoned_board_when_anchored():
    xs = [0.44, 0.47, 0.50, 0.53, 0.56]
    view = rd.assign_regions(_fixture_frame([(x, 0.45) for x in xs]))
    assert len(view["board"]) == 5
    assert view["unanchored_cards"] == 0


def test_unanchorable_frame_drops_cards_and_flags():
    """FAIL CLOSED: too few landmarks -> no zone is guessed, cards are dropped."""
    view = rd.assign_regions(_fixture_frame(
        [(0.44, 0.45), (0.47, 0.45), (0.50, 0.45), (0.49, 0.68), (0.51, 0.68)],
        anchor_seats={0, 4},
    ))
    assert view["anchor_ok"] is False
    assert view["anchor_source"] is None
    assert view["board"] == []
    assert view["hero"] == []
    assert view["villain_cards"] == {}
    assert view["unanchored_cards"] == 5


def test_no_hardcoded_window_fallback():
    """Structural guard: the unanchored rectangles must not creep back in."""
    import ast
    import inspect

    assert not hasattr(rd, "_zone_for_box")
    imported = {
        alias.name
        for node in ast.walk(ast.parse(inspect.getsource(rd)))
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert "_zone_for_box" not in imported
    assert "zone_for_ref" in imported


def _cv_fixture_frame(name):
    import json
    import pathlib

    path = pathlib.Path(__file__).parent / "fixtures" / "cv" / name
    return rd.frames_from_fixture(json.loads(path.read_text(encoding="utf-8")))[0]


# --------------------------------------------------------------------------- #
# Pot vs side pot
# --------------------------------------------------------------------------- #
def test_side_pot_detected_from_real_frame():
    """The detector finds BOTH pot boxes; picking max-confidence between them
    silently discarded one. On this real frame max(conf) happens to pick the main
    pot, but at t=6.0 of the same recording it picks the SIDE pot (0.758 vs
    0.558) and reports 0.2 as the pot of a 240 BB hand. The main/side row split
    makes the choice deterministic and keeps both values."""
    view = rd.assign_regions(_cv_fixture_frame("g0715_side_pot_frame.json"))
    assert view["pot"] == 24.09          # main pot, reference ry 0.372
    assert view["side_pot"] == 0.2       # side pot, reference ry 0.546
    assert view["pots"] == [24.09, 0.2]


def test_duplicate_main_pot_detections_are_not_a_side_pot():
    """Two pot_text boxes are not automatically a main/side pair: the real 07-23
    baseline has frames with duplicate detections of the SAME region, at reference
    ry 0.3412 and 0.3807 -- both in the main band."""
    dets = _stack_anchor_dets()
    for ry, conf in ((0.341, 0.9), (0.381, 0.7)):
        x, y = 0.50 * FIX_W, ry * FIX_H
        dets.append({"cls": "pot_text", "conf": conf, "attr": 12.5,
                     "xyxy": [x - 50, y - 15, x + 50, y + 15]})
    frame = rd.frames_from_fixture([{"image": "f0", "time_s": 0.0, "width": FIX_W,
                                     "height": FIX_H, "detections": dets}])[0]
    view = rd.assign_regions(frame)
    assert view["pot"] == 12.5
    assert view["side_pot"] is None
    assert view["pots"] == [12.5]


def test_off_column_pot_text_is_not_a_side_pot():
    """A pot_text box outside the table's centre column is a misclassified HUD
    element, not a pot.

    STRUCTURAL, not empirical: every genuine pot on the 5 development recordings
    sits at reference rx 0.4916..0.5076 (n=1267) and not one off-column pot_text
    occurs there, so this rule has no positive development evidence and costs
    nothing on measured material. The previous revision of this test transcribed
    a measurement from a 2796x1914 recording -- a geometry that exists only in the
    locked-test and validation splits -- which is a corpus-split violation
    whichever of the two it came from. The coordinates below are synthetic: a box
    at the main-pot row, and one far outside the centre column at the side-pot
    row."""
    dets = _stack_anchor_dets()
    for rx, ry, value, conf in ((0.499, 0.377, 7.0, 0.94), (0.897, 0.461, 219.4, 0.93)):
        x, y = rx * FIX_W, ry * FIX_H
        dets.append({"cls": "pot_text", "conf": conf, "attr": value,
                     "xyxy": [x - 50, y - 15, x + 50, y + 15]})
    frame = rd.frames_from_fixture([{"image": "f0", "time_s": 0.0, "width": FIX_W,
                                     "height": FIX_H, "detections": dets}])[0]
    view = rd.assign_regions(frame)
    assert view["pot"] == 7.0
    assert view["side_pot"] is None


def test_unanchored_frame_reports_main_pot_only():
    """With no table transform the main/side row test is unavailable, so the
    reader must not guess which band a box came from."""
    dets = _stack_anchor_dets(seats={0, 4})
    x, y = 0.50 * FIX_W, 0.35 * FIX_H
    dets.append({"cls": "pot_text", "conf": 0.9, "attr": 8.0,
                 "xyxy": [x - 50, y - 15, x + 50, y + 15]})
    frame = rd.frames_from_fixture([{"image": "f0", "time_s": 0.0, "width": FIX_W,
                                     "height": FIX_H, "detections": dets}])[0]
    view = rd.assign_regions(frame)
    assert view["anchor_ok"] is False
    assert view["pot"] == 8.0
    assert view["side_pot"] is None


# --------------------------------------------------------------------------- #
# Rejection signal A: a community row on screen that nothing zoned as board
# --------------------------------------------------------------------------- #
def test_board_row_missed_fires_when_a_row_is_not_zoned_board():
    """The same real AR 1.750 frame, anchored correctly and then with the
    transform drifted so the community row falls outside the board band. The
    drifted read is exactly the old failure -- an empty board, no warning
    anywhere -- and this is the net that makes it loud."""
    from cv_lab.scripts.pipeline.landmark_anchor import TableAnchor

    frame = _cv_fixture_frame("g0621_ar1750_board_frame.json")
    good = rd.anchor_for_frame(frame)
    assert rd.assign_regions(frame, anchor=good)["board_row_missed"] is False

    drifted = TableAnchor(s=good.s, tx=good.tx, ty=good.ty + 0.10 * good.s * 1470,
                          resid=good.resid, n_points=good.n_points, source="frame")
    view = rd.assign_regions(frame, anchor=drifted)
    assert view["board"] == []
    assert view["board_row_missed"] is True


def test_wide_card_strip_is_not_a_community_row():
    """False-positive guard. A 4-card row spanning reference rx 0.671 is the
    measured g0723a mid-deal top strip, not a board (a 5-card board spans at most
    0.247), so it must not fire the signal even though none of it is zoned board."""
    xs = [0.16, 0.33, 0.66, 0.83]
    view = rd.assign_regions(_fixture_frame([(x, 0.30) for x in xs]))
    assert view["board"] == []
    assert view["board_row_missed"] is False


def test_two_card_hero_row_is_not_a_community_row():
    """A hero pair is a 2-card row: below the 3-card minimum, so no signal."""
    view = rd.assign_regions(_fixture_frame([(0.49, 0.68), (0.51, 0.68)]))
    assert view["hero"] == ["As", "Kd"]
    assert view["board_row_missed"] is False


def test_g0621_board_recovered_from_real_frame():
    """AR 1.750 regression: the 06-21 community row sits at raw cy 0.335-0.338,
    which the old hardcoded board window (0.36 <= cy <= 0.55) rejected outright --
    0 of 435 board detections were zoned board and all 5 community cards were
    re-routed to villain seats. Anchored, the row lands at ref ry ~0.44."""
    import json
    import pathlib

    path = (pathlib.Path(__file__).parent / "fixtures" / "cv"
            / "g0621_ar1750_board_frame.json")
    frame = rd.frames_from_fixture(json.loads(path.read_text(encoding="utf-8")))[0]
    view = rd.assign_regions(frame)
    assert view["anchor_ok"] is True
    assert len(view["board"]) == 5
    assert len(view["hero"]) == 2
    assert view["villain_cards"] == {}
    assert view["unanchored_cards"] == 0


def test_river_card_pushed_out_of_the_band_is_a_partial_row_on_a_real_frame():
    """Round 4, adversary A. The same real AR 1.750 frame, with ONE community card
    displaced just past the board band's upper edge.

    This is the failure ``_board_row_partial`` was written for, on real geometry
    rather than a synthetic row: four of five community cards zone board, the
    surviving four are a complete, board-shaped, fully-zoned row, and every
    surviving count is legal downstream -- so without the net the hand exports as
    a completed FOUR-card turn board at confidence 1.0 with warnings []. Before
    the row-grouping tolerance was re-derived, the displaced card fell outside the
    row entirely and the net reported False.
    """
    import dataclasses

    from cv_lab.scripts.pipeline.landmark_anchor import REF_BOARD_RY, REF_DET_H, zone_for_ref

    frame = _cv_fixture_frame("g0621_ar1750_board_frame.json")
    anchor = rd.anchor_for_frame(frame)
    assert anchor is not None
    base = rd.assign_regions(frame, anchor=anchor)
    assert len(base["board"]) == 5
    assert base["board_row_partial"] is False and base["board_row_missed"] is False

    board = []
    for det in frame.detections:
        if det.cls != "face_card":
            continue
        rx, ry = anchor.to_ref(*rd._center_px(det))
        if zone_for_ref(rx, ry) == "board":
            board.append((rx, ry, det))
    board.sort()
    _rx, ry, river = board[-1]
    dy = (REF_BOARD_RY[1] - ry + 0.0002) * anchor.s * REF_DET_H
    shifted = dataclasses.replace(
        river, xyxy=(river.xyxy[0], river.xyxy[1] + dy, river.xyxy[2], river.xyxy[3] + dy)
    )
    dets = [shifted if d is river else d for d in frame.detections]
    view = rd.assign_regions(dataclasses.replace(frame, detections=dets), anchor=anchor)

    assert len(view["board"]) == 4          # the silent, legal-looking outcome
    assert view["board_row_partial"] is True
    assert view["board_row_missed"] is False


def test_hero_seat_confirmed_requires_positive_evidence_not_just_silence():
    """Round 4, adversary C: the producer of ``hero_seat_confirmed`` had no test.
    Reducing it to ``not hero_seat_mismatch`` left the whole CV suite green -- the
    only two tests naming the field pass it in as a literal and exercise the
    CONSUMER (``layout_supported`` in the export).

    The guard exists for a measured failure: under a vertical stretch, hero's own
    hole cards leave the hero band and are re-attributed to villain seats as
    showdown reveals. ``hero_seat_mismatch`` is a MAJORITY VOTE over the hero
    zone's cards, so with the zone emptied it evaluates ``0 > 0`` and reports
    False -- and every other net stays silent too: the board still zones 5/5, the
    anchor residual stays in tolerance, no row net fires. An empty vote list is
    the single remaining signal, and reading it as "confirmed" asserts a layout
    the pipeline never observed.
    """
    frame = _cv_fixture_frame("g0621_ar1750_board_frame.json")
    anchor = rd.anchor_for_frame(frame)
    healthy = rd.assign_regions(frame, anchor=anchor)
    assert healthy["hero_seat_votes"] > 0
    assert healthy["hero_seat_confirmed"] is True
    assert healthy["hero_seat_mismatch"] is False

    # Same frame with the hero zone emptied: no mismatch, and no evidence either.
    import dataclasses

    from cv_lab.scripts.pipeline.landmark_anchor import zone_for_ref

    kept = [d for d in frame.detections
            if d.cls != "face_card"
            or zone_for_ref(*anchor.to_ref(*rd._center_px(d))) != "hero"]
    view = rd.assign_regions(dataclasses.replace(frame, detections=kept), anchor=anchor)
    assert view["hero"] == []
    assert view["hero_seat_votes"] == 0
    assert view["hero_seat_mismatch"] is False, "the majority vote is vacuous here"
    assert view["hero_seat_confirmed"] is False, (
        "confirmation must rest on a seat that was actually observed")


# --------------------------------------------------------------------------- #
# Phase 6: UNKNOWN as a first-class value, propagated hop by hop.
#
# The reader's contract now returns a number only when the read is PROVABLY
# unambiguous and a NAMED refusal otherwise. These pin the channel that carries
# that refusal downstream, at each hop where it used to be flattened into
# "there is no number here".
# --------------------------------------------------------------------------- #
def _amount_frame(dets):
    """One 2054x1470 frame: the 8 stack_text landmarks plus `dets` overrides."""
    out = []
    for seat, (nx, ny) in rd.SEAT_ANCHORS_BY_CLASS["stack_text"].items():
        override = dets.get(seat, ("value", 100.0))
        attr = override[1] if override[0] == "value" else None
        source = override[1] if override[0] == "unknown" else None
        out.append(rd.Detection("stack_text", 0.9,
                                (nx * 2054 - 40, ny * 1470 - 20,
                                 nx * 2054 + 40, ny * 1470 + 20),
                                attr, attr_source=source))
    return rd.Frame("f0", 0.0, 2054, 1470, out)


def test_amount_state_separates_a_refusal_from_an_absent_box():
    """THE THREE-WAY DISTINCTION. ``read_amount`` returns None for all three, which
    is why it must not be used to decide anything: a refused read is UNKNOWN, an
    absent one was never attempted, and 0.0 is a genuine all-in seat."""
    box = (0.0, 0.0, 10.0, 10.0)
    proven = rd.Detection("stack_text", 0.9, box, 0.0, attr_source="integer")
    refused = rd.Detection("stack_text", 0.9, box, None, attr_source="suffix_not_bb")
    absent = rd.Detection("stack_text", 0.9, box, None)
    junk = rd.Detection("stack_text", 0.9, box, "??")

    assert rd.amount_state(proven) == ("value", 0.0)
    assert rd.amount_state(refused) == ("unknown", "suffix_not_bb")
    assert rd.amount_state(absent) == ("absent", "")
    assert rd.amount_state(junk) == ("unknown", rd.AMOUNT_UNREADABLE_ATTR)
    # ... and the value-only projection cannot tell the middle two apart.
    assert rd.read_amount(refused) is rd.read_amount(absent) is None
    assert rd.read_amount(proven) == 0.0, "a proven zero is a value, not a refusal"


def test_a_refused_stack_is_carried_not_dropped():
    """`stacks` omits a refused seat exactly as it omits a seat with no box, so
    the refusal was INVISIBLE downstream (note 10: "unknown is neither zero nor
    unknown, it is invisible"). The reason now travels alongside."""
    from cv_lab.scripts.pipeline.build_yolo_hand_timeline import _frame_state

    frame = _amount_frame({3: ("unknown", "integer_over_decimal_band")})
    view = rd.assign_regions(frame)
    assert view["seats"][3]["stack"] is None
    assert view["seats"][3]["stack_unknown"] == "integer_over_decimal_band"
    assert view["stack_unknown"] == {3: "integer_over_decimal_band"}

    state = _frame_state(frame)
    assert 3 not in state["stacks"]
    assert state["stacks_unknown"] == {3: "integer_over_decimal_band"}
    assert state["amounts_unknown_by_code"] == {"integer_over_decimal_band": 1}


def test_a_refusal_is_distinguishable_from_an_absent_box_at_every_hop():
    """The channel existed and was UNREAD, and one branch destroyed it: the
    Detection layer set ``attr_source`` only inside ``if detail is not None``, so
    a run with no calibrated template bank produced attr=None / attr_source=None
    -- byte-identical to a fixture detection carrying no amount at all."""
    from cv_lab.scripts.pipeline import ocr_readers as ocr
    from cv_lab.scripts.pipeline.build_yolo_hand_timeline import _frame_state

    absent = _frame_state(_amount_frame({3: ("absent", None)}))
    assert 3 not in absent["stacks"]
    assert absent["stacks_unknown"] == {}, "an absent box is not a refusal"

    saved = dict(ocr._CACHE)
    ocr._CACHE[str(ocr.DEFAULT_TEMPLATE_PATH)] = None      # no bank calibrated
    try:
        frame = rd.frame_from_models(
            _img(200, 300), 0.0,
            [{"class": "stack_text", "conf": 0.9, "x1": 10, "y1": 10,
              "x2": 60, "y2": 40}],
            classifier=None)
    finally:
        ocr._CACHE.clear()
        ocr._CACHE.update(saved)
    det = frame.detections[0]
    assert det.attr is None
    assert det.attr_source == rd.AMOUNT_READER_UNAVAILABLE
    assert rd.amount_state(det) == ("unknown", "reader_unavailable")


def test_a_stack_box_conflict_does_not_erase_the_readers_refusal_code():
    """Two stack_text boxes can map to one seat and DISAGREE (3 of 1309 real
    frames). The conflict used to overwrite the read's provenance with the word
    "conflict", so a seat the reader had positively refused became
    indistinguishable from a seat whose two boxes contradicted each other."""
    nx, ny = rd.SEAT_ANCHORS_BY_CLASS["stack_text"][3]
    frame = _amount_frame({})
    frame.detections.append(rd.Detection(
        "stack_text", 0.9,
        (nx * 2054 - 30, ny * 1470 - 10, nx * 2054 + 50, ny * 1470 + 30),
        215.0, attr_source="dot"))
    view = rd.assign_regions(frame)
    assert view["stack_conflicts"] == 1
    assert view["seats"][3]["stack"] is None
    assert view["seats"][3]["stack_unknown"] == rd.AMOUNT_STACK_BOXES_DISAGREE
    assert view["amounts_unknown_by_code"] == {"stack_boxes_disagree": 1}

    # ...and a seat the READER refused keeps the reader's own code.
    refused = rd.assign_regions(_amount_frame({5: ("unknown", "run_clipped")}))
    assert refused["seats"][5]["stack_unknown"] == "run_clipped"
    assert refused["stack_conflicts"] == 0


def test_a_refused_pot_is_not_the_same_fact_as_no_pot_box():
    """A refused pot must reach the spine as UNKNOWN with a reason. It used to be
    the same None as "no pot_text was detected", which is what let a settlement
    scan substitute a stale pot for it."""
    from cv_lab.scripts.pipeline.build_yolo_hand_timeline import _frame_state

    frame = _amount_frame({})
    frame.detections.append(rd.Detection(
        "pot_text", 0.9, (0.5 * 2054 - 60, 0.32 * 1470 - 20,
                          0.5 * 2054 + 60, 0.32 * 1470 + 20),
        None, attr_source="separator_unreconciled"))
    view = rd.assign_regions(frame)
    assert view["pot"] is None
    assert view["pot_unknown"] == "separator_unreconciled"
    assert _frame_state(frame)["pot_unknown"] == "separator_unreconciled"

    # A pot read of exactly 0 is a dropout on a region the client always renders,
    # so it is unknown too -- named, not silently discarded.
    zeroed = _amount_frame({})
    zeroed.detections.append(rd.Detection(
        "pot_text", 0.9, (0.5 * 2054 - 60, 0.32 * 1470 - 20,
                          0.5 * 2054 + 60, 0.32 * 1470 + 20),
        0.0, attr_source="integer"))
    state = _frame_state(zeroed)
    assert state["pot"] is None
    assert state["pot_unknown"] == "pot_zero_impossible"

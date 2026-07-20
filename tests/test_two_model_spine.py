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

import region_detections as rd  # noqa: E402


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
# Hero-seat cross-check: hero zone must agree with the seat-0 convention
# --------------------------------------------------------------------------- #
def _fixture_frame(card_centers):
    """One frame with face_cards centered at the given normalized (cx, cy)."""
    w = h = 1000
    dets = []
    for i, (cx, cy) in enumerate(card_centers):
        x, y = cx * w, cy * h
        dets.append({"cls": "face_card", "conf": 0.9, "attr": ["As", "Kd"][i % 2],
                     "xyxy": [x - 20, y - 40, x + 20, y + 40]})
    return rd.frames_from_fixture(
        [{"image": "f0", "time_s": 0.0, "width": w, "height": h, "detections": dets}]
    )[0]


def test_hero_zone_on_seat0_anchor_no_mismatch():
    # Cards where hero hole cards actually render: bottom-center, nearest
    # seat 0's card anchor.
    view = rd.assign_regions(_fixture_frame([(0.49, 0.80), (0.53, 0.80)]))
    assert view["hero"] == ["As", "Kd"]
    assert view["hero_seat_mismatch"] is False


def test_hero_zone_near_other_seat_flags_mismatch():
    # Still inside the hero zone (cx 0.40-0.58, cy >= 0.64) but nearer seat 1's
    # card anchor (0.194, 0.623) than seat 0's (0.500, 0.860): the layout has
    # drifted, so the seat-0 hero convention is flagged.
    view = rd.assign_regions(_fixture_frame([(0.405, 0.65), (0.41, 0.66)]))
    assert view["hero_seat_mismatch"] is True


def test_single_flapped_card_does_not_flag_mismatch():
    # Majority vote: one on-anchor card + one off-anchor card -> no warning.
    view = rd.assign_regions(_fixture_frame([(0.50, 0.82), (0.405, 0.65)]))
    assert view["hero_seat_mismatch"] is False


def test_no_hero_cards_no_mismatch():
    view = rd.assign_regions(_fixture_frame([]))
    assert view["hero_seat_mismatch"] is False

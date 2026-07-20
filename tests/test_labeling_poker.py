import struct

import pytest

from labeling_poker.app import create_app
from labeling_poker.config import normalize_card_label
from labeling_poker.db import connect, get_annotations, get_status, next_matching, progress, save_annotations, seek, sync_files
from labeling_poker.export import image_size, split_ids, write_dataset
from labeling_poker.label_audit import (
    audit_boxes,
    audit_unlabeled_frames,
    filter_card_label_suspects,
    write_suspect_queue,
)


def write_png(path, width=100, height=50):
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\x0dIHDR" + struct.pack(">II", width, height) + b"\x08\x02\x00\x00\x00")


def test_db_save_replace_and_progress(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    write_png(images / "002.png")
    write_png(images / "001.png")
    connection = connect(tmp_path / "labels.sqlite3")
    assert sync_files(connection, images) == ["001", "002"]
    save_annotations(connection, "001", "labeled", [{"class": "face_card", "x1": 2, "y1": 3, "x2": 40, "y2": 20}])
    assert get_status(connection, "001") == "labeled"
    assert len(get_annotations(connection, "001")) == 1
    assert get_annotations(connection, "001")[0]["label"] is None
    save_annotations(connection, "001", "clean", [])
    assert get_annotations(connection, "001") == []
    assert progress(connection) == {"total": 2, "labeled": 0, "clean": 1, "duplicate": 0, "undecided": 1}


def test_duplicate_is_excluded_without_deleting_source(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    write_png(images / "001.png")
    connection = connect(tmp_path / "labels.sqlite3")
    sync_files(connection, images)
    save_annotations(connection, "001", "duplicate", [])
    assert get_status(connection, "001") == "duplicate"
    assert images.joinpath("001.png").is_file()
    assert progress(connection)["undecided"] == 0


def test_status_filtered_navigation_keeps_saved_labels_browseable(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    for file_id in ("001", "002", "003", "004"):
        write_png(images / f"{file_id}.png")
    connection = connect(tmp_path / "labels.sqlite3")
    sync_files(connection, images)
    save_annotations(connection, "001", "labeled", [])
    save_annotations(connection, "002", "clean", [])
    save_annotations(connection, "003", "labeled", [])

    assert next_matching(connection, "labeled") == "001"
    assert seek(connection, "001", "next", "labeled") == "003"
    assert seek(connection, "003", "prev", "labeled") == "001"
    # Once relabeled as clean, navigation still advances from its original spot.
    save_annotations(connection, "001", "clean", [])
    assert next_matching(connection, "labeled", current_id="001") == "003"


def test_index_defaults_to_two_model_validation_queue(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    write_png(images / "001.png")
    client = create_app(tmp_path / "labels.sqlite3", images).test_client()
    response = client.get("/")
    assert response.status_code == 302
    assert response.headers["Location"].endswith("/?queue=two_model_validation")
    response = client.get("/?view=labeled_sus")
    assert response.status_code == 302
    assert "queue=two_model_validation" in response.headers["Location"]
    assert "view=labeled_sus" in response.headers["Location"]


def test_api_can_browse_labeled_images(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    for file_id in ("001", "002", "003"):
        write_png(images / f"{file_id}.png")
    db_path = tmp_path / "labels.sqlite3"
    client = create_app(db_path, images).test_client()
    client.get("/", follow_redirects=True)
    with connect(db_path) as connection:
        save_annotations(connection, "001", "labeled", [])
        save_annotations(connection, "002", "clean", [])
        save_annotations(connection, "003", "labeled", [])

    assert client.get("/api/next?status=labeled").get_json()["item"]["id"] == "001"
    assert client.get("/api/seek?dir=next&id=001&status=labeled").get_json()["item"]["id"] == "003"


def test_audit_flags_duplicate_card_labels():
    reasons = audit_boxes([
        {"class": "face_card", "label": "As", "x1": 10, "y1": 10, "x2": 40, "y2": 70},
        {"class": "face_card", "label": "As", "x1": 50, "y1": 10, "x2": 80, "y2": 70},
        {"class": "dealer_button", "label": None, "x1": 90, "y1": 90, "x2": 110, "y2": 110},
    ])
    assert any(reason.startswith("duplicate_card_labels:") for reason in reasons)


def test_audit_flags_any_unlabeled_face_card():
    reasons = audit_boxes([
        {"class": "face_card", "label": None, "x1": 10, "y1": 10, "x2": 40, "y2": 70},
    ])
    assert any(reason.startswith("face_cards_missing_rank:") for reason in reasons)


def test_filter_card_label_suspects_keeps_rank_suit_issues_only():
    suspects = [
        {
            "id": "geo",
            "reasons": ["multi_dealer_button:2", "multi_pot_text:3"],
            "severity": 110,
            "box_count": 3,
            "updated_at": "t",
        },
        {
            "id": "cards",
            "reasons": ["duplicate_card_labels:As", "multi_dealer_button:2"],
            "severity": 170,
            "box_count": 2,
            "updated_at": "t",
        },
        {
            "id": "face_geo",
            "reasons": ["weird_face_aspect:2.0"],
            "severity": 25,
            "box_count": 1,
            "updated_at": "t",
        },
    ]
    filtered = filter_card_label_suspects(suspects)
    assert [item["id"] for item in filtered] == ["cards", "face_geo"]
    assert filtered[0]["reasons"] == ["duplicate_card_labels:As"]
    assert filtered[1]["reasons"] == ["weird_face_aspect:2.0"]


def test_api_can_browse_labeled_sus_queue(tmp_path, monkeypatch):
    images = tmp_path / "images"
    images.mkdir()
    for file_id in ("001", "002", "003"):
        write_png(images / f"{file_id}.png")
    priority = tmp_path / "priority"
    write_suspect_queue(
        [
            {
                "id": "003",
                "updated_at": "2026-01-03T00:00:00+00:00",
                "reasons": ["duplicate_card_labels:As"],
                "severity": 100,
                "box_count": 2,
            },
            {
                "id": "001",
                "updated_at": "2026-01-01T00:00:00+00:00",
                "reasons": ["multi_dealer_button:2"],
                "severity": 70,
                "box_count": 3,
            },
        ],
        priority_dir=priority,
    )
    monkeypatch.setenv("POKER_LABELER_PRIORITY_DIR", str(priority))
    db_path = tmp_path / "labels.sqlite3"
    client = create_app(db_path, images).test_client()
    client.get("/", follow_redirects=True)
    with connect(db_path) as connection:
        save_annotations(connection, "001", "labeled", [{"class": "face_card", "label": "As", "x1": 1, "y1": 2, "x2": 20, "y2": 30}])
        save_annotations(connection, "002", "labeled", [{"class": "face_card", "label": "Kd", "x1": 1, "y1": 2, "x2": 20, "y2": 30}])
        save_annotations(connection, "003", "labeled", [{"class": "face_card", "label": "As", "x1": 1, "y1": 2, "x2": 20, "y2": 30}])

    first = client.get("/api/next?status=labeled_sus").get_json()["item"]
    assert first["id"] == "003"
    assert first["sus_reasons"] == ["duplicate_card_labels:As"]
    assert client.get("/api/seek?dir=next&id=003&status=labeled_sus").get_json()["item"]["id"] == "001"
    progress_payload = client.get("/api/progress?status=labeled_sus").get_json()
    assert progress_payload["labeled_sus"] == 2
    review = client.get("/manual-review?view=labeled_sus")
    assert review.status_code == 200
    assert b"Labeled sus" in review.data
    assert b"003" in review.data
    assert b"duplicate_card_labels:As" in review.data


def test_audit_unlabeled_flags_duplicate_prediction_cards(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    write_png(images / "001.png")
    write_png(images / "002.png")
    db_path = tmp_path / "labels.sqlite3"
    with connect(db_path) as connection:
        sync_files(connection, images)
        save_annotations(
            connection,
            "002",
            "labeled",
            [{"class": "face_card", "label": "As", "x1": 1, "y1": 2, "x2": 20, "y2": 30}],
        )
        suspects = audit_unlabeled_frames(
            connection,
            two_model_cache={
                "001": [
                    {"class": "face_card", "label": "As", "x1": 10, "y1": 10, "x2": 40, "y2": 70},
                    {"class": "face_card", "label": "As", "x1": 50, "y1": 10, "x2": 80, "y2": 70},
                ],
                "002": [
                    {"class": "face_card", "label": "Kd", "x1": 10, "y1": 10, "x2": 40, "y2": 70},
                    {"class": "face_card", "label": "Kd", "x1": 50, "y1": 10, "x2": 80, "y2": 70},
                ],
            },
        )
    assert [item["id"] for item in suspects] == ["001"]
    assert any(reason.startswith("duplicate_card_labels:") for reason in suspects[0]["reasons"])


def test_api_can_browse_unlabeled_sus_queue(tmp_path, monkeypatch):
    images = tmp_path / "images"
    images.mkdir()
    for file_id in ("001", "002", "003"):
        write_png(images / f"{file_id}.png")
    priority = tmp_path / "priority"
    write_suspect_queue(
        [
            {
                "id": "003",
                "updated_at": None,
                "reasons": ["duplicate_card_labels:As"],
                "severity": 100,
                "box_count": 2,
            },
            {
                "id": "001",
                "updated_at": None,
                "reasons": ["empty_prediction"],
                "severity": 75,
                "box_count": 0,
            },
        ],
        priority_dir=priority,
        queue_name="unlabeled_sus",
    )
    cache = {
        "source": "test",
        "items": {
            "001": [],
            "003": [
                {"class": "face_card", "label": "As", "x1": 1, "y1": 2, "x2": 20, "y2": 30},
                {"class": "face_card", "label": "As", "x1": 21, "y1": 2, "x2": 40, "y2": 30},
            ],
        },
    }
    cache_path = tmp_path / "two_model_validation.json"
    cache_path.write_text(__import__("json").dumps(cache), encoding="utf-8")
    monkeypatch.setenv("POKER_LABELER_PRIORITY_DIR", str(priority))
    monkeypatch.setenv("POKER_LABELER_TWO_MODEL_CACHE", str(cache_path))
    db_path = tmp_path / "labels.sqlite3"
    client = create_app(db_path, images).test_client()
    client.get("/", follow_redirects=True)

    first = client.get("/api/next?status=unlabeled_sus").get_json()["item"]
    assert first["id"] == "003"
    assert first["status"] == "undecided"
    assert first["sus_reasons"] == ["duplicate_card_labels:As"]
    assert client.get("/api/seek?dir=next&id=003&status=unlabeled_sus").get_json()["item"]["id"] == "001"
    progress_payload = client.get("/api/progress?status=unlabeled_sus").get_json()
    assert progress_payload["unlabeled_sus"] == 2
    review = client.get("/manual-review?view=unlabeled_sus")
    assert review.status_code == 200
    assert b"Unlabeled sus" in review.data
    assert b"003" in review.data


def test_api_labeled_sus_done_clears_queue_keeps_labels(tmp_path, monkeypatch):
    images = tmp_path / "images"
    images.mkdir()
    write_png(images / "001.png")
    priority = tmp_path / "priority"
    write_suspect_queue(
        [
            {
                "id": "001",
                "updated_at": "2026-01-01T00:00:00+00:00",
                "reasons": ["face_cards_missing_rank:1"],
                "severity": 50,
                "box_count": 1,
            }
        ],
        priority_dir=priority,
    )
    monkeypatch.setenv("POKER_LABELER_PRIORITY_DIR", str(priority))
    db_path = tmp_path / "labels.sqlite3"
    client = create_app(db_path, images).test_client()
    client.get("/", follow_redirects=True)
    with connect(db_path) as connection:
        save_annotations(
            connection,
            "001",
            "labeled",
            [{"class": "face_card", "label": "As", "x1": 1, "y1": 2, "x2": 20, "y2": 30}],
        )

    assert client.get("/api/progress?status=labeled_sus").get_json()["labeled_sus"] == 1
    done = client.post("/api/labeled-sus/done").get_json()
    assert done["cleared"] == 1
    assert done["ids"] == ["001"]
    assert (priority / "labeled_sus.txt").read_text(encoding="utf-8") == ""
    assert client.get("/api/progress?status=labeled_sus").get_json()["labeled_sus"] == 0
    assert client.get("/api/next?status=labeled_sus").get_json()["item"] is None
    with connect(db_path) as connection:
        assert get_status(connection, "001") == "labeled"
        assert get_annotations(connection, "001")[0]["label"] == "As"


def test_api_two_model_bootstrap_is_review_only(tmp_path, monkeypatch):
    images = tmp_path / "images"
    images.mkdir()
    write_png(images / "001.png")
    db_path = tmp_path / "labels.sqlite3"
    client = create_app(db_path, images).test_client()
    client.get("/", follow_redirects=True)

    def fake_predict(_path):
        return [{"class": "face_card", "label": "As", "x1": 1, "y1": 2, "x2": 20, "y2": 30}]

    monkeypatch.setattr("labeling_poker.app.predict_two_model", fake_predict)
    response = client.get("/api/bootstrap/001?source=two_model")
    assert response.status_code == 200
    assert response.get_json()["boxes"][0]["label"] == "As"
    with connect(db_path) as connection:
        assert get_status(connection, "001") == "undecided"


def test_api_uses_cached_two_model_predictions_without_writing_labels(tmp_path, monkeypatch):
    images = tmp_path / "images"
    images.mkdir()
    write_png(images / "001.png")
    db_path = tmp_path / "labels.sqlite3"
    cache = tmp_path / "two_model_validation.json"
    cache.write_text('{"items": {"001": [{"class": "face_card", "label": "Kd", "x1": 1, "y1": 2, "x2": 20, "y2": 30}]}}')
    monkeypatch.setenv("POKER_LABELER_TWO_MODEL_CACHE", str(cache))
    client = create_app(db_path, images).test_client()
    client.get("/", follow_redirects=True)
    monkeypatch.setattr("labeling_poker.app.predict_two_model", lambda _path: pytest.fail("cache was not used"))

    response = client.get("/api/bootstrap/001?source=two_model")
    assert response.status_code == 200
    assert response.get_json()["cached"] is True
    assert response.get_json()["boxes"][0]["label"] == "Kd"
    with connect(db_path) as connection:
        assert get_status(connection, "001") == "undecided"


def test_manual_review_lists_saved_annotations(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    write_png(images / "001.png")
    db_path = tmp_path / "labels.sqlite3"
    client = create_app(db_path, images).test_client()
    client.get("/", follow_redirects=True)
    with connect(db_path) as connection:
        save_annotations(connection, "001", "labeled", [{"class": "face_card", "label": "As", "x1": 1, "y1": 2, "x2": 20, "y2": 30}])

    response = client.get("/manual-review?view=labeled")
    assert response.status_code == 200
    assert b"Manual review" in response.data
    assert b"001" in response.data
    assert b"face_card" in response.data


def test_model1_labeled_queue_starts_newest_and_wraps_forward(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    for file_id in ("001", "002", "003", "outside_queue"):
        write_png(images / f"{file_id}.png")
    connection = connect(tmp_path / "labels.sqlite3")
    sync_files(connection, images)
    for file_id in ("001", "002", "003", "outside_queue"):
        save_annotations(connection, file_id, "labeled", [])
    connection.executemany(
        "UPDATE status SET updated_at = ? WHERE file_id = ?",
        [("2026-01-01T00:00:00+00:00", "001"), ("2026-01-02T00:00:00+00:00", "002"),
         ("2026-01-03T00:00:00+00:00", "003"), ("2026-01-04T00:00:00+00:00", "outside_queue")],
    )
    connection.commit()

    options = {"priority_only": True, "order_by_updated_at": True, "wrap_next": True}
    assert next_matching(connection, "labeled", ["001", "002", "003"], start_with_latest=True, **options) == "003"
    assert seek(connection, "003", "prev", "labeled", ["001", "002", "003"], **options) == "002"
    assert seek(connection, "003", "next", "labeled", ["001", "002", "003"], **options) == "001"


def test_sync_files_supports_nested_image_directories(tmp_path):
    images = tmp_path / "images"
    (images / "train").mkdir(parents=True)
    write_png(images / "train" / "nested.png")
    connection = connect(tmp_path / "labels.sqlite3")
    assert sync_files(connection, images) == ["nested"]
    assert connection.execute("SELECT path FROM files WHERE id = 'nested'").fetchone()[0] == "train/nested.png"


def test_sync_files_prunes_missing_undecided_images(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    write_png(images / "keep.png")
    write_png(images / "gone.png")
    connection = connect(tmp_path / "labels.sqlite3")
    sync_files(connection, images)
    save_annotations(connection, "keep", "labeled", [{"class": "face_card", "x1": 1, "y1": 1, "x2": 20, "y2": 20}])
    (images / "gone.png").unlink()
    # Also leave a labeled row whose file vanishes — that should be kept.
    write_png(images / "labeled_gone.png")
    sync_files(connection, images)
    save_annotations(
        connection,
        "labeled_gone",
        "labeled",
        [{"class": "face_card", "x1": 1, "y1": 1, "x2": 20, "y2": 20}],
    )
    (images / "labeled_gone.png").unlink()
    assert sync_files(connection, images) == ["keep"]
    ids = {row["id"] for row in connection.execute("SELECT id FROM files")}
    assert ids == {"keep", "labeled_gone"}
    assert next_matching(connection, "undecided") is None


def test_export_normalizes_boxes_and_includes_clean_negative(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    write_png(images / "001.png", 100, 50)
    write_png(images / "002.png", 100, 50)
    connection = connect(tmp_path / "labels.sqlite3")
    sync_files(connection, images)
    save_annotations(connection, "001", "labeled", [{"class": "face_card", "x1": 10, "y1": 5, "x2": 60, "y2": 25}])
    save_annotations(connection, "002", "clean", [])
    output = tmp_path / "dataset"
    assert write_dataset(tmp_path / "labels.sqlite3", images, output) == {"train": 1, "val": 0, "test": 1}
    assert (output / "labels/train/001.txt").read_text() == "0 0.35000000 0.30000000 0.50000000 0.40000000\n"
    assert (output / "labels/test/002.txt").read_text() == ""
    assert image_size(images / "001.png") == (100, 50)


def test_export_can_exclude_card_only_frames_for_region_detector(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    for file_id in ("001", "002", "003"):
        write_png(images / f"{file_id}.png", 100, 50)
    db_path = tmp_path / "labels.sqlite3"
    connection = connect(db_path)
    sync_files(connection, images)
    save_annotations(connection, "001", "labeled", [{"class": "face_card", "x1": 1, "y1": 1, "x2": 20, "y2": 20}])
    save_annotations(connection, "002", "labeled", [
        {"class": "face_card", "x1": 1, "y1": 1, "x2": 20, "y2": 20},
        {"class": "pot_text", "x1": 30, "y1": 1, "x2": 60, "y2": 20},
    ])
    save_annotations(connection, "003", "clean", [])

    output = tmp_path / "dataset"
    assert write_dataset(db_path, images, output, exclude_card_only=True) == {"train": 1, "val": 0, "test": 1}
    exported = {path.stem for path in (output / "labels/train").glob("*.txt")} | {path.stem for path in (output / "labels/test").glob("*.txt")}
    assert exported == {"002", "003"}


def test_normalize_card_label_canonicalizes_and_validates():
    # Picker form passes through; detector form is canonicalized to rank+suit.
    assert normalize_card_label("face_card", "Kd") == "Kd"
    assert normalize_card_label("face_card", "KD") == "Kd"
    assert normalize_card_label("face_card", "10C") == "Tc"
    assert normalize_card_label("face_card", " ah ") == "Ah"
    # joker and empty collapse to no label; non-card classes never carry one.
    assert normalize_card_label("face_card", "joker") is None
    assert normalize_card_label("face_card", "") is None
    assert normalize_card_label("face_card", None) is None
    assert normalize_card_label("pot_text", "Kd") is None
    with pytest.raises(ValueError):
        normalize_card_label("face_card", "ZZ")
    with pytest.raises(ValueError):
        normalize_card_label("face_card", "Kx")


def test_api_annotate_stores_and_validates_card_label(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    write_png(images / "001.png")
    db_path = tmp_path / "labels.sqlite3"
    client = create_app(db_path, images).test_client()
    client.get("/", follow_redirects=True)  # sync files into the db

    ok = client.post("/api/annotate", json={
        "id": "001", "status": "labeled",
        "boxes": [{"class": "face_card", "label": "10C", "x1": 2, "y1": 3, "x2": 40, "y2": 20}],
    })
    assert ok.status_code == 200
    with connect(db_path) as connection:
        assert get_annotations(connection, "001")[0]["label"] == "Tc"

    bad = client.post("/api/annotate", json={
        "id": "001", "status": "labeled",
        "boxes": [{"class": "face_card", "label": "ZZ", "x1": 2, "y1": 3, "x2": 40, "y2": 20}],
    })
    assert bad.status_code == 400


def test_split_is_deterministic():
    assert split_ids(["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]) == {
        "train": ["01", "02", "03", "04", "05", "06", "07", "08"],
        "val": ["09"],
        "test": ["10"],
    }

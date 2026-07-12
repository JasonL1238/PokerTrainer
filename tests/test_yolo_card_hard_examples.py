import csv
from pathlib import Path

from cv_lab.scripts.mine_yolo_card_hard_examples import mine_hard_examples


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_report(path: Path) -> dict[str, dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return {row["image"]: row for row in csv.DictReader(f)}


MANIFEST_FIELDS = ["split", "video", "frame", "time_s", "image", "label", "detections", "min_conf", "labels"]
DETECTION_FIELDS = [
    "split",
    "video",
    "frame",
    "time_s",
    "image",
    "label",
    "detection_index",
    "pred_class_id",
    "pred_label",
    "conf",
    "x0",
    "y0",
    "x1",
    "y1",
    "correct_label",
    "action",
]


def _row(image: str, det_i: int, label: str, conf: float, xyxy: tuple[int, int, int, int]) -> dict:
    x0, y0, x1, y1 = xyxy
    frame_i = int(image[1:])
    return {
        "split": "train",
        "video": "session.mp4",
        "frame": frame_i,
        "time_s": f"{float(frame_i):.1f}",
        "image": f"images/train/{image}.jpg",
        "label": f"labels/train/{image}.txt",
        "detection_index": det_i,
        "pred_class_id": det_i,
        "pred_label": label,
        "conf": f"{conf:.3f}",
        "x0": x0,
        "y0": y0,
        "x1": x1,
        "y1": y1,
        "correct_label": "",
        "action": "",
    }


def test_mine_hard_examples_flags_geometry_confidence_duplicates_and_missing(tmp_path):
    dataset = tmp_path / "dataset"
    frames = [
        {
            "split": "train",
            "video": "session.mp4",
            "frame": i,
            "time_s": f"{float(i):.1f}",
            "image": f"images/train/f{i}.jpg",
            "label": f"labels/train/f{i}.txt",
            "detections": 0,
            "min_conf": "",
            "labels": "",
        }
        for i in range(4)
    ]
    corrections = [
        _row("f0", 0, "AS", 0.20, (420, 700, 520, 850)),
        _row("f0", 1, "KS", 0.80, (430, 710, 530, 860)),
        _row("f1", 0, "2C", 0.90, (380, 430, 450, 540)),
        _row("f2", 0, "7D", 0.90, (850, 100, 920, 220)),
    ]
    corrections[1]["correct_label"] = "AS"

    _write_csv(dataset / "manifest.csv", MANIFEST_FIELDS, frames)
    _write_csv(dataset / "detections.csv", DETECTION_FIELDS, corrections)
    _write_csv(dataset / "corrections.csv", DETECTION_FIELDS, corrections)
    _write_csv(dataset / "missing_labels.csv", ["image", "note"], [{"image": "images/train/f3.jpg", "note": "board card absent"}])
    (dataset / "review").mkdir(parents=True)
    (dataset / "review" / "f0.jpg").write_bytes(b"not a real jpeg, just a link target")
    before = (dataset / "corrections.csv").read_text(encoding="utf-8")

    summary = mine_hard_examples(
        dataset,
        dataset / "hard_examples",
        image_width=1000,
        image_height=1000,
        low_conf=0.35,
        overlap_iou=0.50,
        write_html=True,
    )

    rows = _read_report(dataset / "hard_examples" / "hard_examples.csv")
    f0_issues = rows["images/train/f0.jpg"]["issue_types"].split(";")
    assert "low_confidence" in f0_issues
    assert "duplicate_corrected_label" in f0_issues
    assert "overlapping_boxes" in f0_issues
    assert rows["images/train/f0.jpg"]["review_image"] == "review/f0.jpg"
    assert "partial_board_count" in rows["images/train/f1.jpg"]["issue_types"].split(";")
    assert "missing_labels_entry" in rows["images/train/f3.jpg"]["issue_types"].split(";")
    assert summary["hard_frames"] == 4
    assert summary["source_rows"]["effective"] == "corrections.csv"
    assert (dataset / "corrections.csv").read_text(encoding="utf-8") == before
    assert (dataset / "hard_examples" / "index.html").exists()


def test_mine_hard_examples_flags_one_frame_label_flicker(tmp_path):
    dataset = tmp_path / "dataset"
    frames = [
        {
            "split": "train",
            "video": "session.mp4",
            "frame": i,
            "time_s": f"{float(i):.1f}",
            "image": f"images/train/f{i}.jpg",
            "label": f"labels/train/f{i}.txt",
            "detections": 1,
            "min_conf": "0.9",
            "labels": "",
        }
        for i in range(3)
    ]
    corrections = [
        _row("f0", 0, "AS", 0.90, (420, 700, 520, 850)),
        _row("f1", 0, "KD", 0.90, (420, 700, 520, 850)),
        _row("f2", 0, "AS", 0.90, (420, 700, 520, 850)),
    ]
    _write_csv(dataset / "manifest.csv", MANIFEST_FIELDS, frames)
    _write_csv(dataset / "detections.csv", DETECTION_FIELDS, corrections)
    _write_csv(dataset / "corrections.csv", DETECTION_FIELDS, corrections)

    mine_hard_examples(
        dataset,
        dataset / "hard_examples",
        image_width=1000,
        image_height=1000,
        churn_window_seconds=0,
    )

    rows = _read_report(dataset / "hard_examples" / "hard_examples.csv")
    assert "state_flicker" in rows["images/train/f1.jpg"]["issue_types"].split(";")

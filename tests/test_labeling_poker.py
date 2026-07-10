import struct

from labeling_poker.db import connect, get_annotations, get_status, progress, save_annotations, sync_files
from labeling_poker.export import image_size, split_ids, write_dataset


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


def test_sync_files_supports_nested_image_directories(tmp_path):
    images = tmp_path / "images"
    (images / "train").mkdir(parents=True)
    write_png(images / "train" / "nested.png")
    connection = connect(tmp_path / "labels.sqlite3")
    assert sync_files(connection, images) == ["nested"]
    assert connection.execute("SELECT path FROM files WHERE id = 'nested'").fetchone()[0] == "train/nested.png"


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


def test_split_is_deterministic():
    assert split_ids(["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]) == {
        "train": ["01", "02", "03", "04", "05", "06", "07", "08"],
        "val": ["09"],
        "test": ["10"],
    }

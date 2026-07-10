from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_IMAGES_DIR = ROOT / "data" / "images"
DEFAULT_DB_PATH = ROOT / "data" / "labels.sqlite3"
DEFAULT_PRIORITY_DIR = ROOT / "priority"
EXISTING_DATASET_IMAGES_DIR = ROOT.parent / "cv_lab" / "datasets" / "yolo_cards_autolabel_v1" / "images"
DEFAULT_MODEL_PATH = ROOT.parent / "cv_lab" / "models" / "best (4).pt"
YOLOV12_VENDOR_CANDIDATES = [
    ROOT.parent / "cv-backend" / "vendor" / "yolov12",
    ROOT.parent.parent / "YoloCardDetectTest" / "cv-backend" / "vendor" / "yolov12",
]

# Keep class order stable: numeric IDs in exported YOLO labels follow this list.
CLASSES = [
    "face_card",
    "card_back",
    "dealer_button",
    "stack_text",
    "bet_text",
    "pot_text",
    "player_name_text",
    "active_turn_indicator",
]
CLASS_COLORS = {
    "face_card": "#36cfc9",
    "card_back": "#13c2c2",
    "dealer_button": "#ffc53d",
    "stack_text": "#73d13d",
    "bet_text": "#9254de",
    "pot_text": "#40a9ff",
    "player_name_text": "#ff7875",
    "active_turn_indicator": "#f759ab",
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

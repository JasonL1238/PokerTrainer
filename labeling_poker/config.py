import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_IMAGES_DIR = ROOT / "data" / "images"
DEFAULT_DB_PATH = ROOT / "data" / "labels.sqlite3"
DEFAULT_PRIORITY_DIR = ROOT / "priority"
EXISTING_DATASET_IMAGES_DIR = ROOT.parent / "cv_lab" / "datasets" / "yolo_cards_autolabel_v1" / "images"
DEFAULT_MODEL_PATH = ROOT.parent / "cv_lab" / "models" / "best (4).pt"
DEFAULT_REGION_MODEL_PATH = ROOT.parent / "cv_lab" / "models" / "region_spine_v1.pt"
YOLOV12_VENDOR_CANDIDATES = [
    ROOT.parent / "cv-backend" / "vendor" / "yolov12",
    ROOT.parent.parent / "YoloCardDetectTest" / "cv-backend" / "vendor" / "yolov12",
]

# Keep class order stable: numeric IDs in exported YOLO labels follow this list.
# Lean reconstruction schema: every class feeds hand reconstruction directly.
# player_name_text (irrelevant to single-hand reconstruction) stays dropped.
# bet_text was appended LAST (ID 7) after labeling began so existing IDs 0-6 stay
# fixed: for hands where recording started mid-street there is no start-of-street
# stack baseline, so the chips a player has committed this street can't be recovered
# from stack deltas; the bet_text amount supplies that baseline directly.
CLASSES = [
    "face_card",
    "card_back",
    "dealer_button",
    "pot_text",
    "stack_text",
    "action_pill",
    "active_turn_indicator",
    "bet_text",
]
CLASS_COLORS = {
    "face_card": "#36cfc9",
    "card_back": "#13c2c2",
    "dealer_button": "#ffc53d",
    "pot_text": "#40a9ff",
    "stack_text": "#73d13d",
    "action_pill": "#9254de",
    "active_turn_indicator": "#f759ab",
    "bet_text": "#ff7a45",
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

# Rank/suit attribute for face_card boxes. The label is stored as rank+suit,
# e.g. "Kd" (king of diamonds) or "Ts" (ten of spades), matching the YOLO card
# detector's class naming. Only face_card boxes may carry a card label.
CARD_LABEL_CLASS = "face_card"
CARD_RANKS = ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"]
CARD_SUITS = ["s", "h", "d", "c"]
CARD_SUIT_SYMBOLS = {"s": "♠", "h": "♥", "d": "♦", "c": "♣"}
CARD_LABEL_RE = re.compile(f"^[{''.join(CARD_RANKS)}][{''.join(CARD_SUITS)}]$")


def normalize_card_label(class_name: str, label: object) -> str | None:
    """Return a validated card label, or None. Raises ValueError on bad input.

    Only face_card boxes keep a label; any label on another class is dropped.
    Accepts both the picker's canonical form ("Kd", "Tc") and the YOLO card
    detector's own naming ("KD", "10C", "joker"), canonicalizing to rank+suit
    with an uppercase rank (T for ten) and a lowercase suit, matching
    poker_tracker.cards. "joker" and empty values collapse to None.
    """
    if class_name != CARD_LABEL_CLASS:
        return None
    if label in (None, ""):
        return None
    text = str(label).strip()
    if not text or text.lower() == "joker":
        return None
    if text[:2] == "10":
        text = "T" + text[2:]
    if len(text) == 2:
        text = text[0].upper() + text[1].lower()
    if not CARD_LABEL_RE.match(text):
        raise ValueError(f"invalid card label {label!r}; expected rank+suit like 'Kd'")
    return text

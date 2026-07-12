# Poker YOLO box labeler

This is a local Flask + vanilla JavaScript tool for labeling poker table images with axis-aligned YOLO detection boxes. It has no polygon or live model-assist workflow.

## Run

Put source images in `labeling_poker/data/images/`. The filename stem is the image ID, so `000123.jpg` is served as `/image/000123`. If that folder is empty, the app automatically uses the existing `cv_lab/datasets/yolo_cards_autolabel_v1/images/` dataset, including its train/val/test subfolders.

The classes are `face_card`, `card_back`, `dealer_button`, `pot_text`, `stack_text`, `action_pill`, and `active_turn_indicator`. These are generic visual regions chosen so YOLO detections carry everything hand reconstruction needs: cards define streets and hand boundaries, card backs reveal who is dealt in and who folds, the pot and stacks form the arithmetic reconciliation net (bet sizes come from stack deltas), the dealer button fixes positions, action pills give each seat's last action (color read from the crop; gray check-vs-fold is resolved by card backs disappearing), and the turn indicator orders actions within a street. Rank/suit and amounts are read later as attributes with card classification or OCR. `bet_text` and `player_name_text` were deliberately dropped: bets are redundant with stack deltas and names are irrelevant to single-hand reconstruction.

```bash
python -m labeling_poker.app --port 5055
```

Open <http://127.0.0.1:5055>. A different image directory or SQLite database can be supplied with `--images` and `--db`.

## Labeling

Choose a class in the sidebar or press number keys 1-7. Drag to create a box. Click a box to select it, drag it to move it, or drag a corner to resize it.

`Enter` saves labeled boxes and loads the next undecided image. `K` marks an image clean with no objects. `X` excludes a duplicate from the labeling queue and export. `S` skips without writing to SQLite. Left/Right seeks through the complete image pool. `D` deletes the selected box and `Z` undoes the last box edit. Duplicate exclusion is reversible through Previous; the source image is not physically deleted.

The class list and colors are in `config.py`. Add new classes there; their order becomes the YOLO class ID order.

## Card rank + suit

When a `face_card` box is selected, the sidebar "Card (rank + suit)" picker becomes active. Click a rank (`A`–`2`) and a suit (`♠♥♦♣`) to tag the box with a card label such as `Kd` or `Tc`. The label is optional metadata stored alongside the box; it does not change the YOLO detection class (`face_card` stays class 0). The picker only applies to `face_card` boxes and clears itself when a box is relabeled to another class. YOLO-bootstrap prelabels (e.g. `KD`, `10C`) are canonicalized to the same `rank+suit` form (uppercase rank with `T` for ten, lowercase suit), matching `poker_tracker.cards`.

## Export

Only `labeled` and `clean` images are exported. Clean images receive empty label files and are useful negatives. Duplicate, undecided, and skipped images are excluded. IDs are sorted deterministically and split 80/10/10.

```bash
python -m labeling_poker.export \
  --db labeling_poker/data/labels.sqlite3 \
  --images labeling_poker/data/images \
  --out cv_lab/datasets/poker_boxes_v1
```

The output contains `images/{train,val,test}`, `labels/{train,val,test}`, and `data.yaml`.

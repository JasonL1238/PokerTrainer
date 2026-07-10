# Poker YOLO box labeler

This is a local Flask + vanilla JavaScript tool for labeling poker table images with axis-aligned YOLO detection boxes. It has no polygon or live model-assist workflow.

## Run

Put source images in `labeling_poker/data/images/`. The filename stem is the image ID, so `000123.jpg` is served as `/image/000123`. If that folder is empty, the app automatically uses the existing `cv_lab/datasets/yolo_cards_autolabel_v1/images/` dataset, including its train/val/test subfolders.

The MVP classes are `face_card`, `card_back`, `dealer_button`, `stack_text`, `bet_text`, `pot_text`, `player_name_text`, and `active_turn_indicator`. These are generic visual regions. Rank/suit and amounts are read later as attributes with card classification or OCR; actions such as call, bet, and raise are inferred from changes over time.

```bash
python -m labeling_poker.app --port 5055
```

Open <http://127.0.0.1:5055>. A different image directory or SQLite database can be supplied with `--images` and `--db`.

## Labeling

Choose a class in the sidebar or press number keys 1-6. Drag to create a box. Click a box to select it, drag it to move it, or drag a corner to resize it.

`Enter` saves labeled boxes and loads the next undecided image. `K` marks an image clean with no objects. `X` excludes a duplicate from the labeling queue and export. `S` skips without writing to SQLite. Left/Right seeks through the complete image pool. `D` deletes the selected box and `Z` undoes the last box edit. Duplicate exclusion is reversible through Previous; the source image is not physically deleted.

The class list and colors are in `config.py`. Add new classes there; their order becomes the YOLO class ID order.

## Export

Only `labeled` and `clean` images are exported. Clean images receive empty label files and are useful negatives. Duplicate, undecided, and skipped images are excluded. IDs are sorted deterministically and split 80/10/10.

```bash
python -m labeling_poker.export \
  --db labeling_poker/data/labels.sqlite3 \
  --images labeling_poker/data/images \
  --out cv_lab/datasets/poker_boxes_v1
```

The output contains `images/{train,val,test}`, `labels/{train,val,test}`, and `data.yaml`.

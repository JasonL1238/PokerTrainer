# Poker YOLO box labeler

This is a local Flask + vanilla JavaScript tool for labeling poker table images with axis-aligned YOLO detection boxes. It has no polygon or live model-assist workflow.

## Run

Put source images in `labeling_poker/data/images/`. The filename stem is the image ID, so `000123.jpg` is served as `/image/000123`. If that folder is empty, the app automatically uses the existing `cv_lab/datasets/yolo_cards_autolabel_v1/images/` dataset, including its train/val/test subfolders.

The classes are `face_card`, `card_back`, `dealer_button`, `pot_text`, `stack_text`, `action_pill`, `active_turn_indicator`, and `bet_text`. These are generic visual regions chosen so YOLO detections carry everything hand reconstruction needs: cards define streets and hand boundaries, card backs reveal who is dealt in and who folds, the pot and stacks form the arithmetic reconciliation net (bet sizes come from stack deltas), the dealer button fixes positions, action pills give each seat's last action (color read from the crop; gray check-vs-fold is resolved by card backs disappearing), the turn indicator orders actions within a street, and bet text provides a committed-chip baseline when a recording begins mid-street. Rank/suit and amounts are read later as attributes with card classification or OCR. `player_name_text` was deliberately dropped because names are irrelevant to single-hand reconstruction.

```bash
python -m labeling_poker.app --port 5055
```

Open <http://127.0.0.1:5055> — it redirects to the one labeling setup:

`http://127.0.0.1:5055/?queue=two_model_validation`

Use the **Browse** dropdown inside that page for New / Labeled / Labeled sus / Unlabeled sus / etc. Everything bootstraps with Model 1 regions + Model 2 card names. A different image directory or SQLite database can be supplied with `--images` and `--db`.

## Labeling

Choose a class in the sidebar or press number keys 1-8. Drag to create a box. Click a box to select it, drag it to move it, or drag a corner to resize it.

`Enter` saves labeled boxes and loads the next image in the active Browse selection. `K` marks an image clean with no objects. `X` excludes a duplicate from the labeling queue and export. `S` skips without writing to SQLite. Left/Right seeks through the active Browse selection. `D` deletes the selected box and `Z` undoes the last box edit. Duplicate exclusion is reversible through Previous; the source image is not physically deleted.

Use the **Browse** menu above the image to revisit saved work: choose **Labeled** to move only through images with saved boxes, **Labeled sus** for heuristic re-review of likely mistakes, **Clean** or **Excluded duplicates** for those decisions, or **All images**. Saved images load their SQLite annotations directly; they are not re-bootstraped by a model. In `/?queue=model1_review&view=labeled` and `/?queue=two_model_validation&view=labeled`, the view is limited to that queue and opens the most recently saved label first. **Older label** (or Left) moves backward in time; **Newer label** (or Right) moves forward and wraps from the newest label to the oldest.

**Labeled sus** is built by scanning saved boxes for things like duplicate card names in one frame, heavy overlapping face cards, multiple dealer buttons, missing rank/suit on some face cards, and disagreements with the cached two-model predictions. Regenerate it anytime with:

```bash
python -m labeling_poker.label_audit
```

Then open `/?view=labeled_sus` (or pick **Labeled sus** in Browse). The status line shows why each frame was flagged. After you fix and re-save, re-run the audit to shrink the queue.

**Unlabeled sus** is the same idea for frames that are still undecided: it audits the cached two-model auto-labels (duplicate cards, overlaps, missing ranks, empty predictions, etc.) so you can fix the worst auto-labels first. Regenerate with:

```bash
python -m labeling_poker.label_audit --target unlabeled
```

Then open `/?view=unlabeled_sus` (or pick **Unlabeled sus** in Browse / Gallery). Saving a frame removes it from undecided; re-run the audit to refresh the queue.

The class list and colors are in `config.py`. Add new classes there; their order becomes the YOLO class ID order.

## Model 1 vs. Model 2

The production CV path is deliberately split by job:

```
full recorded frame
  -> Model 1: 8-class region detector
  -> face_card crops (with padding)
  -> Model 2: 52-class rank/suit classifier
  -> reconstruction spine / completed-hand timeline
```

**Model 1** finds the geometry needed to reconstruct a hand: `face_card`, `card_back`, `dealer_button`, `pot_text`, `stack_text`, `action_pill`, `active_turn_indicator`, and `bet_text`. It only decides that a `face_card` region is present; it does not name the card. Its training data is the full-frame boxes exported by `labeling_poker.export`. The optional rank/suit value on a `face_card` is ignored by that YOLO export.

**Model 2** receives only Model 1's `face_card` crop and returns one of the 52 canonical cards, such as `As`, `Kd`, or `Tc`. Its training data is made by `build_card_cls_dataset.py`, which uses only reviewed `face_card` boxes that also have a rank and suit selected in this app. Other region classes are irrelevant to Model 2.

The queues serve different review loops:

- `/?queue=model1_review` stages fresh full frames and bootstraps every region from Model 1. Correct all region boxes and classes; add card ranks/suits too only when you also want those crops to improve Model 2.
- `cv_weak_cards` is produced by the combined runtime when Model 2 is low-confidence or hits a known weak rank. It prioritizes frames for card-label correction. In the labeler today, only the `model1_review` queue switches bootstrap source; other queues use the legacy full-frame card detector (`best (4).pt`) for a convenient starting face-card box. That detector is not Model 2.
- `two_model_validation` is an approval queue created by `stage_two_model_validation_queue.py`. It contains only undecided frames, and each frame receives Model 1's region boxes plus Model 2's card names when opened. Those predictions are never written to SQLite unless the reviewer saves them, so approval or an edit is required before either model can train on them.

The end-to-end completed-session runner is `cv_lab/scripts/run_two_model_pipeline.py`; it runs Model 1 first, invokes Model 2 only for its card boxes, and then sends the results into the reconstruction spine. This is offline post-session analysis only.

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

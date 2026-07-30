# CV Lab — Finding 15: mid-hand button-position repair

## Observed defect

Manual reconstruction review on job 2, hand 1 flagged four source frames. The
recording begins during preflop action with the dealer button visibly at seat 4.
Seats 1, 3, and 4 had already folded, so their card backs and short-lived action
pills were gone even though all eight occupied stack HUDs remained.

The old roster used card backs and persistent pills only. It reduced the hand to
five players, could not find dealer seat 4 in that reduced ring, and defaulted
the first surviving seat (hero seat 0) to BTN. This produced the reviewed errors:

- hero seat 0 reported BTN instead of UTG+1;
- seat 5 reported BB instead of SB;
- seat 6 reported UTG instead of BB;
- actions already visible in frame zero were sorted by numeric seat index rather
  than legal preflop order.

## Repair

When the opening state proves capture is already mid-hand, stable opening stack
HUDs now recover occupied seats whose folds preceded capture. The dealer-button
seat is always retained as independent participation evidence. Position labels
are assigned only by walking the physical seat ring from that button.

Seats recovered only from opening occupancy and already lacking cards, a live
pill, or a standing bet are latched as folded before observation. They remain in
the position ring but cannot receive invented later actions.

When multiple actions first become visible in one sampled state, their order is
derived from the button-anchored street order: left of the big blind preflop and
left of the button postflop. Action observations never determine positions.

## Regression

`test_mid_hand_start_keeps_pre_capture_folders_in_button_position_ring` recreates
the reviewed eight-seat frame-zero shape. It pins the BTN/SB/BB/UTG/UTG+1 labels,
retention of all eight occupied seats, and the coherent preflop action order
ending with the big blind's raise.

The verification environment also exposed that the newly released OpenCV 5
changes synthetic text rasterization enough to break the calibrated OCR tests.
The runtime dependency is therefore capped below 5; OpenCV 4.11 retains the
verified reader behavior.

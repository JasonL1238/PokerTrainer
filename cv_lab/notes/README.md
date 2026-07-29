# CV research notes

These files are a chronological engineering record, not the active product
roadmap. Read them in numeric order when investigating why the pipeline works
the way it does.

- `01`–`03` establish the layout and early VLM-assisted validation baseline.
- `04`–`05` establish deterministic frame selection and landmark anchoring.
- `06` records the decision that VLMs are validation-only, never part of the
  shipping reconstruction pipeline.
- `07`–`08` implement and validate the deterministic readers and full pipeline.
- `09` is the reconstruction-bridge evaluation and lists the gaps measured at
  that point.
- `10`–`12` record the adversarial repair rounds against the geometry-generalized
  pipeline, including the findings whose proposed remedies the measurements
  disproved.
- `13` is the final Phase 5 measurement across all five development geometries,
  with the before/after numbers, the one read that regressed, and an explicit
  list of what the phase does not establish.
- `14` is the latest note: the Option A reader contract (a value only when
  provably unambiguous, else a named UNKNOWN) and the round-1 adversarial
  repair — occlusion/clipping/missing-digit defences, downstream UNKNOWN
  integrity, the mutation-testing kills, and the corrected coverage figures
  (9 of 31 hands over six development recordings), superseding note 13's
  11-of-21 and the stale PLAN.md paragraph.

Status and TODO statements in an earlier finding are historical and may be
completed or superseded by a later finding. Use the repository root `PLAN.md`
for current priorities.

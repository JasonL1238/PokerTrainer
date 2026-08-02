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
- `14` records the Option A reader contract (a value only when provably
  unambiguous, else a named UNKNOWN) and the round-1 adversarial repair —
  occlusion/clipping/missing-digit defences, downstream UNKNOWN integrity, the
  mutation-testing kills, and the corrected coverage figures (9 of 31 hands over
  six development recordings), superseding note 13's 11-of-21 and the stale
  PLAN.md paragraph.
- `15` records adversarial repair round 2 against the Option A
  contract. Nine of fifteen findings were already closed and were re-verified;
  the three real ones were an unpinned P1, a sign check that enumerated two of
  four result values, and an unpinned row-band split. It also records the
  ablation audit of all nineteen acceptance-path predicates, a false positive
  disproved in an adversary's own evidence, and the fact that the round moved no
  coverage at all.

The last two notes are not CV findings. They are the adversarial-round record for
the product as a whole, moved out of `PLAN.md` on 2026-08-02 so the plan holds
the current plan and this directory holds the history:

- `16` is the Phase 1 record: fifteen adversarial rounds against hand completion
  and study readiness, the close-out pass, the structural repair that replaced
  the per-field disclosure gate with the assumption-dependence rule, the
  regression inventory round by round, and the gaps as they stood when round 15
  closed. It ends with an explicit list of its own statements that later work has
  superseded.
- `17` is the whole-product record: rounds 1 through 3 of the two-agent release
  loop. Round 1 broke the release gate's own reporting; rounds 2 and 3 each
  landed criticals in pot accounting, and in both cases at least one was
  introduced by the repair to the round before.

Status and TODO statements in an earlier finding are historical and may be
completed or superseded by a later finding. Use the repository root `PLAN.md`
for current priorities: it holds every still-open item, and nothing in this
directory is a claim about what is true today.

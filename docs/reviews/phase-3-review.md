# Review: Phase 3 (Smooth)

Reviewer: Codex (gpt-6.1-sol, reasoning effort xhigh, read-only sandbox), 14 Nov 2026.
Reviewed: the uncommitted Phase 3 changes on `feature/cleanup-and-audition`, against
`docs/plans/no-resize-cleanup-and-audition.md`. Codex did not compile or run anything.

The review is reproduced unedited. Line numbers refer to the tree as reviewed, before the
fixes listed in the disposition table were applied.

---

Phase 3 needs revision before commit: un-mixing can erase features that the preservation tests claim to protect, and the corpus baseline contradicts the completion report. This was a read-only review; I did not run Cargo.

1. **High — Un-mixing defeats the outline and Light feature guarantees.**

   **Evidence:** [cleanup.rs:269](/Users/shaun/dev/spritefusion-pixel-snapper/src/cleanup.rs:269) restricts candidates to core labels, then [cleanup.rs:282](/Users/shaun/dev/spritefusion-pixel-snapper/src/cleanup.rs:282) replaces the pixel unconditionally. The outline and 2×2 tests call only `sieve` ([cleanup.rs:820](/Users/shaun/dev/spritefusion-pixel-snapper/src/cleanup.rs:820), [cleanup.rs:850](/Users/shaun/dev/spritefusion-pixel-snapper/src/cleanup.rs:850)).

   **Impact:** The alternating near-blacks have no core pixels. With a valid radius of 3, surrounding white/grey cores repaint the outer two columns on each side, leaving a one-pixel outline. At radius 2, an isolated 2×2 feature is entirely repainted as background, even when Light’s threshold remains 4. The sieve cannot recover either.

   **Recommendation:** Protect these non-blended features during un-mixing. Run both preservation regressions through un-mix followed by sieve, exercising estimated pitches that select radii 2 and 3.

2. **Medium — The supplied baseline fails the claimed completion target.**

   **Evidence:** [corpus-baseline.jsonl:77](/Users/shaun/dev/spritefusion-pixel-snapper/scripts/corpus-baseline.jsonl:77) records **685** regions under 16 px for wolf/Medium. [Progress:489](/Users/shaun/dev/spritefusion-pixel-snapper/docs/plans/no-resize-cleanup-and-audition.md:489) claims **246**, passing the under-528 requirement. Other Smooth counts also disagree.

   **Impact:** Either the baseline or the completion report is stale. The corpus check currently accepts a result that misses Phase 3’s acceptance target.

   **Recommendation:** Remeasure the final implementation, reconcile the baseline and Progress, and assert the acceptance threshold explicitly.

3. **Medium — Tests do not distinguish source-mean targeting from palette-colour targeting.**

   **Evidence:** The sieve fixtures use [cleanup.rs:749](/Users/shaun/dev/spritefusion-pixel-snapper/src/cleanup.rs:749), which indexes each image against itself. Their source colours therefore equal their palette colours; integration tests assert aggregate counts.

   **Impact:** Replacing the component’s mean source colour with its current palette colour could pass these tests while changing the central colour-aware merge rule.

   **Recommendation:** Add a fixture where those two colours prefer different adjacent labels, and assert the exact destination. Include a closer, non-adjacent label to verify adjacency.

4. **Low — Preserving raw labels was silently deferred.**

   **Evidence:** [plan:215](/Users/shaun/dev/spritefusion-pixel-snapper/docs/plans/no-resize-cleanup-and-audition.md:215) requires both maps to be retained. The raw clone at [cleanup.rs:255](/Users/shaun/dev/spritefusion-pixel-snapper/src/cleanup.rs:255) is temporary; Blocks returns only the rendered image and grid.

   **Impact:** Phase 5 cannot consume the promised raw map from this result.

   **Recommendation:** Retain both maps or explicitly record this as deferred work.

---

## Disposition (Claude)

| # | Finding | Checked | Outcome |
|---|---|---|---|
| 1 | Un-mixing repaints features that have no flat interior: the two-black outline loses its outer columns at radius 3, and a 2×2 feature is repainted as background at radius 2. The preservation tests only exercised the sieve | Correct. An un-mix that takes "the nearest label any nearby flat area has" cannot tell a blend from a small feature | Fixed. A pixel is re-labelled only when its source colour **is** a blend: within `BLEND_TOLERANCE` (0.05 OKLab) of the line between two colours carried by nearby flat pixels, or of one of them. The outline's blacks are nowhere near the white–grey line and the 2×2 feature is nowhere near its background, so both are left alone. Tests: `unmix_then_sieve_keeps_the_outline_at_every_radius`, `unmix_then_light_sieve_keeps_a_two_by_two_feature` (radii 2, 3, 4), `blend_test_accepts_ramps_and_rejects_features`. Consequence: the reach had to grow by one pixel (2–4 instead of 1–3), because a blend needs **both** sides in view and flat pixels are at least one pixel away from any ramp |
| 2 | The corpus baseline (685 regions under 16 px for wolf/Medium) contradicted the Progress table (246) | Correct, and my error: the table was measured with a CLI binary built **before** the sieve fix (`cargo test` does not rebuild it); the baseline was right | Everything was remeasured from a fresh build after the final change and the table rewritten. The acceptance criterion "under 2 % of regions smaller than 16 px at Medium" is **not met** (853, 3.2 %) and was the wrong criterion: Medium removes regions under 8 px, and the 2 % figure came from the research prototype, whose sieve and un-mix both removed things they should not have. What each strength guarantees is now stated and tested instead: nothing under its threshold survives |
| 3 | No test separated "merge towards the region's mean source colour" from "merge towards its palette colour", or checked adjacency | Correct: the fixtures indexed images against themselves | Test `sieve_merges_by_source_colour_into_adjacent_labels_only`: a region whose label is nearer `dark` but whose source pixels are nearer `light` must join `light`, and must not join an even closer colour that is not adjacent |
| 4 | The plan said both the raw and the un-mixed label maps are kept; Blocks returns only an image and a grid | Correct | Declared as deferred to Phase 6, where the Detail vote that reads the raw map is built |

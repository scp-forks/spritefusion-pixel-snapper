# Review: Phase 4 (DP grid fit)

Reviewer: Codex (gpt-6.1-sol, reasoning effort xhigh, read-only sandbox), 14 Nov 2026.
Reviewed: the uncommitted Phase 4 changes on `feature/cleanup-and-audition`, against
`docs/plans/no-resize-cleanup-and-audition.md`. Codex did not compile or run anything.

The review is reproduced unedited. Line numbers refer to the tree as reviewed, before the
fixes listed in the disposition table were applied.

---

Phase 4 needs revision before commit: border cleanup invalidates the optimized result and its score, small-image fallbacks can collapse the output, and degenerate axes disable pitch search. The claimed step bounds and refinement coverage also need correction. This review was read-only; I did not run Cargo.

1. **Medium — Sliver removal invalidates optimality and pitch scores.**

   **Evidence:** [src/grid.rs:478](/Users/shaun/dev/spritefusion-pixel-snapper/src/grid.rs:478) saves the score before removing border cuts. For a length-24 profile, pitch 8, and nonzero samples `{2:3, 11:3, 17:3, 19:4, 23:4}`, backtracking selects `[0,2,11,17,23,24]`, then deletes 23. The returned grid scores approximately 12.393, while `[0,2,11,19,24]` scores 14.055. The reported score remains 17.790.

   **Impact:** The returned grid is not globally optimal, and pitch selection can reward edges absent from its output.

   **Recommendation:** Enforce border constraints within the recurrence and terminal selection. Score the returned grid, and add a small brute-force optimality test covering border slivers.

2. **Medium — Subminimum fallback pitches collapse valid images to one cell.**

   **Evidence:** [src/grid.rs:200](/Users/shaun/dev/spritefusion-pixel-snapper/src/grid.rs:200) accepts the fallback pitch unchanged; [src/grid.rs:434](/Users/shaun/dev/spritefusion-pixel-snapper/src/grid.rs:434) rejects pitches below 2. A 64×64 image split into two solid halves has insufficient peaks to estimate spacing, so its fallback is 1. Every searched pitch is below 2, leaving cuts `[0,64]` on both axes.

   **Impact:** DP produces a 1×1 snap output despite a strong, perfectly representable boundary.

   **Recommendation:** Clamp DP’s fallback/search seed to the minimum feasible pitch. Add a small image with one clear boundary and no spacing estimate.

3. **Medium — A 3 px axis disables pitch search on the other axis.**

   **Evidence:** [src/grid.rs:434](/Users/shaun/dev/spritefusion-pixel-snapper/src/grid.rs:434) returns `NEG_INFINITY` whenever `limit < 4`. [src/grid.rs:206](/Users/shaun/dev/spritefusion-pixel-snapper/src/grid.rs:206) sums axis scores, making every candidate score `NEG_INFINITY` for a valid 3×N image.

   **Impact:** No candidate can beat the initial estimate, even when another pitch substantially improves the long axis.

   **Recommendation:** Give a legal, forced single-cell axis a neutral score. Test pitch correction on both 3×N and N×3 images.

4. **Medium — The actual step bounds exceed the promised ±25%.**

   **Evidence:** [src/grid.rs:437](/Users/shaun/dev/spritefusion-pixel-snapper/src/grid.rs:437) rounds the lower bound down and upper bound up. Pitch 5 therefore permits steps 3–7: ±40%. A profile peaking every 3 pixels can return 3 px interior cells at pitch 5.

   **Impact:** This is an undeclared deviation from Phase 4’s hard bounds and materially affects small pitches and refinement.

   **Recommendation:** Use `ceil` for the lower bound and `floor` for the upper bound, or explicitly declare the wider integer tolerance. Test actual interior intervals.

5. **Low — The “No free ×1.5” claim is contradicted and unguarded.**

   **Evidence:** [Progress:586](/Users/shaun/dev/spritefusion-pixel-snapper/docs/plans/no-resize-cleanup-and-audition.md:586) says a clean lattice refits to its original cuts and returns `None`. On the existing clean 8 px checkerboard, the recurrence instead refines 12×12 cells to 24×24. [src/grid.rs:844](/Users/shaun/dev/spritefusion-pixel-snapper/src/grid.rs:844) permits either success or refusal at ×1.5.

   **Impact:** The test cannot establish the claimed rejection behavior, and Progress overstates what was added.

   **Recommendation:** Correct the claim and assert the intended ×1.5 behavior explicitly, including a fixture that must return `None`.

6. **Low — The local-median regression fixture also accepts a local mean.**

   **Evidence:** [src/grid.rs:769](/Users/shaun/dev/spritefusion-pixel-snapper/src/grid.rs:769) checks dimensions, zero residual, and pitch within 0.15 of 5. An in-memory reproduction using a local-mean baseline still satisfies those assertions: 80×70 cells, zero residual, pitch 5.0625.

   **Impact:** This fixture does not distinguish the median from the rejected local-mean approach described in Progress.

   **Recommendation:** Add a direct `edge_gains` assertion with an outlier that separates median from mean, plus a sparse-edge fixture with missing peaks.

---

## Disposition (Claude)

| # | Finding | Checked | Outcome |
|---|---|---|---|
| 1 | Border slivers were removed after optimising, so the returned grid was no longer optimal and its reported score belonged to a different grid | Correct. Reproduced Codex's 24-sample example | Fixed by moving the border rules into the recurrence: cuts always run from 0 to the limit, the first and last cell may be as narrow as 0.35 of the pitch, and nothing is edited afterwards. `score_cuts` scores any set of cuts independently of the recurrence, and `dp_fit_is_optimal_and_reports_its_own_score` checks 300 random short axes against brute-force enumeration, plus the example. A first attempt made short border cells free; the brute-force test passed but three lattice tests showed the fit then shaves a 2 px sliver off every border, so border cells pay the normal step penalty |
| 2 | A fallback step under 2 px (small image, no spacing estimate) made every candidate pitch illegal and collapsed the grid to one cell | Correct | Fixed: the DP's seed pitch is clamped to the minimum cell. Test `dp_handles_short_and_edgeless_axes` (a 64×64 image of two solid halves finds the boundary) |
| 3 | A 3 px axis scored −∞ and vetoed the pitch search on the other axis | Correct | Fixed: an axis too short to cut is one forced cell with a neutral score. Same test, on a 200×3 strip and its rotation |
| 4 | Step bounds were floor/ceil, so "±25 %" was really 3–7 at a pitch of 5 (±40 %) | Correct, an undeclared deviation | Fixed: bounds round inwards (4–6 at a pitch of 5), with at least two consecutive step sizes so that every length can be tiled. Test `dp_interior_steps_stay_within_tolerance`. The irregular-lattice fixture was narrowed to 7–10 px cells, which is inside the tolerance it claims |
| 5 | "No free ×1.5" overstated: on a clean 8 px lattice the ×1.5 re-fit succeeds (as a ×2 subdivision), and the test accepted either outcome | Correct | Progress corrected, and `fractional_refinement_goes_where_the_edges_are` asserts both behaviours explicitly: 8 px cells re-fit to the ×2 grid, 3 px cells re-fit to themselves and are refused |
| 6 | The flat-margin fixture would also pass with a local-mean baseline | Correct | Added `edge_gains_use_a_baseline_that_edges_do_not_move`, which asserts that positions next to an edge stay at exactly zero and the edge keeps its full height (a mean fails both), and `dp_bridges_missing_edges` (three boundaries in four have no edge) |

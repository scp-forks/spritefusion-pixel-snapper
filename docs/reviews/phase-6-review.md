# Review: Phase 6 (Detail vote and lost-detail overlay)

Reviewer: Codex (gpt-6.1-sol, reasoning effort xhigh, read-only sandbox), 14 Nov 2026.
Reviewed: the uncommitted Phase 6 changes on `feature/cleanup-and-audition`, against
`docs/plans/no-resize-cleanup-and-audition.md`. Codex did not compile or run anything.

The review is reproduced unedited. Line numbers refer to the tree as reviewed, before the
fixes listed in the disposition table were applied.

---

I found no confirmed runtime or compatibility defect in Phase 6. I found five verification gaps, including a test that calls the wrong voter and a missing test for the frontend rasterization that replaced Rust rasterization. This was a read-only review; I did not run cargo.

1. **Medium — The promised overlay alignment test never exercises the overlay renderer.**

   **Evidence:** [src/candidates.rs:645](/Users/shaun/dev/spritefusion-pixel-snapper/src/candidates.rs:645) checks map length and painted-cell uniformity. It never calls [comparison.ts:133](/Users/shaun/dev/spritefusion-pixel-snapper/spritefusion-frontend/src/components/comparison.ts:133); no frontend test covers that component.

   **Impact:** Incorrect rasterization, transposed values, or misuse of source cuts for snapped output could pass the suite. The plan’s overlay-boundary acceptance test remains unimplemented.

   **Recommendation:** Check the actual rasterized pixels for irregular full-size cells and a non-square snapped result whose aspect differs from its source. Include a case where only `width === cols`.

2. **Low — Candidate-only Blocks fallback to Majority would escape the focused tests.**

   **Evidence:** [src/candidates.rs:639](/Users/shaun/dev/spritefusion-pixel-snapper/src/candidates.rs:639) checks cell uniformity under both votes; [scripts/wasm-smoke.mjs:191](/Users/shaun/dev/spritefusion-pixel-snapper/scripts/wasm-smoke.mjs:191) checks only Blocks Detail dimensions. Candidate corpus records contain metadata and map means, without output checksums.

   **Impact:** Changing candidate rendering to always use Majority could pass these checks. The thin-line tests exercise other routes and would catch a global fallback.

   **Recommendation:** Build Blocks candidates from a marked-feature fixture and assert that Detail preserves the feature where Majority erases it, comparing matching grids.

3. **Low — The Detail tie-breaking test calls the majority voter.**

   **Evidence:** [src/cleanup.rs:1502](/Users/shaun/dev/spritefusion-pixel-snapper/src/cleanup.rs:1502) calls `vote_cells`, inside `detail_vote_handles_flat_and_empty_cells_and_ties`.

   **Impact:** Reversing or removing Detail’s lowest-label tie-break would leave this test passing.

   **Recommendation:** Call `vote_cells_detail`. Supply explicitly symmetric Lab values so floating-point conversion does not turn the intended tie into unequal weights.

4. **Low — Lost-detail tests do not verify an unsaturated numeric result or degenerate splits.**

   **Evidence:** [src/candidates.rs:585](/Users/shaun/dev/spritefusion-pixel-snapper/src/candidates.rs:585) asserts zero, nonzero, and saturation. Odd cells occur, but their expected magnitudes are unchecked; no case exercises a one-pixel-wide cell’s empty half.

   **Impact:** Incorrect normalization, floor/scale arithmetic, or empty-half handling could survive those assertions.

   **Recommendation:** Add independently calculated expected bytes for an odd-sized cell, `1×N`, `N×1`, and `1×1`, including mixed transparency and a value between the floor and saturation.

5. **Low — The new snap alpha contract is untested at its decision boundary.**

   **Evidence:** [src/cleanup.rs:1574](/Users/shaun/dev/spritefusion-pixel-snapper/src/cleanup.rs:1574) tests 16, 6, and 10 opaque pixels out of 16, using only alpha 0 and 255.

   **Impact:** Changing `>=` to `>` would pass. Counting only fully opaque pixels, or preserving partial output alpha accidentally, could also pass.

   **Recommendation:** Test 7/16, 8/16, and 9/16 coverage, an odd-area cell, and source alpha 1/128. Assert the declared binary output alpha.

---

## Disposition (Claude)

Codex found no runtime or compatibility defect in this phase; all five findings are gaps in verification.

| # | Finding | Checked | Outcome |
|---|---|---|---|
| 1 | The overlay's alignment was only tested on the Rust side; nothing exercised the frontend code that actually paints it | Correct, and it mattered more because the rasterising had moved from Rust to the frontend | The painting was extracted into a pure function, `rasterizeOverlay` in `src/overlay.ts`, with four tests against exact expected pixels: irregular full-size cells, a non-square snapped result whose cuts say nothing about where to paint, the case where only the width matches the cell count, and cuts beyond the image |
| 2 | A build that ignored the vote when making Blocks candidates would have passed | Correct | Test `blocks_candidates_differ_by_vote_where_it_matters`: same grids under both votes, a three-tenths-of-a-cell line present in all 16 of its rows under Detail and in none under the majority, for both fitters |
| 3 | The detail vote's tie-break test called the majority voter | Correct, a slip | It now calls `vote_cells_detail` with Lab values chosen to be exact in binary, in both orders of first appearance. Writing it showed something worth knowing: a lone cell of two colours is *always* a tie under this vote, whatever their shares, because deviations from the cell's own mean balance; it is the neighbouring cells that decide. A second assertion checks that with a neighbour |
| 4 | The lost-detail map was only tested for zero, non-zero and saturation | Correct | Test `lost_detail_matches_a_naive_computation`: every cell of a grid with 1 px wide, 1 px tall, 1×1 and odd cells, on an image with scattered transparency, against an independent computation (gather the pixels, subtract the mean, sum the squares), to within one step; plus a hand-derived value for a cell split evenly between two greys |
| 5 | The snap alpha rule was not tested at its boundary | Correct | Test `snap_detail_alpha_is_decided_at_half_the_cell`: 6, 7, 8, 9, 10 and 16 of 16, 4 and 5 of 9, and source alphas of 1 and 128, asserting the output alpha is exactly 0 or 255 |

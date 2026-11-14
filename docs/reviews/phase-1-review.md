# Review: Phase 1 (Plumbing)

Reviewer: Codex (gpt-6.1-sol, reasoning effort xhigh, read-only sandbox), 14 Nov 2026.
Reviewed: the uncommitted Phase 1 changes on `feature/cleanup-and-audition`, against
`docs/plans/no-resize-cleanup-and-audition.md`. Codex did not compile or run anything.

The review is reproduced unedited. Line numbers refer to the tree as reviewed, before the
fixes listed in the disposition table were applied.

---

Phase 1 needs changes before commit: integer refinement can violate the 2 px minimum, and large finite scales can panic. The acceptance tests also miss important boundaries. This was a read-only review; I did not run cargo.

1. **Medium — Integer refinement preserves existing 1 px cells.**

   **Evidence:** [src/grid.rs:119](/Users/shaun/dev/spritefusion-pixel-snapper/src/grid.rs:119) forces at least one part even when `width / 2 == 0`. Acceptance checks only medians ([src/grid.rs:108](/Users/shaun/dev/spritefusion-pixel-snapper/src/grid.rs:108)). The subdivision test starts exclusively with widths ≥ 2.

   **Impact:** With columns `[0,1,5]`, rows `[0,4]`, and scale 2, refinement returns columns `[0,1,3,5]` and rows `[0,2,4]`. Both medians pass and cell count triples, but a 1 px interval survives. Legacy detection permits such input cells.

   **Recommendation:** Apply narrow-interval merging to integer results too, and verify every interval. Add cases with existing first, interior, and last 1 px cells.

2. **Medium — Allocation uses the uncapped subdivision factor.**

   **Evidence:** [src/grid.rs:116](/Users/shaun/dev/spritefusion-pixel-snapper/src/grid.rs:116) reserves `(cuts.len() - 1) * factor + 1`, although actual subdivision caps parts by cell width.

   **Impact:** For cuts `[0,4,8]` and finite scale `1e20`, the factor saturates to `usize::MAX`. Capacity arithmetic overflows in debug builds; release builds reach a capacity-overflow panic. Only five output cuts are needed. Smaller excessive scales can also request disproportionate memory.

   **Recommendation:** Calculate capacity from the capped per-cell part counts with checked arithmetic. Add large finite-scale cases.

3. **Low — Tests do not pin the 1.2× acceptance threshold.**

   **Evidence:** Accepted cases assert their resulting gain exceeds 1.2 ([src/grid.rs:254](/Users/shaun/dev/spritefusion-pixel-snapper/src/grid.rs:254)); rejection cases cover zero improvement and invalid scales ([src/grid.rs:259](/Users/shaun/dev/spritefusion-pixel-snapper/src/grid.rs:259)).

   **Impact:** Changing the threshold from 1.2 to 1.01 would still pass these tests. Median rejection is also never isolated from insufficient gain.

   **Recommendation:** Add cases with one axis fixed: `10→11` total cells must reject, and `10→12` must accept. Separately test a candidate that meets the gain requirement but fails the median requirement.

---

## Disposition (Claude)

| # | Finding | Checked | Outcome |
|---|---|---|---|
| 1 | Integer refinement keeps 1 px cells that the input grid already had; only medians were checked | Correct. `[0,1,5]` at scale 2 gave `[0,1,3,5]` | Fixed. The narrow-interval merge now runs on integer results as well, so no cell of any refined grid is under 2 px. That guarantee is stronger than the plan's median-cell rule, which became redundant and was removed (declared in the doc comment and the plan). Test `refinement_removes_existing_one_pixel_cells` covers first, interior, last and several 1 px cells, plus Codex's example |
| 2 | `Vec::with_capacity((cuts - 1) * factor + 1)` overflows or over-allocates for huge scales | Correct | Fixed: no capacity hint. Test `absurd_scales_are_capped_by_cell_width` runs 1e6, 1e20 and `f64::MAX` |
| 3 | Nothing pinned the 1.2× threshold | Correct | Test `refinement_needs_twenty_percent_more_cells`: 10 → 11 cells is rejected, 10 → 12 accepted |

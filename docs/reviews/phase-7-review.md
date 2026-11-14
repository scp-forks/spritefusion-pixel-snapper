# Review: Phase 7 (Evaluation corpus)

Reviewer: Codex (gpt-6.1-sol, reasoning effort xhigh, read-only sandbox), 14 Nov 2026.
Reviewed: the uncommitted Phase 7 changes on `feature/cleanup-and-audition`, against
`docs/plans/no-resize-cleanup-and-audition.md`. Codex did not compile or run anything.

The review is reproduced unedited. Line numbers refer to the tree as reviewed, before the
fixes listed in the disposition table were applied.

---

Request changes before committing: the new DP estimator regresses on short rectangular images, fractional-size scoring misplaces reference borders, and the Potts experiment does not yet support its conclusions reliably. This review was read-only; numerical counterexamples were checked with arithmetic and standalone translations. Cargo was not run.

1. **[P2] Short-axis rejection can halve a valid lattice pitch.**

   **Evidence:** [src/grid.rs:275](/Users/shaun/dev/spritefusion-pixel-snapper/src/grid.rs:275), [src/grid.rs:337](/Users/shaun/dev/spritefusion-pixel-snapper/src/grid.rs:337). `comb_capture` discards axes shorter than `2 * hi`, while their energy remains in `total`. For 120×18 profiles with edges every 12 px, `comb_pitch` returns 6; detection produces 20×3 cells, versus the previous 10×2.

   **Impact:** Coarse candidates lose solely because the short axis cannot accommodate their comb. Finer harmonics win on correctly gridded, cropped images.

   **Recommendation:** Handle short axes consistently across candidates or support partial combs. Add this rectangular case and its transpose as regressions.

2. **[P2] `pixel_accuracy` uses pixel corners instead of centres.**

   **Evidence:** [src/eval.rs:174](/Users/shaun/dev/spritefusion-pixel-snapper/src/eval.rs:174). Reference coordinates use `floor(x / sx)`, whereas resizing uses pixel centres. Three distinct pixels nearest-resized to eight score **87.5%**, despite being a perfect crisp upscale. The identity test at [src/eval.rs:267](/Users/shaun/dev/spritefusion-pixel-snapper/src/eval.rs:267) only covers integer scaling.

   **Impact:** Fractional-scale tables penalize correctly placed borders, undermining mode rankings and the reported 0.1-point differences.

   **Recommendation:** Map through `(x + 0.5) / sx` and likewise for y. Add non-integer identity tests, then regenerate the tables and dependent claims.

3. **[P2] The Potts comparison weakens the Smooth baseline by disabling pitch scaling.**

   **Evidence:** [src/eval.rs:491](/Users/shaun/dev/spritefusion-pixel-snapper/src/eval.rs:491) passes `None` to both cleanup parameter functions; shipping Smooth uses the detected pitch at [src/cleanup.rs:295](/Users/shaun/dev/spritefusion-pixel-snapper/src/cleanup.rs:295). The report gives coliseum Light **3,496** regions under 16 px, while the shipping baseline records **1,355** at [scripts/corpus-baseline.jsonl:53](/Users/shaun/dev/spritefusion-pixel-snapper/scripts/corpus-baseline.jsonl:53).

   **Impact:** The claimed 18% Potts improvement compares against a different baseline; Potts’s 2,860 regions actually exceed the shipping Light result. The promised comparison at matched cleanup strength remains unresolved.

   **Recommendation:** Use production pitch parameters, share the sieve threshold between arms, and compare outline preservation at matched residual speckle.

4. **[P2] “Within 1 px” edge agreement accepts boundaries displaced by 2 px.**

   **Evidence:** [src/eval.rs:424](/Users/shaun/dev/spritefusion-pixel-snapper/src/eval.rs:424), [src/eval.rs:441](/Users/shaun/dev/spritefusion-pixel-snapper/src/eval.rs:441). Both boundary endpoints are marked, dilated, then compared against either source-edge endpoint. A source cut at x=2 and output cut at x=4 count as preserved. Unrelated boundary orientations also qualify.

   **Impact:** The near-100% scores do not establish outline preservation. Meanwhile, the spurious-boundary metric allows no displacement, giving the two measurements inconsistent tolerances.

   **Recommendation:** Match boundary segments with explicit distance and orientation rules. Test one-pixel shifts, rejected two-pixel shifts, and unrelated crossings; apply consistent tolerance to both metrics.

5. **[P2] The Potts optimizer can stop before reaching an ICM local minimum.**

   **Evidence:** [src/eval.rs:383](/Users/shaun/dev/spritefusion-pixel-snapper/src/eval.rs:383). The claim that only current or neighbouring labels can lower energy is false after earlier updates: a pixel’s original nearest palette label can disappear locally and later become preferable again.

   **Impact:** `changed == 0` can terminate with an excluded label offering lower energy, biasing the Potts comparison.

   **Recommendation:** Evaluate all palette labels, or include each pixel’s unary-nearest label alongside current and neighbouring labels. Add a regression that checks conditional optimality when convergence is reported.

---

## Disposition (Claude)

| # | Finding | Checked | Outcome |
|---|---|---|---|
| 1 | The new pitch estimate halves the pitch on wide, low images: an axis too short to hold a comb still had its edge energy counted against that comb | Correct | Fixed: an axis that cannot hold a comb is left out of that comb's share altogether. Test `comb_pitch_survives_a_short_axis` (profiles and images, both orientations) |
| 2 | `pixel_accuracy` mapped each pixel by its corner, not its centre; a perfect crisp upscale at a fractional scale scored 87.5 % | Correct, and it mattered: the tables and two of my written conclusions changed | Fixed (centre mapping), with identity tests at six scales including fractional ones. Tables regenerated. "Blocks on the elastic grid is best or equal best in nine of twelve rows" became eleven of twelve and never below plain |
| 3 | The Potts comparison ran un-mix + sieve without the pitch scaling that shipping Smooth uses, weakening that arm (coliseum Light: 3 496 regions in my table against 1 355 in the corpus baseline) | Correct. My conclusion "Potts is the better pre-pass" rested on it | Redone with shipping's parameters and one sieve threshold per row across all arms, plus an arm with both. The conclusion is now: not consistently better in place of un-mixing; a small gain when added, at several times the cost. The document says what changed and why |
| 4 | The "within 1 px" edge metric accepted boundaries two pixels away and of any orientation, while the spurious-boundary metric allowed none | Correct | Rewritten: one pixel, same orientation, same row or column, in both directions of the comparison. Test `edge_agreement_tolerates_one_pixel_and_no_more` (exact, ±1, ±2, a crossing boundary, no boundaries) |
| 5 | The Potts optimiser tried only the current and neighbouring labels, so it could report convergence while a pixel's own nearest palette colour would still lower the energy | Correct | Each pixel's nearest label is now a candidate (which is sufficient: any other label costs more in colour and no less in smoothness). Test `potts_converges_to_a_true_local_minimum` checks every pixel against every palette label after convergence |

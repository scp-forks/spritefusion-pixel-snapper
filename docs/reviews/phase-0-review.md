# Review: Phase 0 (Bugs and a safety net)

Reviewer: Codex (gpt-6.1-sol, reasoning effort xhigh, read-only sandbox), 14 Nov 2026.
Reviewed: the uncommitted Phase 0 changes on `feature/cleanup-and-audition`, against
`docs/plans/no-resize-cleanup-and-audition.md`. Codex did not compile or run anything.

The review is reproduced unedited. Line numbers refer to the tree as reviewed, before the
fixes listed in the disposition table were applied.

---

Request changes before committing: reseeding has a correctness hole, and the regression checks leave several promises unenforced. The old penalty’s scale supports your collapse diagnosis. Reseeding itself is deterministic for fixed inputs and costs O(nk), becoming quadratic when k approaches n. No Cargo was run and no files were modified.

1. **Medium — Reseeding can empty a cluster already visited.**

   **Evidence:** [main.rs:2051](/Users/shaun/dev/spritefusion-pixel-snapper/src/main.rs:2051) checks that the selected donor owns multiple pixels, but [main.rs:2063](/Users/shaun/dev/spritefusion-pixel-snapper/src/main.rs:2063) can transfer all of them. Take vectors `[L,0,0]`, pixels `L=[0.1,0.1,0.5,0.51]`, and palette `L=[0.15,0.505,1.0]`. Reseeding entry 2 selects `0.1`, producing ownership counts `[0,2,2]`; entry 0 is never revisited.

   **Impact:** Three distinct input colours still use only two palette entries. Subsequent Lloyd refinement leaves that empty centroid unchanged. The current reseeding test misses this case.

   **Recommendation:** Queue newly emptied donors for repair, stopping when no positive-error split remains. Add this example and assert actual nearest-centroid occupancy after both reseeding and refinement.

2. **Medium — The exact corpus baseline depends on nondeterministic quantizers.**

   **Evidence:** [corpus-check.sh:50](/Users/shaun/dev/spritefusion-pixel-snapper/scripts/corpus-check.sh:50) compares every statistic exactly. The sampled quantizers consume randomized `HashMap::values()` order at [main.rs:1932](/Users/shaun/dev/spritefusion-pixel-snapper/src/main.rs:1932). This affects floating-point reductions, especially algorithm 5’s `f32` gradients; truncation also has no deterministic tie-breaker.

   **Impact:** Unchanged code can report corpus regressions on the same machine. The script’s platform/libm caveat does not cover this source of variation.

   **Recommendation:** Establish repeatability before recording exact baselines. Use deterministic ordering for corpus execution or define suitable comparisons for unstable modes, while explicitly accounting for the promise to preserve algorithm 5’s existing behaviour.

3. **Medium — Golden tests do not enforce PNG byte identity.**

   **Evidence:** [main.rs:3070](/Users/shaun/dev/spritefusion-pixel-snapper/src/main.rs:3070) decodes the returned PNG before hashing dimensions and RGBA pixels.

   **Impact:** PNG encoding, compression, or metadata can change—and therefore change the WASM return bytes—without failing the test’s stated “byte-identical” contract.

   **Recommendation:** Also record and assert hashes of the returned PNG byte vectors against the pre-change build.

4. **Low — Spatial acceptance coverage differs from the plan and overclaims feature bounds.**

   **Evidence:** [main.rs:3147](/Users/shaun/dev/spritefusion-pixel-snapper/src/main.rs:3147) constructs noisy grey pixels with sharp black features, omitting the blurred lattice promised by [plan:70](/Users/shaun/dev/spritefusion-pixel-snapper/docs/plans/no-resize-cleanup-and-audition.md:70). The “still a 2×2 feature” assertion checks only one neighbouring pixel at [main.rs:3185](/Users/shaun/dev/spritefusion-pixel-snapper/src/main.rs:3185).

   **Impact:** Feature growth in other directions passes, and preservation under blurred colour boundaries remains untested. This substitution is undeclared.

   **Recommendation:** Add the blurred-lattice case and check the complete feature component or surrounding perimeter.

5. **Low — `--stats` does not always emit valid, single-line JSON.**

   **Evidence:** [main.rs:385](/Users/shaun/dev/spritefusion-pixel-snapper/src/main.rs:385) escapes backslashes and quotes, but leaves control characters unchanged.

   **Impact:** Valid filenames containing newline, tab, or carriage return produce malformed JSON; newline also breaks the one-line guarantee and `tail -n 1` extraction.

   **Recommendation:** Escape all JSON control characters or use a serializer. Extend the test with control characters and validate the resulting JSON.

---

## Disposition (Claude)

| # | Finding | Checked | Outcome |
|---|---|---|---|
| 1 | `reseed_empty_clusters` can empty a cluster it already passed | Traced Codex's example by hand: ownership goes `[2,2,0]` → `[0,2,2]` and entry 0 stayed empty. Correct | Fixed. The function now loops until no cluster is empty or no pixel can be served better (each re-seed lowers total error, so it terminates; capped at 4k). Test `reseed_repairs_clusters_it_empties` uses the example and checks occupancy after re-seeding and after refinement |
| 2 | Corpus baseline rests on quantizers whose sample order comes from `HashMap` iteration | Correct, and older than this phase: `build_weighted_samples` returned bins in hash order, so Grey Wolf and the differentiable quantizer summed floats in a different order every process | Fixed at the source: bins are sorted by key before use, truncation ties keep that order. Test `weighted_samples_have_a_stable_order`. Baseline re-recorded (the numbers did not move) and the check passed three consecutive runs. This does change algorithm 5's output in the last float bit relative to an arbitrary earlier run; there was no single "existing behaviour" to preserve |
| 3 | Golden tests hash decoded pixels, not the PNG bytes callers receive | Correct | Fixed. Each fixture now also asserts the FNV-1a hash of the encoded PNG, recorded from the pre-change build's output files |
| 4 | Spatial test used noise, not the blurred lattice the plan named; "still 2×2" checked one neighbour | Correct on both counts | Fixed. The noise test now checks the whole 4×4 perimeter of each feature. Added `spatial_reduces_speckle_on_a_blurred_lattice` (blurred, noisy 12×12 lattice with a marked 2×2 feature): 73 regions under 16 px for plain k-means, 14 for Spatial, feature intact |
| 5 | `--stats` did not escape control characters in the file name | Correct | Fixed (`\u00XX` for anything below 0x20). Test `stats_json_escapes_control_characters` |

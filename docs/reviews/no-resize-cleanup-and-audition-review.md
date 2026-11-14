# Review: `no-resize-cleanup-and-audition` plan

Reviewer: Codex (gpt-6.1-sol, reasoning effort xhigh, read-only sandbox, web search on), 14 Nov 2026.
Reviewed: `docs/plans/no-resize-cleanup-and-audition.md` as first drafted, against `src/main.rs`,
the frontend sources and `docs/research/`. Codex did not compile or run anything.

The review is reproduced unedited below. The table at the end records what was done with each
finding (Claude). Line numbers in the review refer to the first draft of the plan and no longer
match after the revision.

---

The plan needs several contract fixes before implementation: transparency rules conflict, the grid-confidence label fails on the clean sprite sheet, and early acceptance targets depend on the deferred DP fitter. Phase 0a’s diagnosis is correct, but its test needs to cover the compressed assignment map. Most other cited code constraints check out. This was a read-only review; I did not compile or modify files.

1. **High — Blocks has contradictory alpha requirements.**

   **Evidence:** [plan:100](/Users/shaun/dev/spritefusion-pixel-snapper/docs/plans/no-resize-cleanup-and-audition.md:100) makes majority-transparent cells transparent, while [plan:125](/Users/shaun/dev/spritefusion-pixel-snapper/docs/plans/no-resize-cleanup-and-audition.md:125) requires alpha preservation. Also, `compute_profiles` substitutes black for transparent pixels; gradients crossing transparency still contribute ([main.rs:2019](/Users/shaun/dev/spritefusion-pixel-snapper/src/main.rs:2019)). “Ignored by compute_profiles” is inaccurate.

   **Impact:** A thin opaque feature surrounded by transparency can disappear. Partial-alpha pixels also need an explicit voting rule.

   **Recommendation:** For full-size Blocks, preserve each source alpha and vote RGB among pixels with `alpha > 0`. Define snap alpha separately. Test transparent-majority cells containing thin features, partial alpha, and wholly transparent inputs.

2. **High — `fit_gain` cannot mean “pixel grid found.”**

   **Evidence:** The proposed labels use fitted-versus-even residual ([plan:184](/Users/shaun/dev/spritefusion-pixel-snapper/docs/plans/no-resize-cleanup-and-audition.md:184)). The clean sprite sheet has identical fitted and even residuals, 0.0268, in [04:73](/Users/shaun/dev/spritefusion-pixel-snapper/docs/research/04-second-batch.md:73).

   **Impact:** The strongest regular lattice gets “no pixel grid found.” A perfectly reconstructed lattice can additionally produce `0 / 0`.

   **Recommendation:** Call this score “advantage over an even grid,” and separate it from lattice confidence. Test aligned regular grids, phase-shifted grids, constant images, and noise. Comparing against deliberately unaligned or phase-sampled grids would better test recoverable alignment.

3. **High — The 2 px minimum is incompatible with the proposed scaling implementation.**

   **Evidence:** [plan:86](/Users/shaun/dev/spritefusion-pixel-snapper/docs/plans/no-resize-cleanup-and-audition.md:86) promises subdivision and a hard minimum. Existing `walk` permits cuts only one pixel beyond the previous cut ([main.rs:2217](/Users/shaun/dev/spritefusion-pixel-snapper/src/main.rs:2217)); its minimum search window is 2 px regardless of pitch.

   **Impact:** A nominal pitch above 2 px does not guarantee legal cells. Subdividing a native 3 px cell by two creates a 1 px cell. Re-walking at 1.5× can likewise produce slivers or fail to become finer.

   **Recommendation:** Specify constraints on actual cut intervals, including border cells. Integer subdivision is reasonable when feasible; define what happens otherwise. Add irregular-width, short-border-cell, and 1.5× refinement tests. The perfect 8 px fixture does not exercise this.

4. **High — The early audition targets and compatibility test describe different pipelines.**

   **Evidence:** The 187–473-wide ogtrail results use source OKLab profiles, DP with cut cost, cleaned labels for majority, and raw labels for Detail ([e16:49](/Users/shaun/dev/spritefusion-pixel-snapper/docs/research/experiments/e16_second_batch.py:49)). Phase 4 instead precedes DP and requires scale 1 to equal the legacy export ([plan:196](/Users/shaun/dev/spritefusion-pixel-snapper/docs/plans/no-resize-cleanup-and-audition.md:196)). Legacy snap uses RGB k-means and votes complete RGBA tuples ([main.rs:211](/Users/shaun/dev/spritefusion-pixel-snapper/src/main.rs:211), [main.rs:2426](/Users/shaun/dev/spritefusion-pixel-snapper/src/main.rs:2426)).

   **Impact:** The reported sizes and legibility are unverified for the proposed Walk pipeline. Reusing the Blocks label voter or un-mix stage can also break scale-1 compatibility, especially with alpha.

   **Recommendation:** Explicitly preserve the legacy quantizer, grid and RGBA voter for the compatibility candidate. Either bring opt-in DP forward for audition or remeasure the actual Walk implementation and replace the inherited targets. Keep raw and cleaned label maps available for the two vote flavours.

5. **Medium — The WASM shapes are supported, but incomplete as written.**

   **Evidence:** [plan:113](/Users/shaun/dev/spritefusion-pixel-snapper/docs/plans/no-resize-cleanup-and-audition.md:113) exports `ProcessOptions` without a constructor; [plan:172](/Users/shaun/dev/spritefusion-pixel-snapper/docs/plans/no-resize-cleanup-and-audition.md:172) only comments that Candidate has getters. The pinned version is 0.2.105 ([Cargo.lock:291](/Users/shaun/dev/spritefusion-pixel-snapper/Cargo.lock:291)).

   **Impact:** `&ProcessOptions` expects a generated class instance, not a plain JS options literal. Candidate’s private fields expose no properties. Making `png` public requires a cloning getter because `Vec<u8>` is not `Copy`. **`Vec<Candidate>` itself is supported** by the [0.2.105 macro implementation](https://github.com/wasm-bindgen/wasm-bindgen/blob/0.2.105/crates/macro-support/src/codegen.rs#L380).

   **Recommendation:** Add a constructor with defaults and validation, explicit getters or `getter_with_clone`, and a byte-extraction/freeing policy. Add a generated-JS smoke test early. Preserve target gating and use `std::result::Result<_, JsValue>`: the existing local `Result` alias accepts one type parameter. See the [exported-type rules](https://wasm-bindgen.github.io/wasm-bindgen/reference/types/exported-rust-types.html).

6. **Medium — Recovering labels is viable, but `IndexedImage` needs narrower guarantees.**

   **Evidence:** [plan:71](/Users/shaun/dev/spritefusion-pixel-snapper/docs/plans/no-resize-cleanup-and-audition.md:71) stores no RGB for transparent pixels, and [plan:273](/Users/shaun/dev/spritefusion-pixel-snapper/docs/plans/no-resize-cleanup-and-audition.md:273) assumes `k ≤ 64`. The CLI accepts any positive `usize` ([main.rs:311](/Users/shaun/dev/spritefusion-pixel-snapper/src/main.rs:311)); WASM also has no upper bound.

   **Impact:** `render(index_image(img))` cannot preserve hidden transparent RGB, and an all-transparent image has an empty palette. A `u16` mapper cannot rely on the UI’s 64-colour limit. Cleanup and scoring also require original source colours, which this type does not retain.

   **Recommendation:** Keep label recovery rather than refactoring six quantizers, but retain the source alongside it. Define empty-palette rendering, deterministic palette ordering/ties, and checked overflow with a fallible mapper.

7. **Medium — Smooth’s strength and no-grid behaviour are underspecified.**

   **Evidence:** [plan:141](/Users/shaun/dev/spritefusion-pixel-snapper/docs/plans/no-resize-cleanup-and-audition.md:141) always derives pitch from detection; missing estimates fall back to approximately `min_dimension / 64` ([main.rs:2116](/Users/shaun/dev/spritefusion-pixel-snapper/src/main.rs:2116)). Strength Off only sets the sieve threshold to zero ([plan:148](/Users/shaun/dev/spritefusion-pixel-snapper/docs/plans/no-resize-cleanup-and-audition.md:148)).

   **Impact:** On a 1024² image with failed detection, fallback pitch 16 makes Light remove regions below 16 px, despite its gentle-default rationale. Off can still un-mix pixels. Smooth also inherits the detector’s rejection of dimensions below 3 px.

   **Recommendation:** Distinguish estimated pitch from fallback pitch; use absolute thresholds when grid support is absent. Define whether strength Off bypasses both stages. Name the source-based flatness calculation and treat the changed un-mix radius as a parameter requiring validation.

8. **Medium — The tests do not yet guard the research’s principal failures.**

   **Evidence:** Phase 0 tests relative fragment counts ([plan:56](/Users/shaun/dev/spritefusion-pixel-snapper/docs/plans/no-resize-cleanup-and-audition.md:56)); Phase 6 only checks coliseum residual and the sprite sheet ([plan:230](/Users/shaun/dev/spritefusion-pixel-snapper/docs/plans/no-resize-cleanup-and-audition.md:230)). The observed DP failure was wolf over-segmentation, 235×237 instead of about 200×200 ([04:64](/Users/shaun/dev/spritefusion-pixel-snapper/docs/research/04-second-batch.md:64)).

   **Impact:** Fragment counts can improve through feature erasure. The outlined tests do not enforce retention of mixed-block detail or prevent the known DP regression.

   **Recommendation:** Add three focused checks:
   - Spatial output uses `assignments[coord_to_idx[y * ww + x].unwrap()]` **after** the transparent-pixel branch, with interspersed transparent pixels and distinctive opaque colours.
   - Wolf outline connectivity/contrast and a soft-lattice cell-count bound; specify the prototype’s mean-normalized profile units for cut cost ([lab.py:185](/Users/shaun/dev/spritefusion-pixel-snapper/docs/research/experiments/lab.py:185)).
   - A mixed-block fixture where the ladder retains a sufficiently fine candidate and a marked thin feature survives.

   Bring these small corpus checks forward; the broader Phase 7 can remain later.

9. **Medium — Candidate metadata cannot position the Blocks overlay correctly.**

   **Evidence:** Candidate returns dimensions and scale ([plan:172](/Users/shaun/dev/spritefusion-pixel-snapper/docs/plans/no-resize-cleanup-and-audition.md:172)); the overlay is only `cols × rows` ([plan:209](/Users/shaun/dev/spritefusion-pixel-snapper/docs/plans/no-resize-cleanup-and-audition.md:209)). Blocks paints variable-width fitted rectangles.

   **Impact:** Uniformly enlarging the small overlay misaligns it with the full-size result.

   **Recommendation:** Return cut coordinates or rasterize the overlay into source coordinates before crossing WASM.

**Open questions**

- **Stateless versus session:** What latency budget applies to changing only width or vote? Stateless batching is sufficient initially, but each call repeats quantization. Measure those interactions before introducing a session. Cache identity must include requested geometry and fitter; thumbnail URLs also need protection from the cache’s single-entry pinning policy ([result-cache.ts:34](/Users/shaun/dev/spritefusion-pixel-snapper/spritefusion-frontend/src/utils/result-cache.ts:34)).

- **Uniform sizes on non-lattice inputs:** I disagree with generalizing the coliseum’s fitted-grid advantage to city/cat. Compare uniform and fitted candidates at equal pixel counts and identical voting there. An actual uniform fallback remains worth testing.

- **Potts:** Deferring it is reasonable, but the experiment does not reject the earlier edge-weighted proposal: [lab.py:299](/Users/shaun/dev/spritefusion-pixel-snapper/docs/research/experiments/lab.py:299) uses spatially constant neighbour weights. A matched cleanup-versus-outline-preservation comparison, including Rust/WASM timing, would settle whether edge weighting adds enough value.

---

## Disposition (Claude)

Every finding was checked against the code or the research data before being accepted.

| # | Finding | Checked | Outcome in the plan |
|---|---|---|---|
| 1 | Contradictory alpha rules in Blocks; `compute_profiles` treats transparent pixels as black, not "ignored" | `main.rs:2019` returns 0.0 for alpha 0, so transparency edges do add gradient. Correct | Accepted. Full-size Blocks keeps every source pixel's alpha and votes colour among pixels with alpha > 0. Small snap output keeps the legacy RGBA vote. Constraint reworded. Three alpha tests added |
| 2 | `fit_gain` (fitted vs even grid) labels the clean sprite sheet "no pixel grid" | Correct, and it is an error in the research, not only the plan: `04-second-batch.md` shows 0.0268 vs 0.0268 for the sprite sheet and still called this number the measure of "is this pixel art" | Accepted. Two numbers now: **lattice score** (fitted grid vs deliberately unaligned grids; the E9 formula, 0.52 on the sprite sheet) for "is there a grid", and **irregularity** (fitted vs even) only as a diagnostic. `03` and `04` corrected. Division-by-zero guard and four tests added |
| 3 | 2 px minimum cannot be guaranteed by subdividing or by `walk` | `main.rs:2217`: `walk` allows a cut 1 px after the previous one. Correct | Accepted. The minimum is enforced on actual intervals after every grid operation; subdivision is per cell and capped by the cell's width; rungs that fail to get finer are dropped. Tests on irregular widths added |
| 4 | Research numbers came from a different pipeline (OKLab profiles, DP fit, cut cost) than Phases 2–4 will build (`walk` on the quantized image); scale-1 compatibility needs the legacy voter | Correct. The inherited targets (187–473 wide on ogtrail, region counts) are not predictions for the `walk` pipeline | Accepted. The compatibility candidate runs the legacy path untouched. Absolute targets replaced by ones that hold for either fitter, and Phase 2 starts by remeasuring. DP stays opt-in but moves ahead of the audition UI so the ladder can be compared under both fitters |
| 5 | WASM shapes incomplete: no constructor, no getters, `Vec<u8>` field needs a cloning getter, local `Result` alias takes one parameter | `Cargo.lock` pins 0.2.105; `Vec<Candidate>` is supported. Correct | Accepted. Constructor with defaults and validation, `getter_with_clone`, explicit `std::result::Result`, and a generated-JS smoke test in Phase 2 |
| 6 | `IndexedImage` loses hidden RGB of transparent pixels, has no source colours, assumes k ≤ 64 | `parse_args` accepts any positive k (`main.rs:311`). Correct | Accepted. Source image is carried alongside; transparent pixels are passed through from the source at render; palette order is first appearance in raster order; more than 65 535 colours is an error |
| 7 | Smooth: fallback pitch (min dimension / 64) would make "Light" remove 16 px regions; strength Off still un-mixes | `resolve_step_sizes` `(None, None)` branch, `main.rs:2116`. Correct | Accepted. `detect_grid` reports whether the step was estimated or a fallback; cell-relative thresholds apply only to estimated steps; strength has three levels and Cleanup Off bypasses everything; images under 3 px skip grid detection |
| 8 | Tests do not guard the failures research actually found | Fair | Accepted. Added: Spatial test with interspersed transparent pixels; outline connectivity and contrast test; soft-lattice cell-count bound for the DP fitter; mixed-block fixture. A six-image corpus check moves into Phase 0 |
| 9 | A `cols × rows` overlay cannot be aligned with variable-width Blocks cells | Correct | Accepted. The overlay is rasterised by Rust into source coordinates for full-size output |
| Q1 | Stateless vs session; cache pinning protects one entry only | `result-cache.ts:34` pins a single key. Correct | Stateless first, with timings recorded for "change only the size" and "change only the vote". Cache gains a pinned group for the visible candidate strip |
| Q2 | Fitted-grid advantage should not be generalised to non-lattice inputs; test a uniform fallback | Research data agrees more than the plan said: on the city the fitted grid beats an even one by only 2–7 % | Accepted. When the lattice score is low, a typed size is honoured exactly with an even grid |
| Q3 | Potts was tested with constant neighbour weights only; edge-weighted Potts is not refuted | Correct: `lab.potts_icm` uses a constant weight | Not in the plan. Recorded as an open experiment; the claim in `03` is narrowed to what was tested |

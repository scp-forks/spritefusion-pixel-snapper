# Review: Phase 2 (Blocks)

Reviewer: Codex (gpt-6.1-sol, reasoning effort xhigh, read-only sandbox), 14 Nov 2026.
Reviewed: the uncommitted Phase 2 changes on `feature/cleanup-and-audition`, against
`docs/plans/no-resize-cleanup-and-audition.md`. Codex did not compile or run anything.

The review is reproduced unedited. Line numbers refer to the tree as reviewed, before the
fixes listed in the disposition table were applied.

---

I would fix the two contract violations below before committing Phase 2. The review also found gaps in the claimed test coverage and misleading grid metadata. This was read-only; I ran the generated-package smoke test and in-memory WASM probes, but no Cargo commands.

1. **Medium — Blocks bypasses the 2 px minimum at the base grid and after fallback.**

   **Evidence:** [cleanup.rs:24](/Users/shaun/dev/spritefusion-pixel-snapper/src/cleanup.rs:24) returns `base` unchanged at scale 1 and when every refinement fails; [cleanup.rs:49](/Users/shaun/dev/spritefusion-pixel-snapper/src/cleanup.rs:49) passes that grid directly to voting. Confirmed with a 32×24 opaque checkerboard: Blocks retains all 768 isolated one-pixel regions at scales 1 and 2.

   **Impact:** The new Blocks path violates the advertised minimum and can preserve precisely the speckle it promises to remove.

   **Recommendation:** Merge narrow intervals in the Blocks base grid and fallback result. Keep the legacy snap grid untouched. Add coverage where refinement returns `None`; the existing width-5 fallback test actually succeeds through capped subdivision.

2. **Medium — JS integer options are converted before validation.**

   **Evidence:** Public `u32` fields at [main.rs:368](/Users/shaun/dev/spritefusion-pixel-snapper/src/main.rs:368) lose their original JS values before [main.rs:398](/Users/shaun/dev/spritefusion-pixel-snapper/src/main.rs:398) validates them. Confirmed: `algorithm = NaN` and `algorithm = 4294967296` become `0`; `cleanup = 1.5` becomes `1`; `k_colors = 4294967312` becomes `16`. Processing succeeds.

   **Impact:** Invalid options silently select different behavior, contrary to the explicit rejection contract. Native validation tests cannot catch this conversion.

   **Recommendation:** Validate finite integer JS values before conversion, using checked setters or raw numeric storage validated by `process_image_with`. Extend the JS smoke cases to fractions, non-finite values, negatives, and overflow.

3. **Medium — The smoke test never verifies PNG decoding.**

   **Evidence:** [wasm-smoke.mjs:24](/Users/shaun/dev/spritefusion-pixel-snapper/scripts/wasm-smoke.mjs:24) checks the signature and reads dimensions from fixed offsets. It never decodes image data. [corpus-check.sh:48](/Users/shaun/dev/spritefusion-pixel-snapper/scripts/corpus-check.sh:48) relies on this script for WASM coverage.

   **Impact:** Corrupt or truncated image data with plausible headers can pass the promised “PNG decodes” check.

   **Recommendation:** Decode each processing result with a PNG decoder and assert decoded dimensions and pixel-buffer length.

4. **Low — Partial-alpha voting lacks a discriminating test.**

   **Evidence:** [cleanup.rs:163](/Users/shaun/dev/spritefusion-pixel-snapper/src/cleanup.rs:163) tests only alpha 0/255 voting; [cleanup.rs:274](/Users/shaun/dev/spritefusion-pixel-snapper/src/cleanup.rs:274) checks alpha preservation without asserting the winning colour.

   **Impact:** Ignoring partial-alpha pixels or weighting votes by alpha could pass these tests despite violating the plan.

   **Recommendation:** Add a known cell containing two alpha-1 pixels of colour A and one alpha-255 pixel of colour B; A must win. Place it in a multi-row irregular grid with transparent holes and assert the complete painted output.

5. **Low — The grid-doubling test does not establish that the lattice remains intact.**

   **Evidence:** [cleanup.rs:234](/Users/shaun/dev/spritefusion-pixel-snapper/src/cleanup.rs:234) creates a blurred, noisy input and checks only centre-pixel recovery through [test_util.rs:86](/Users/shaun/dev/spritefusion-pixel-snapper/src/test_util.rs:86).

   **Impact:** Incorrect colours elsewhere in each cell, displaced boundaries, or fringe regions can pass a test claiming that refinement leaves a clean lattice unchanged.

   **Recommendation:** Assert full-image equality on an exactly explained lattice, using explicit cuts if necessary to isolate refinement from the acknowledged estimator limitation. Rename the existing test to describe centre-colour recovery.

6. **Low — Output statistics present the requested scale as achieved.**

   **Evidence:** [stats.ts:41](/Users/shaun/dev/spritefusion-pixel-snapper/spritefusion-frontend/src/components/stats.ts:41) displays cached requested settings as “grid 3x finer,” although [cleanup.rs:28](/Users/shaun/dev/spritefusion-pixel-snapper/src/cleanup.rs:28) may return a coarser grid or the base grid.

   **Impact:** Identical outputs can receive different claimed grid refinements, obscuring fallback behavior.

   **Recommendation:** Label this as the requested scale, or return actual grid metadata from WASM and display that.

---

## Disposition (Claude)

| # | Finding | Checked | Outcome |
|---|---|---|---|
| 1 | Blocks used the detected grid as-is at scale 1 and on fallback, so 1 px cells voted for themselves | Correct. Reproduced Codex's 1 px checkerboard: all 768 specks survived | Fixed. `Grid::with_min_cells` merges cells under 2 px, and Blocks applies it to the detected grid before anything else; the legacy snap grid is untouched. Test `one_pixel_cells_are_merged_before_voting` (scales 1, 2, 3). The fallback test now asserts that `refine_grid` really returns `None` in its last case. On the corpus only the coliseum moved: 188×100 → 187×99 |
| 2 | wasm-bindgen converts JS numbers to `u32` before validation: NaN → 0, 1.5 → 1, 2³² + 16 → 16 | Correct, and invisible to native tests | Fixed. Numeric option fields are `f64` and go through `checked_u32` (finite, whole, in range) before use. Native test plus 18 new cases in the JS smoke test |
| 3 | The smoke test read PNG dimensions from fixed offsets without decoding | Correct | Fixed. It now walks the chunks, checks every CRC, inflates the IDAT stream, and checks the scanline byte count and filter types against the header |
| 4 | No test distinguished "partial-alpha pixels vote with weight 1" from ignoring or alpha-weighting them | Correct | Test `partial_alpha_pixels_vote_with_full_weight`: two alpha-1 pixels outvote one alpha-255 pixel, on an irregular multi-row grid with transparent holes, compared against the complete expected image |
| 5 | The grid-doubling test only looked at cell centres on a noisy image | Correct | Added `an_exact_lattice_is_unchanged_on_its_grid_and_on_finer_ones` (full-image equality at scales 1, 2, 3 on explicit cuts) and renamed the existing test to say what it checks |
| 6 | The stats panel showed the requested grid scale as if achieved | Correct | The label now says "…x finer grid requested". The grid actually used reaches the frontend with the candidates of Phase 5; the CLI's `--stats` already reports it |

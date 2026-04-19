# Review: `bubbly-twirling-bunny` Plan

## Scope

Reviewed:
- Plan file: `/Users/shaun/.claude/plans/bubbly-twirling-bunny.md`
- Rust core: `src/main.rs`
- Frontend wiring: `spritefusion-frontend/src/components/controls.ts`, `spritefusion-frontend/src/components/stats.ts`, `spritefusion-frontend/src/wasm.ts`, `spritefusion-frontend/src/main.ts`

Baseline checks run before review:
- `cargo test -q` (pass)
- `npx tsc --noEmit` in `spritefusion-frontend` (pass)

## Findings (ordered by severity)

### 1) High: Main-thread freeze risk is not addressed for the proposed algorithm costs

**Evidence**
- Plan explicitly deprioritizes performance and sets heavy defaults (`line 11`, `line 79`, `line 106`).
- Frontend currently executes WASM quantization synchronously on the UI thread (`spritefusion-frontend/src/wasm.ts:21-23`, `spritefusion-frontend/src/main.ts:120-131`).

**Impact**
- Larger images in no-resize mode can block the browser tab for long periods, making the UI appear hung.
- The "spinner render" workaround in `main.ts` only helps before the blocking call starts; it does not solve responsiveness during processing.

**Recommendation**
- Add an explicit compute budget strategy to the plan:
  - Image-size-based iteration caps.
  - Quality presets (fast/normal/high).
  - Preferred: move processing to a Web Worker for no-resize algorithms 3-5.

Authors Note: Defer this until I do some testing - I have a powerful machine - it might run perfectly fast enough.

### 2) High: Spatial memory strategy is internally inconsistent and under-budgeted

**Evidence**
- Plan defines large structures for spatial quantization (`lines 49-53`) and only caps `soft_assign` to 16 MB (`line 70`).
- The same algorithm depends on spatial neighbor coupling (`line 62`), but subsampling is proposed when `n_opaque * k` is large (`line 70`).
- Repo allows images up to `10000 x 10000` (`src/main.rs:189`).

**Impact**
- Real memory use is much higher than 16 MB once pixel buffers, index maps, and temporary arrays are included.
- Subsampling can break neighborhood semantics and degrade/unstabilize the very spatial behavior the algorithm is meant to provide.

**Recommendation**
- Replace this with a full-memory budget section in the plan (all buffers, not only `soft_assign`).
- Use tile-based optimization with overlap/halo regions instead of naive global subsampling. Authors Note: DO NOT DO THIS IF IT WILL MAKE THE ALGORITHM WORSE.
- Specify a deterministic fallback when memory budget is exceeded.

### 3) High: Numerical-stability and degeneracy handling are missing from the algorithm specs

**Evidence**
- Spatial M-step updates centroids from soft weights but does not define zero-weight handling (`line 66`).
- Differentiable optimizer defines unconstrained gradient updates without finite checks/clipping (`lines 116-119`).
- Output conversion path assumes valid finite OKLab values (`src/main.rs:121-138`).

**Impact**
- Empty clusters, exploding logits, or NaNs can yield invalid palettes and silently collapse colors.
- This would be hard to detect with the currently proposed tests (mostly structural checks).

**Recommendation**
- Add explicit safeguards to the plan:
  - Empty-cluster reinitialization rule.
  - Palette/logit clamping bounds.
  - `is_finite()` checks with deterministic recovery/fallback.
  - Early-stop on divergence.

### 4) Medium: Test plan does not validate the strongest claims (quality/variety/back-compat)

**Evidence**
- Plan goals emphasize quality and visual variety (`line 11`) and claims backward compatibility (`line 160`).
- Proposed tests are mostly tiny synthetic property checks (`lines 125-136`), mirroring current small-image unit tests in `src/main.rs:1701-1922`.

**Impact**
- Regressions in real-image quality can pass undetected.
- "Existing algorithms unchanged" is not verified by a baseline fixture comparison.

**Recommendation**
- Add fixture-based regression tests:
  - Golden outputs for algorithms 0-2 on fixed images (to verify back-compat claim).
  - Snapshot metrics (SSIM/PSNR or at least deterministic checksum comparisons) for new algorithms on representative sprites.
  - One medium-size no-resize smoke test per new algorithm to guard extreme runtime regressions.

### 5) Low: CLI/docs consistency updates are missing from the plan

**Evidence**
- CLI parser comment currently documents only `--algorithm=<0|1|2>` (`src/main.rs:312`).
- Plan updates enum/UI but does not include docs/help update tasks.

**Impact**
- User-facing behavior changes without synchronized documentation.

**Recommendation**
- Add explicit checklist items for:
  - CLI help/comment updates.
  - README/no-resize algorithm documentation.

## Open Questions / Assumptions

1. Are long-running no-resize algorithms acceptable on the main thread, or should responsiveness be a hard requirement? Yes, up to 30 seconds. We can adjust to web workers if it turns out this is an issue in reality.
2. Is strict determinism required across targets (native + wasm), or only per-target reproducibility? I don't understand this question.
3. Should algorithm 3-5 be exposed in CLI docs immediately, or gated behind an experimental flag first? CLI support is fine, but GUI is most important.

## Overall Assessment

The plan is directionally strong and maps to the right integration points in this repo, but it is not implementation-safe yet. The blocking risks are compute-budgeting, memory modeling (especially spatial), and numerical stability guarantees. I recommend revising those sections before coding starts.

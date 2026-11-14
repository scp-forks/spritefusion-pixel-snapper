# Review: Phase 5 (Candidates)

Reviewer: Codex (gpt-6.1-sol, reasoning effort xhigh, read-only sandbox), 14 Nov 2026.
Reviewed: the uncommitted Phase 5 changes on `feature/cleanup-and-audition`, against
`docs/plans/no-resize-cleanup-and-audition.md`. Codex did not compile or run anything.

The review is reproduced unedited. Line numbers refer to the tree as reviewed, before the
fixes listed in the disposition table were applied.

---

Phase 5 needs revision before commit: exact typed widths can disappear from the audition, cache eviction can invalidate incoming thumbnails, and the detail-retention test accepts the wrong colour. I also found a selection bug when resubmitting a width. Review was read-only; I did not run cargo.

1. **Medium — Deduplication discards explicitly requested widths.**

   **Evidence:** [src/candidates.rs:115](/Users/shaun/dev/spritefusion-pixel-snapper/src/candidates.rs:115) suppresses a typed rung when any existing width is within 5%, regardless of height or cuts. Reproduced through WASM: city image, DP, target 102 returns the unchanged ladder starting at **100×95**, while `process_image_with` returns **102×102**. The exact-width assertions at [src/candidates.rs:419](/Users/shaun/dev/spritefusion-pixel-snapper/src/candidates.rs:419) exercise only single-result calls.

   **Impact:** The audition violates the promised exact width on low-score images. This additional width-based deduplication is an undeclared plan deviation.

   **Recommendation:** Preserve exact manual geometry on low-score images; deduplicate only genuinely equivalent results. Add batch tests for widths near existing rungs, including rectangular images.

2. **Medium — Incoming sets can be evicted before being displayed.**

   **Evidence:** [result-cache.ts:71](/Users/shaun/dev/spritefusion-pixel-snapper/spritefusion-frontend/src/utils/result-cache.ts:71) evicts during insertion, while [main.ts:324](/Users/shaun/dev/spritefusion-pixel-snapper/spritefusion-frontend/src/main.ts:324) inserts before `showResult` pins the incoming set. With an 80 MiB pinned set and a 60 MiB incoming set, the cache revokes every incoming URL.

   **Impact:** The comparison and strip receive revoked URLs. The inherited insertion order becomes particularly problematic when each entry contains several full-size Blocks outputs.

   **Recommendation:** Make insertion and transfer of the visible-set pin atomic, then evict. Test the case where the old and incoming sets together exceed the byte budget.

3. **Medium — The detail-survival assertion accepts background green as the marker.**

   **Evidence:** [src/candidates.rs:394](/Users/shaun/dev/spritefusion-pixel-snapper/src/candidates.rs:394) checks only `red < 70`. The fixture includes background colour `[50,160,70]` ([src/test_util.rs:119](/Users/shaun/dev/spritefusion-pixel-snapper/src/test_util.rs:119)), which satisfies that predicate.

   **Impact:** Recolouring the dark line green can pass the claimed survival check; a solid green output also satisfies that predicate once the grid condition holds.

   **Recommendation:** Check distance from the marker’s complete colour and contrast against neighbouring pixels. Verify that removing or recolouring the marker makes the test fail.

4. **Low — Resubmitting the same width does not select it.**

   **Evidence:** [main.ts:213](/Users/shaun/dev/spritefusion-pixel-snapper/spritefusion-frontend/src/main.ts:213) recognizes a width submission only when its numeric value differs from the current set’s target.

   **Impact:** Submit 48, select the 72-wide thumbnail, then submit 48 again: the cache hit preserves 72 instead of selecting 48.

   **Recommendation:** Track an explicit width submission separately from settings differences, and use it to select the requested result on cache hits too.

---

## Disposition (Claude)

| # | Finding | Checked | Outcome |
|---|---|---|---|
| 1 | A typed width was dropped from the audition whenever another rung was within 5 % of it, even on images where the width is promised exactly (city, DP, 102 → ladder unchanged at 100×95) | Correct, and an undeclared deviation | Fixed. Where no pixel grid was found the typed rung is always offered unless it is an existing rung cut for cut. Where one was found the width is a request, and a rung within a twentieth of it counts as the answer; that rule is now declared in the plan. Batch test over widths two below, equal to, and one and two above every rung of a rectangular image |
| 2 | A new set was stored, and evicted, before it was pinned: with an 80 MiB set on screen and a 60 MiB set arriving, every URL of the new set was revoked before display | Correct | Fixed. `cache.set(set, { pin: true })` moves the pin before evicting. The frontend had no tests; it now has 14 (`npm test`, Node's own runner, no new dependency), including this case, and they run from `scripts/corpus-check.sh` |
| 3 | The "line survives" assertion only checked `red < 70`, which the fixture's own background green satisfies | Correct | Fixed. The check now needs a pixel dark on all three channels with non-dark pixels either side, on at least 150 rows, and the test proves the check can fail by running it on copies with the line erased and recoloured green |
| 4 | Submitting the same width again after clicking another thumbnail did not re-select it | Correct | Fixed. A submission sets a flag that the next shown set consumes, cached or not. The selection rule moved to `src/selection.ts` (pure, tested) |

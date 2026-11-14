import type { CachedResult, ResultSet } from './utils/result-cache';

/** Index of the result whose grid width is nearest `cols`; -1 if the set has no grid. */
export function nearestByWidth(set: ResultSet, cols: number): number {
  let best = -1;
  set.results.forEach((r, i) => {
    if (r.cols <= 0) return;
    if (best < 0 || Math.abs(r.cols - cols) < Math.abs(set.results[best].cols - cols)) {
      best = i; // ties keep the first (coarser) one
    }
  });
  return best;
}

/**
 * Which result of a set to put in the comparison view when the set is shown.
 *
 * 1. The size the user just asked for (`selectTyped`), whether the set was computed
 *    for it or came out of the cache.
 * 2. The size they were looking at, or the nearest to it within a tenth, so that
 *    changing the colour count or the grid fit, which shift every size by a few
 *    cells, does not throw their choice away.
 * 3. The detected grid, which in snap mode with the standard fitter is exactly what
 *    the tool has always produced.
 */
export function chooseSelection(
  set: ResultSet,
  previous: CachedResult | null,
  selectTyped: boolean,
): number {
  if (selectTyped && set.settings.targetWidth > 0) {
    const typed = nearestByWidth(set, set.settings.targetWidth);
    if (typed >= 0) return typed;
  }
  if (previous && previous.cols > 0) {
    const nearest = nearestByWidth(set, previous.cols);
    if (
      nearest >= 0 &&
      Math.abs(set.results[nearest].cols - previous.cols) <= previous.cols * 0.1
    ) {
      return nearest;
    }
  }
  const detected = set.results.findIndex((r) => r.isLegacy || r.gridScale === 1);
  return detected >= 0 ? detected : 0;
}

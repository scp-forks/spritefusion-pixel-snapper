import { test } from 'node:test';
import assert from 'node:assert/strict';
import { OVERLAY_COLOR, OVERLAY_STRENGTH, rasterizeOverlay } from '../src/overlay.ts';

/** Alpha of every pixel, as rows of numbers. */
function alphas(data: Uint8ClampedArray, width: number, height: number): number[][] {
  return Array.from({ length: height }, (_, y) =>
    Array.from({ length: width }, (_, x) => data[(y * width + x) * 4 + 3]),
  );
}

const a = (v: number) => Math.round(v * OVERLAY_STRENGTH);

test('a full-size result: every cell fills its own rectangle between the cuts', () => {
  // 3 x 2 cells of different sizes on an 8 x 5 image.
  const data = rasterizeOverlay({
    width: 8,
    height: 5,
    colCuts: Uint32Array.of(0, 2, 5, 8),
    rowCuts: Uint32Array.of(0, 3, 5),
    values: Uint8Array.of(10, 0, 250, 100, 200, 0),
  });
  assert.deepEqual(alphas(data, 8, 5), [
    [a(10), a(10), 0, 0, 0, a(250), a(250), a(250)],
    [a(10), a(10), 0, 0, 0, a(250), a(250), a(250)],
    [a(10), a(10), 0, 0, 0, a(250), a(250), a(250)],
    [a(100), a(100), a(200), a(200), a(200), 0, 0, 0],
    [a(100), a(100), a(200), a(200), a(200), 0, 0, 0],
  ]);
  // Tinted pixels carry the overlay colour; untinted ones are fully clear.
  assert.deepEqual([...data.slice(0, 3)], [...OVERLAY_COLOR]);
  assert.deepEqual([...data.slice(2 * 4, 2 * 4 + 4)], [0, 0, 0, 0]);
});

test('a snapped result: one pixel per cell, in row-major order, ignoring the cuts', () => {
  // A 3 x 2 snapped image of a 90 x 40 source: the cuts are in source pixels and say
  // nothing about where to paint; a transposed map would show here.
  const data = rasterizeOverlay({
    width: 3,
    height: 2,
    colCuts: Uint32Array.of(0, 28, 61, 90),
    rowCuts: Uint32Array.of(0, 19, 40),
    values: Uint8Array.of(1, 2, 3, 4, 5, 6).map((v) => v * 40),
  });
  assert.deepEqual(alphas(data, 3, 2), [
    [a(40), a(80), a(120)],
    [a(160), a(200), a(240)],
  ]);
});

test('matching the cell count on one axis only is still a full-size result', () => {
  // 4 cells across a 4 px wide image would be 1 px cells, which the engine never
  // makes; but if width happens to equal cols and height does not equal rows, the
  // cuts must be used, not the one-pixel-per-cell shortcut.
  const data = rasterizeOverlay({
    width: 4,
    height: 6,
    colCuts: Uint32Array.of(0, 1, 2, 3, 4),
    rowCuts: Uint32Array.of(0, 2, 6),
    values: Uint8Array.of(50, 0, 0, 0, 0, 0, 0, 150),
  });
  assert.deepEqual(alphas(data, 4, 6), [
    [a(50), 0, 0, 0],
    [a(50), 0, 0, 0],
    [0, 0, 0, a(150)],
    [0, 0, 0, a(150)],
    [0, 0, 0, a(150)],
    [0, 0, 0, a(150)],
  ]);
});

test('cuts beyond the image are clipped, and an empty grid paints nothing', () => {
  const clipped = rasterizeOverlay({
    width: 3,
    height: 2,
    colCuts: Uint32Array.of(0, 2, 9),
    rowCuts: Uint32Array.of(0, 7),
    values: Uint8Array.of(0, 255),
  });
  assert.equal(clipped.length, 3 * 2 * 4);
  assert.deepEqual(alphas(clipped, 3, 2), [
    [0, 0, a(255)],
    [0, 0, a(255)],
  ]);
  const empty = rasterizeOverlay({
    width: 2,
    height: 2,
    colCuts: new Uint32Array(0),
    rowCuts: new Uint32Array(0),
    values: new Uint8Array(0),
  });
  assert.ok(empty.every((v) => v === 0));
});

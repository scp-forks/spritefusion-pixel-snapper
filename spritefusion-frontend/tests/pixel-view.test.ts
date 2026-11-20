import { test } from 'node:test';
import assert from 'node:assert/strict';
import { MAX_ZOOM, cssSize, expandCells, fitZoom, thumbZoom } from '../src/pixel-view.ts';

test('fit is the largest whole number of device pixels per pixel that fits', () => {
  // 128 px into 800 x 600 CSS px.
  assert.equal(fitZoom(128, 128, 800, 600, 1), 4); // 600 / 128 = 4.69
  assert.equal(fitZoom(128, 128, 800, 600, 2), 9); // 1200 / 128 = 9.4
  assert.equal(fitZoom(128, 64, 800, 600, 1), 6); // width-bound: 800 / 128 = 6.25
  // Exactly fitting counts.
  assert.equal(fitZoom(100, 100, 400, 400, 1), 4);
  // Fractional device pixel ratios still give a whole number of device pixels.
  assert.equal(fitZoom(100, 100, 400, 400, 1.25), 5);
});

test('fit never scales an image down: too large means one device pixel and scrolling', () => {
  assert.equal(fitZoom(2172, 724, 800, 600, 1), 1);
  assert.equal(fitZoom(2172, 724, 800, 600, 2), 1);
  assert.equal(fitZoom(0, 10, 800, 600, 1), 1);
});

test('fit is capped', () => {
  assert.equal(fitZoom(2, 2, 4000, 4000, 2), MAX_ZOOM);
});

test('CSS size is a whole number of device pixels per image pixel', () => {
  assert.deepEqual(cssSize(128, 64, 3, 1), { width: 384, height: 192 });
  // At 2x, 3 device pixels per image pixel is 1.5 CSS pixels.
  assert.deepEqual(cssSize(128, 64, 3, 2), { width: 192, height: 96 });
});

test('thumbnails zoom by whole numbers or report that they must shrink', () => {
  assert.equal(thumbZoom(32, 16, 112, 1), 3);
  assert.equal(thumbZoom(32, 16, 112, 2), 7);
  assert.equal(thumbZoom(112, 50, 112, 1), 1);
  assert.equal(thumbZoom(113, 50, 112, 1), null);
  assert.equal(thumbZoom(305, 97, 112, 2), null);
});

/** RGBA bytes with pixel n = [n, n, n, 255]. */
function numbered(count: number): Uint8ClampedArray {
  const px = new Uint8ClampedArray(count * 4);
  for (let n = 0; n < count; n++) px.set([n, n, n, 255], n * 4);
  return px;
}

/** The first channel of every pixel, as rows. */
function grid(px: Uint8ClampedArray, width: number, height: number): number[][] {
  return Array.from({ length: height }, (_, y) =>
    Array.from({ length: width }, (_, x) => px[(y * width + x) * 4]),
  );
}

test('each output pixel fills its own cell of the source, cells of any size', () => {
  // A 3 x 2 snapped output of an 8 x 5 source, uneven cells.
  const out = expandCells(
    numbered(6),
    3,
    2,
    Uint32Array.of(0, 2, 5, 8),
    Uint32Array.of(0, 3, 5),
    8,
    5,
  );
  assert.ok(out);
  assert.deepEqual(grid(out, 8, 5), [
    [0, 0, 1, 1, 1, 2, 2, 2],
    [0, 0, 1, 1, 1, 2, 2, 2],
    [0, 0, 1, 1, 1, 2, 2, 2],
    [3, 3, 4, 4, 4, 5, 5, 5],
    [3, 3, 4, 4, 4, 5, 5, 5],
  ]);
});

test('transparent pixels stay fully transparent, colours are copied exactly', () => {
  const px = Uint8ClampedArray.of(10, 20, 30, 255, 0, 0, 0, 0);
  const out = expandCells(px, 2, 1, Uint32Array.of(0, 1, 3), Uint32Array.of(0, 2), 3, 2);
  assert.ok(out);
  assert.deepEqual(
    [...out],
    [10, 20, 30, 255, 0, 0, 0, 0, 0, 0, 0, 0, 10, 20, 30, 255, 0, 0, 0, 0, 0, 0, 0, 0],
  );
});

test('an unaligned input buffer is handled', () => {
  const backing = new Uint8ClampedArray(1 + 8);
  const px = backing.subarray(1);
  px.set([1, 2, 3, 255, 4, 5, 6, 255]);
  const out = expandCells(px, 2, 1, Uint32Array.of(0, 1, 2), Uint32Array.of(0, 1), 2, 1);
  assert.deepEqual([...out!], [1, 2, 3, 255, 4, 5, 6, 255]);
});

test('cuts that do not describe the result are refused', () => {
  const px = numbered(4);
  const cols = Uint32Array.of(0, 2, 4);
  const rows = Uint32Array.of(0, 2, 4);
  assert.ok(expandCells(px, 2, 2, cols, rows, 4, 4));
  assert.equal(expandCells(px, 2, 2, cols, rows, 5, 4), null, 'wrong source width');
  assert.equal(expandCells(px, 2, 2, Uint32Array.of(0, 4), rows, 4, 4), null, 'too few cells');
  assert.equal(expandCells(px, 2, 2, Uint32Array.of(1, 2, 4), rows, 4, 4), null, 'not from 0');
  assert.equal(expandCells(px, 2, 2, new Uint32Array(0), new Uint32Array(0), 4, 4), null, 'no grid');
});

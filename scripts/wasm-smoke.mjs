// Smoke test of the generated WASM package, as JavaScript sees it.
//
//   wasm-pack build --target web --out-dir pkg --release
//   node scripts/wasm-smoke.mjs
//
// The Rust unit tests cover behaviour. This covers what they cannot: that the exported
// names exist, that the options class can be constructed and set from JS, that typed
// arrays cross the boundary intact, and that invalid options are rejected with an
// error rather than a trap.
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import assert from 'node:assert/strict';
import { crc32, inflateSync } from 'node:zlib';

const root = fileURLToPath(new URL('..', import.meta.url));
const pkg = await import(`${root}pkg/spritefusion_pixel_snapper.js`);
await pkg.default({
  module_or_path: readFileSync(`${root}pkg/spritefusion_pixel_snapper_bg.wasm`),
});

const fixture = new Uint8Array(readFileSync(`${root}tests/fixtures/wolf_ear_128.png`));
const PNG_MAGIC = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];

/**
 * Decode a PNG far enough to know it is intact: every chunk's CRC, a complete IDAT
 * stream that inflates, the right number of scanline bytes for the declared size, and
 * a valid filter type on every scanline. Returns [width, height].
 */
function pngSize(bytes) {
  assert.deepEqual([...bytes.slice(0, 8)], PNG_MAGIC, 'output is not a PNG');
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let header = null;
  let sawEnd = false;
  const idat = [];
  for (let pos = 8; pos < bytes.length; ) {
    const length = view.getUint32(pos);
    const type = String.fromCharCode(...bytes.slice(pos + 4, pos + 8));
    const data = bytes.slice(pos + 8, pos + 8 + length);
    assert.equal(data.length, length, `${type} chunk is truncated`);
    assert.equal(
      view.getUint32(pos + 8 + length),
      crc32(bytes.slice(pos + 4, pos + 8 + length)),
      `${type} chunk has a bad CRC`,
    );
    if (type === 'IHDR') {
      const d = new DataView(data.buffer, data.byteOffset, data.byteLength);
      header = {
        width: d.getUint32(0),
        height: d.getUint32(4),
        bitDepth: data[8],
        colorType: data[9],
        interlace: data[12],
      };
    } else if (type === 'IDAT') {
      idat.push(data);
    } else if (type === 'IEND') {
      sawEnd = true;
    }
    pos += 12 + length;
  }
  assert.ok(header, 'no IHDR chunk');
  assert.ok(sawEnd, 'no IEND chunk');
  assert.equal(header.interlace, 0, 'interlaced PNG not expected');
  const channels = { 0: 1, 2: 3, 3: 1, 4: 2, 6: 4 }[header.colorType];
  assert.ok(channels, `unknown colour type ${header.colorType}`);
  const rowBytes = Math.ceil((header.width * channels * header.bitDepth) / 8);
  const raw = inflateSync(Buffer.concat(idat));
  assert.equal(raw.length, header.height * (rowBytes + 1), 'pixel data has the wrong length');
  for (let y = 0; y < header.height; y++) {
    assert.ok(raw[y * (rowBytes + 1)] <= 4, `scanline ${y} has an invalid filter type`);
  }
  return [header.width, header.height];
}

function run(configure) {
  const opts = new pkg.ProcessOptions();
  try {
    configure(opts);
    return pkg.process_image_with(fixture, opts);
  } finally {
    opts.free();
  }
}

// 1. Defaults are readable from JS and are the documented ones.
{
  const opts = new pkg.ProcessOptions();
  assert.deepEqual(
    [opts.k_colors, opts.no_resize, opts.algorithm, opts.cleanup, opts.cleanup_strength, opts.grid_scale],
    [16, false, 0, 0, 1, 1],
  );
  opts.free();
}

// 2. Default options give exactly what the original export gives.
const legacy = pkg.process_image(fixture, 16);
assert.deepEqual([...run(() => {})], [...legacy], 'defaults differ from process_image');
assert.deepEqual(pngSize(legacy), [27, 28]);

// 3. No-resize keeps the size, with and without Blocks, at several grid scales.
for (const [cleanup, gridScale] of [[0, 1], [1, 1], [1, 1.5], [1, 2], [1, 3]]) {
  const out = run((o) => {
    o.no_resize = true;
    o.cleanup = cleanup;
    o.grid_scale = gridScale;
  });
  assert.deepEqual(pngSize(out), [128, 128], `cleanup=${cleanup} grid_scale=${gridScale}`);
}

// 4. Blocks actually changes the picture.
const plain = run((o) => { o.no_resize = true; });
const blocks = run((o) => { o.no_resize = true; o.cleanup = 1; });
assert.notDeepEqual([...blocks], [...plain], 'Blocks returned the plain reduction');

// 4b. So does Smooth, differently at each strength, and it keeps the size.
const smooth = [1, 2, 3].map((strength) =>
  run((o) => { o.no_resize = true; o.cleanup = 2; o.cleanup_strength = strength; }),
);
for (const out of smooth) {
  assert.deepEqual(pngSize(out), [128, 128]);
  assert.notDeepEqual([...out], [...plain], 'Smooth returned the plain reduction');
  assert.notDeepEqual([...out], [...blocks], 'Smooth returned the Blocks result');
}
assert.notDeepEqual([...smooth[0]], [...smooth[2]], 'Light and Strong gave the same result');

// 5. A finer grid in snap mode gives a larger output.
const [w2, h2] = pngSize(run((o) => { o.grid_scale = 2; }));
assert.ok(w2 > 27 && h2 > 28, `grid_scale=2 gave ${w2}x${h2}`);

// 6. Invalid options are errors that name the problem. That includes values that a
//    plain u32 conversion would quietly turn into something valid: NaN and 2^32 into 0,
//    1.5 into 1, 2^32 + 16 into 16.
const invalid = [
  [(o) => { o.k_colors = 0; }, /k_colors/],
  [(o) => { o.no_resize = true; o.algorithm = 9; }, /algorithm/i],
  [(o) => { o.no_resize = true; o.cleanup = 7; }, /cleanup/i],
  [(o) => { o.cleanup = 1; }, /no_resize/],
  [(o) => { o.cleanup = 2; }, /no_resize/],
  [(o) => { o.no_resize = true; o.cleanup = 2; o.cleanup_strength = 0; }, /strength/i],
  [(o) => { o.no_resize = true; o.cleanup = 2; o.cleanup_strength = 4; }, /strength/i],
  [(o) => { o.no_resize = true; o.cleanup = 2; o.cleanup_strength = 1.5; }, /strength/i],
  [(o) => { o.grid_scale = 0.5; }, /grid scale/i],
  [(o) => { o.grid_scale = Number.NaN; }, /grid scale/i],
  [(o) => { o.grid_scale = Infinity; }, /grid scale/i],
];
for (const bad of [Number.NaN, Infinity, -1, 1.5, 2 ** 32, 2 ** 32 + 16]) {
  invalid.push([(o) => { o.k_colors = bad; }, /k_colors/]);
  invalid.push([(o) => { o.no_resize = true; o.algorithm = bad; }, /algorithm/]);
  invalid.push([(o) => { o.no_resize = true; o.cleanup = bad; }, /cleanup/]);
}
for (const [configure, pattern] of invalid) {
  assert.throws(() => run(configure), (err) => pattern.test(String(err)), String(pattern));
}

// 7. A plain object is not an options instance.
assert.throws(() => pkg.process_image_with(fixture, { k_colors: 16 }));

// 8. Bytes that are not an image are an error, not a trap.
{
  const opts = new pkg.ProcessOptions();
  assert.throws(() => pkg.process_image_with(new Uint8Array([1, 2, 3]), opts), /image/i);
  opts.free();
}

console.log('WASM smoke test passed.');

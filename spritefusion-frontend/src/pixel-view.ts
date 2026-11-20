/**
 * Showing pixel art without the browser resampling it.
 *
 * A browser scaling an image by a non-integer factor either blurs it (bilinear) or,
 * with `image-rendering: pixelated`, draws some pixels wider than others. Both hide
 * what the image really holds. Everything here is drawn at a whole number of *device*
 * pixels per image pixel (`zoom`), never less than one; what does not fit scrolls.
 */

/** Largest zoom offered, in device pixels per image pixel. */
export const MAX_ZOOM = 64;

/**
 * The largest whole zoom at which a `width` x `height` image fits in `availWidth` x
 * `availHeight` CSS pixels, at least 1 (larger images then scroll) and at most
 * `MAX_ZOOM`.
 */
export function fitZoom(
  width: number,
  height: number,
  availWidth: number,
  availHeight: number,
  dpr: number,
): number {
  if (width <= 0 || height <= 0) return 1;
  const z = Math.floor(Math.min((availWidth * dpr) / width, (availHeight * dpr) / height));
  return Math.max(1, Math.min(MAX_ZOOM, z));
}

/** CSS size of a `width` x `height` image drawn at `zoom` device pixels per pixel. */
export function cssSize(
  width: number,
  height: number,
  zoom: number,
  dpr: number,
): { width: number; height: number } {
  return { width: (width * zoom) / dpr, height: (height * zoom) / dpr };
}

/**
 * The zoom for a thumbnail in a `box` x `box` CSS pixel square: a whole number of
 * device pixels per pixel when the image fits at one or more, or `null` when it is
 * larger than the box even at one device pixel per pixel and has to be scaled down.
 */
export function thumbZoom(width: number, height: number, box: number, dpr: number): number | null {
  const z = Math.floor(Math.min((box * dpr) / width, (box * dpr) / height));
  return z >= 1 ? Math.min(z, MAX_ZOOM) : null;
}

/**
 * Paint a snapped result back over the source it came from: output pixel (i, j) fills
 * the source rectangle between cuts i and i + 1 across and j and j + 1 down.
 *
 * Cells are not all the same size, so stretching the small image to the source's size
 * would put its pixels in the wrong places (and make the browser resample it). This
 * puts each one exactly over the pixels it was computed from, so the result and the
 * original share one pixel grid and can be compared at any whole zoom.
 *
 * `pixels` is RGBA, `outWidth * outHeight * 4` bytes. Returns `null` when the cuts do
 * not describe this result (not one cell per pixel, or not ending at the source size).
 */
export function expandCells(
  pixels: Uint8ClampedArray,
  outWidth: number,
  outHeight: number,
  colCuts: Uint32Array,
  rowCuts: Uint32Array,
  sourceWidth: number,
  sourceHeight: number,
): Uint8ClampedArray | null {
  if (
    colCuts.length !== outWidth + 1 ||
    rowCuts.length !== outHeight + 1 ||
    colCuts[0] !== 0 ||
    rowCuts[0] !== 0 ||
    colCuts[outWidth] !== sourceWidth ||
    rowCuts[outHeight] !== sourceHeight
  ) {
    return null;
  }
  const out = new Uint8ClampedArray(sourceWidth * sourceHeight * 4);
  const words = new Uint32Array(out.buffer);
  // Whole pixels as 32-bit words (byte order does not matter: they are only copied).
  const aligned = pixels.byteOffset % 4 === 0 ? pixels : pixels.slice();
  const src = new Uint32Array(aligned.buffer, aligned.byteOffset, outWidth * outHeight);
  // One output row of the source width, then copied down the cell's height.
  for (let j = 0; j < outHeight; j++) {
    const y0 = rowCuts[j];
    const y1 = rowCuts[j + 1];
    const rowStart = y0 * sourceWidth;
    for (let i = 0; i < outWidth; i++) {
      words.fill(src[j * outWidth + i], rowStart + colCuts[i], rowStart + colCuts[i + 1]);
    }
    for (let y = y0 + 1; y < y1; y++) {
      words.copyWithin(y * sourceWidth, rowStart, rowStart + sourceWidth);
    }
  }
  return out;
}

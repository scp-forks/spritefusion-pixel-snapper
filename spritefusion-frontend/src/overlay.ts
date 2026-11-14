/** A value per grid cell to tint a result with. */
export interface OverlayMap {
  /** Pixel size of the result being tinted. */
  width: number;
  height: number;
  /** The grid's cuts in source pixels, from 0 to the source width and height. */
  colCuts: Uint32Array;
  rowCuts: Uint32Array;
  /** One byte per cell, row-major; 0 leaves the cell untinted. */
  values: Uint8Array;
}

export const OVERLAY_COLOR = [255, 0, 200] as const;
/** Alpha of a cell at the top of the scale, as a fraction of opaque. */
export const OVERLAY_STRENGTH = 0.8;

/**
 * RGBA pixels (`width * height * 4` bytes) tinting each cell by its value.
 *
 * Each cell is painted where the engine painted it. A snapped result has one pixel
 * per cell, whatever the cuts say (they are in source pixels). A full-size result has
 * the size of the source, and each cell covers its own rectangle between the cuts;
 * cells are not all the same size, so stretching a small map over the image would not
 * line up.
 */
export function rasterizeOverlay(map: OverlayMap): Uint8ClampedArray {
  const { width, height, colCuts, rowCuts, values } = map;
  const cols = colCuts.length - 1;
  const rows = rowCuts.length - 1;
  const data = new Uint8ClampedArray(width * height * 4);
  if (cols <= 0 || rows <= 0) return data;
  // Snapped: one pixel per cell on both axes. (A full-size result can never match on
  // both: its cells are at least 2 px.)
  const perCell = width === cols && height === rows;
  for (let cy = 0; cy < rows; cy++) {
    const y0 = perCell ? cy : rowCuts[cy];
    const y1 = Math.min(perCell ? cy + 1 : rowCuts[cy + 1], height);
    for (let cx = 0; cx < cols; cx++) {
      const v = values[cy * cols + cx];
      if (!v) continue;
      const x0 = perCell ? cx : colCuts[cx];
      const x1 = Math.min(perCell ? cx + 1 : colCuts[cx + 1], width);
      const alpha = Math.round(v * OVERLAY_STRENGTH);
      for (let y = y0; y < y1; y++) {
        for (let i = (y * width + x0) * 4; i < (y * width + x1) * 4; i += 4) {
          data[i] = OVERLAY_COLOR[0];
          data[i + 1] = OVERLAY_COLOR[1];
          data[i + 2] = OVERLAY_COLOR[2];
          data[i + 3] = alpha;
        }
      }
    }
  }
  return data;
}

import init, {
  ProcessOptions,
  process_candidates,
} from '../../pkg/spritefusion_pixel_snapper.js';
import { effectiveSettings, type ProcessSettings } from './settings';

let initialized = false;

export async function initWasm(): Promise<void> {
  if (initialized) return;
  await init();
  initialized = true;
}

/** One result of the engine, copied out of WASM memory. */
export interface EngineCandidate {
  png: Uint8Array;
  width: number;
  height: number;
  uniqueColors: number;
  /** Cells of the grid it was built on; 0 when the mode uses no grid. */
  cols: number;
  rows: number;
  /** How many times finer than the detected grid. 1 = the detected grid. */
  gridScale: number;
  /** Is there a pixel grid in the image; the same on every candidate of an image. */
  latticeScore: number;
  /** Exactly what the original `process_image` returns. */
  isLegacy: boolean;
}

/**
 * Run the engine. Snap mode and Blocks return several sizes, coarsest first; other
 * modes return their single result. With default settings the candidate flagged
 * `isLegacy` is exactly what the original `process_image` export returns (the Rust
 * test suite pins that).
 */
export async function processCandidates(
  bytes: Uint8Array,
  settings: ProcessSettings,
): Promise<EngineCandidate[]> {
  await initWasm();
  const s = effectiveSettings(settings);
  const opts = new ProcessOptions();
  try {
    opts.k_colors = s.kColors;
    opts.no_resize = s.noResize;
    opts.algorithm = s.algorithm;
    opts.cleanup = s.cleanup;
    opts.cleanup_strength = s.cleanupStrength;
    opts.grid_fitter = s.gridFitter;
    opts.target_width = s.targetWidth;
    return process_candidates(bytes, opts).map((c) => {
      // Every getter copies its value out of WASM memory: read each once, then
      // release the object. wasm-bindgen objects are not garbage collected.
      try {
        return {
          png: c.png,
          width: c.width,
          height: c.height,
          uniqueColors: c.unique_colors,
          cols: c.cols,
          rows: c.rows,
          gridScale: c.grid_scale,
          latticeScore: c.lattice_score,
          isLegacy: c.is_legacy,
        };
      } finally {
        c.free();
      }
    });
  } finally {
    opts.free();
  }
}

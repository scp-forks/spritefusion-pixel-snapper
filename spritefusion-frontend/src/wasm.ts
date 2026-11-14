import init, {
  ProcessOptions,
  process_image_with,
} from '../../pkg/spritefusion_pixel_snapper.js';
import { effectiveSettings, type ProcessSettings } from './settings';

let initialized = false;

export async function initWasm(): Promise<void> {
  if (initialized) return;
  await init();
  initialized = true;
}

/**
 * Run the engine. With default settings this returns exactly what the original
 * `process_image` export returns (the Rust test suite pins that).
 */
export async function processImage(
  bytes: Uint8Array,
  settings: ProcessSettings,
): Promise<Uint8Array> {
  await initWasm();
  const s = effectiveSettings(settings);
  const opts = new ProcessOptions();
  try {
    opts.k_colors = s.kColors;
    opts.no_resize = s.noResize;
    opts.algorithm = s.algorithm;
    opts.cleanup = s.cleanup;
    opts.cleanup_strength = s.cleanupStrength;
    opts.grid_scale = s.gridScale;
    opts.grid_fitter = s.gridFitter;
    return process_image_with(bytes, opts);
  } finally {
    // wasm-bindgen objects live in WASM memory and are not garbage collected.
    opts.free();
  }
}

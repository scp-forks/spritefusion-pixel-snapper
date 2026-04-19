import init, {
  process_image,
  process_image_advanced,
} from '../../pkg/spritefusion_pixel_snapper.js';

let initialized = false;

export async function initWasm(): Promise<void> {
  if (initialized) return;
  await init();
  initialized = true;
}

export async function processImage(
  bytes: Uint8Array,
  kColors: number,
  noResize: boolean = false,
  algorithm: number = 0,
): Promise<Uint8Array> {
  await initWasm();
  if (noResize) {
    return process_image_advanced(bytes, kColors, true, algorithm);
  }
  return process_image(bytes, kColors);
}

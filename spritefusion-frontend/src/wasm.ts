import init, { process_image } from '../../pkg/spritefusion_pixel_snapper.js';

let initialized = false;

export async function initWasm(): Promise<void> {
  if (initialized) return;
  await init();
  initialized = true;
}

export async function processImage(
  bytes: Uint8Array,
  kColors: number,
): Promise<Uint8Array> {
  await initWasm();
  return process_image(bytes, kColors);
}

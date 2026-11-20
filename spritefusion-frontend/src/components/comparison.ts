import { rasterizeOverlay, type OverlayMap } from '../overlay';
import { MAX_ZOOM, cssSize, expandCells, fitZoom } from '../pixel-view';

/** A result to show against the original. */
export interface ComparedResult {
  /** The result as PNG bytes. */
  bytes: Uint8Array;
  width: number;
  height: number;
  /** The grid's cuts in source pixels; empty when the result has no grid. */
  colCuts: Uint32Array;
  rowCuts: Uint32Array;
}

/** Share of the window's height the view may take before it scrolls. */
const MAX_VIEW_HEIGHT = 0.75;

/** Decode PNG bytes to their exact pixels: no colour management, no premultiplying. */
async function decodePixels(bytes: Uint8Array): Promise<ImageData> {
  const bitmap = await createImageBitmap(new Blob([bytes as BlobPart], { type: 'image/png' }), {
    premultiplyAlpha: 'none',
    colorSpaceConversion: 'none',
  });
  const canvas = document.createElement('canvas');
  canvas.width = bitmap.width;
  canvas.height = bitmap.height;
  const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
  ctx.drawImage(bitmap, 0, 0);
  bitmap.close();
  return ctx.getImageData(0, 0, canvas.width, canvas.height);
}

/**
 * Before/after view of the original and a result.
 *
 * Both are drawn on the original's pixel grid at a whole number of device pixels per
 * image pixel, so the browser never resamples either of them; see `pixel-view.ts`.
 * A snapped (smaller) result is painted back over the source cells it was computed
 * from, so the two line up pixel for pixel. Each side of the divider shows only its
 * own image, over a checkerboard where it is transparent.
 */
export function createComparison(container: HTMLElement) {
  container.innerHTML = `
    <div class="comparison-wrapper">
      <div class="comparison-viewport">
        <div class="comparison-stage">
          <img class="comparison-original" alt="Original" draggable="false" />
          <canvas class="comparison-processed"></canvas>
          <canvas class="comparison-overlay hidden"></canvas>
        </div>
      </div>
      <div class="comparison-divider">
        <div class="comparison-handle"></div>
      </div>
      <span class="comparison-label comparison-label-left">Original</span>
      <span class="comparison-label comparison-label-right">Processed</span>
      <div class="comparison-zoom" title="Zoom: each image pixel is drawn as this many screen pixels across, always a whole number, so nothing is blurred or resampled. Scroll to pan when the image is larger than the view.">
        <button type="button" class="comparison-zoom-btn" data-zoom="out" aria-label="Zoom out">−</button>
        <span class="comparison-zoom-value">1×</span>
        <button type="button" class="comparison-zoom-btn" data-zoom="in" aria-label="Zoom in">+</button>
        <button type="button" class="comparison-zoom-btn comparison-zoom-fit" data-zoom="fit">Fit</button>
      </div>
    </div>
  `;

  const wrapper = container.querySelector<HTMLDivElement>('.comparison-wrapper')!;
  const viewport = container.querySelector<HTMLDivElement>('.comparison-viewport')!;
  const stage = container.querySelector<HTMLDivElement>('.comparison-stage')!;
  const originalImg = container.querySelector<HTMLImageElement>('.comparison-original')!;
  const processed = container.querySelector<HTMLCanvasElement>('.comparison-processed')!;
  const overlay = container.querySelector<HTMLCanvasElement>('.comparison-overlay')!;
  const divider = container.querySelector<HTMLDivElement>('.comparison-divider')!;
  const zoomBox = container.querySelector<HTMLDivElement>('.comparison-zoom')!;
  const zoomValue = container.querySelector<HTMLSpanElement>('.comparison-zoom-value')!;
  const zoomOut = zoomBox.querySelector<HTMLButtonElement>('[data-zoom="out"]')!;
  const zoomIn = zoomBox.querySelector<HTMLButtonElement>('[data-zoom="in"]')!;
  const zoomFit = zoomBox.querySelector<HTMLButtonElement>('[data-zoom="fit"]')!;

  /**
   * Size of the original as the engine decoded it: the pixel grid everything is drawn
   * on. Taken from the results, not from the browser, which applies a JPEG's EXIF
   * orientation where the engine does not (the original is shown unrotated to match).
   */
  let srcWidth = 0;
  let srcHeight = 0;
  /** Device pixels per image pixel. */
  let zoom = 1;
  /** Follow the window: recompute the fitting zoom on resize until the user zooms. */
  let fitting = true;
  /** Divider position, as a share of the visible width. */
  let pct = 50;
  /** Bumped per result, so a slow decode cannot paint over a newer result. */
  let processedSeq = 0;
  /** The lost-detail map on show, if any; drawn once the source size is known. */
  let overlayMap: Omit<OverlayMap, 'width' | 'height'> | null = null;

  function drawOverlay() {
    overlay.classList.toggle('hidden', !overlayMap || !srcWidth);
    if (!overlayMap || !srcWidth) return;
    overlay.width = srcWidth;
    overlay.height = srcHeight;
    const image = new ImageData(srcWidth, srcHeight);
    image.data.set(rasterizeOverlay({ ...overlayMap, width: srcWidth, height: srcHeight }));
    overlay.getContext('2d')!.putImageData(image, 0, 0);
  }

  function dpr(): number {
    return window.devicePixelRatio || 1;
  }

  function fit(): number {
    const availWidth = container.clientWidth || wrapper.clientWidth;
    const availHeight = window.innerHeight * MAX_VIEW_HEIGHT;
    return fitZoom(srcWidth, srcHeight, availWidth, availHeight, dpr());
  }

  /** Lay out the stage at the current zoom, keeping the image point at the view's centre. */
  function applyZoom(next: number) {
    if (!srcWidth || !srcHeight) return;
    const before = stage.getBoundingClientRect();
    // The image point at the centre of the view, as a share of the image.
    const cx = before.width ? (viewport.scrollLeft + viewport.clientWidth / 2) / before.width : 0.5;
    const cy = before.height ? (viewport.scrollTop + viewport.clientHeight / 2) / before.height : 0.5;

    zoom = Math.max(1, Math.min(MAX_ZOOM, next));
    const size = cssSize(srcWidth, srcHeight, zoom, dpr());
    stage.style.width = `${size.width}px`;
    stage.style.height = `${size.height}px`;
    viewport.style.maxHeight = `${Math.round(window.innerHeight * MAX_VIEW_HEIGHT)}px`;

    // Scroll by whole device pixels too, for the same reason.
    const ratio = dpr();
    viewport.scrollLeft = Math.round((cx * size.width - viewport.clientWidth / 2) * ratio) / ratio;
    viewport.scrollTop = Math.round((cy * size.height - viewport.clientHeight / 2) * ratio) / ratio;
    alignToDevicePixels();

    zoomValue.textContent = `${zoom}×`;
    zoomOut.disabled = zoom <= 1;
    zoomIn.disabled = zoom >= MAX_ZOOM;
    zoomFit.disabled = fitting;
    updateDivider();
  }

  /**
   * Put the view's top-left corner on a device pixel. A whole zoom alone is not
   * enough: drawn from a fractional device position (a centred layout, a fractional
   * DPR), nearest-neighbour scaling makes some pixels a device pixel narrower.
   */
  function alignToDevicePixels() {
    const ratio = dpr();
    viewport.style.translate = '';
    const r = viewport.getBoundingClientRect();
    const dx = (Math.round(r.left * ratio) - r.left * ratio) / ratio;
    const dy = (Math.round(r.top * ratio) - r.top * ratio) / ratio;
    if (dx || dy) viewport.style.translate = `${dx}px ${dy}px`;
  }

  /** Place the divider and clip each layer to its side of it. */
  function updateDivider() {
    const visible = viewport.clientWidth;
    const x = (pct / 100) * visible;
    divider.style.left = `${x}px`;
    // The same line in stage coordinates.
    const boundary = viewport.scrollLeft + x;
    const stageWidth = stage.getBoundingClientRect().width;
    originalImg.style.clipPath = `inset(0 ${Math.max(0, stageWidth - boundary)}px 0 0)`;
    const right = `inset(0 0 0 ${Math.max(0, boundary)}px)`;
    processed.style.clipPath = right;
    overlay.style.clipPath = right;
  }

  zoomBox.addEventListener('pointerdown', (e) => e.stopPropagation());
  zoomBox.addEventListener('click', (e) => {
    const btn = (e.target as HTMLElement).closest<HTMLButtonElement>('[data-zoom]');
    if (!btn) return;
    const action = btn.dataset.zoom;
    if (action === 'fit') {
      fitting = true;
      applyZoom(fit());
    } else {
      fitting = false;
      applyZoom(zoom + (action === 'in' ? 1 : -1));
    }
  });

  function pointerPct(clientX: number): number {
    const r = viewport.getBoundingClientRect();
    return Math.max(0, Math.min(100, ((clientX - r.left) / viewport.clientWidth) * 100));
  }

  wrapper.addEventListener('pointerdown', (e) => {
    if (e.button !== 0) return;
    // Leave the scrollbars to the browser.
    const r = viewport.getBoundingClientRect();
    if (e.clientX - r.left > viewport.clientWidth || e.clientY - r.top > viewport.clientHeight) {
      return;
    }
    e.preventDefault();
    wrapper.setPointerCapture(e.pointerId);
    divider.classList.add('comparison-divider-active');
    pct = pointerPct(e.clientX);
    updateDivider();

    const onMove = (ev: PointerEvent) => {
      pct = pointerPct(ev.clientX);
      updateDivider();
    };
    const onUp = () => {
      divider.classList.remove('comparison-divider-active');
      wrapper.removeEventListener('pointermove', onMove);
      wrapper.removeEventListener('pointerup', onUp);
      wrapper.removeEventListener('pointercancel', onUp);
    };
    wrapper.addEventListener('pointermove', onMove);
    wrapper.addEventListener('pointerup', onUp);
    wrapper.addEventListener('pointercancel', onUp);
  });

  viewport.addEventListener('scroll', updateDivider);
  // Page scrolling can leave the view at a fractional device position.
  window.addEventListener('scroll', () => alignToDevicePixels(), { passive: true });

  // Window resizes and browser zoom (which changes the device pixel ratio).
  window.addEventListener('resize', () => applyZoom(fitting ? fit() : zoom));
  new ResizeObserver(() => {
    alignToDevicePixels();
    updateDivider();
  }).observe(container);

  return {
    /** A new original. Laid out, at the fitting zoom, when its first result arrives. */
    setOriginal(url: string) {
      originalImg.src = url;
      // Clear the previous image's result until the new one arrives.
      processedSeq++;
      srcWidth = 0;
      srcHeight = 0;
      processed.width = 0;
      processed.height = 0;
      overlayMap = null;
      overlay.classList.add('hidden');
      fitting = true;
      pct = 50;
      viewport.scrollLeft = 0;
      viewport.scrollTop = 0;
    },

    /** Show a result. Snapped results are painted over the source cells they came from. */
    async setProcessed(result: ComparedResult) {
      const seq = ++processedSeq;
      const pixels = await decodePixels(result.bytes);
      if (seq !== processedSeq) return;

      // The source size: a snapped result's cuts end at it; any other result has it.
      const hasCuts = result.colCuts.length > 1 && result.rowCuts.length > 1;
      const width = hasCuts ? result.colCuts[result.colCuts.length - 1] : result.width;
      const height = hasCuts ? result.rowCuts[result.rowCuts.length - 1] : result.height;
      if (width !== srcWidth || height !== srcHeight) {
        srcWidth = width;
        srcHeight = height;
        applyZoom(fitting ? fit() : zoom);
        drawOverlay();
      }

      let data: Uint8ClampedArray = pixels.data;
      if (result.width !== srcWidth || result.height !== srcHeight) {
        const expanded = expandCells(
          pixels.data,
          result.width,
          result.height,
          result.colCuts,
          result.rowCuts,
          srcWidth,
          srcHeight,
        );
        if (!expanded) {
          throw new Error(
            `Cannot line up a ${result.width}x${result.height} result with a ${srcWidth}x${srcHeight} original`,
          );
        }
        data = expanded;
      }
      processed.width = srcWidth;
      processed.height = srcHeight;
      const image = new ImageData(srcWidth, srcHeight);
      image.data.set(data);
      processed.getContext('2d')!.putImageData(image, 0, 0);
    },

    /**
     * Tint the result by a per-cell map, or clear the tint with `null`. The map is
     * painted on the original's pixel grid, like the result it covers.
     */
    setOverlay(map: Omit<OverlayMap, 'width' | 'height'> | null) {
      overlayMap = map;
      drawOverlay();
    },
  };
}

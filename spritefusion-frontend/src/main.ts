import './style.css';
import { initWasm, processImage } from './wasm';
import {
  bytesToObjectUrl,
  countUniqueColors,
  fileToUint8Array,
  getImageDimensions,
  getOutputFilename,
  revokeObjectUrl,
} from './utils/image-helpers';
import { createResultCache, type CachedResult } from './utils/result-cache';
import {
  DEFAULT_SETTINGS,
  effectiveSettings,
  settingsKey,
  type ProcessSettings,
} from './settings';
import { createDropzone, validateFile } from './components/dropzone';
import { createControls } from './components/controls';
import { createComparison } from './components/comparison';
import { createStats } from './components/stats';
import { createToolbar } from './components/toolbar';

// --- Toast ---
const toastEl = document.getElementById('toast')!;
let toastTimer: ReturnType<typeof setTimeout> | null = null;

export function showToast(message: string) {
  toastEl.textContent = message;
  toastEl.classList.remove('hidden');
  if (toastTimer) clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toastEl.classList.add('hidden'), 2000);
}

// --- App State ---
interface AppState {
  originalFile: File | null;
  originalBytes: Uint8Array | null;
  originalUrl: string | null;
  originalWidth: number;
  originalHeight: number;
  originalUniqueColors: number;
  current: CachedResult | null;
  /** What the controls currently say; hidden controls keep their value. */
  settings: ProcessSettings;
  isProcessing: boolean;
}

const state: AppState = {
  originalFile: null,
  originalBytes: null,
  originalUrl: null,
  originalWidth: 0,
  originalHeight: 0,
  originalUniqueColors: 0,
  current: null,
  settings: { ...DEFAULT_SETTINGS },
  isProcessing: false,
};

const resultCache = createResultCache();

// Bumped on every new source image so in-flight work from the previous one can
// be discarded instead of being cached against the wrong image.
let fileSeq = 0;
// Bumped on every process request so a slow result that has been superseded is
// still cached, but not displayed.
let requestSeq = 0;
// Set when a debounced change lands mid-run; replayed once the run finishes.
let pendingRun = false;

// --- DOM refs ---
const controlsEl = document.getElementById('controls')!;
const comparisonEl = document.getElementById('comparison')!;
const statsEl = document.getElementById('stats')!;
const toolbarEl = document.getElementById('toolbar')!;

// --- Components ---
const dropzone = createDropzone(document.getElementById('dropzone')!, {
  onFile: handleFile,
});

const controls = createControls(controlsEl, {
  onProcess: runProcess,
  onKColorsChange: (k) => {
    state.settings.kColors = k;
  },
  onNoResizeChange: (noResize) => {
    state.settings.noResize = noResize;
  },
  onAlgorithmChange: (algorithm) => {
    state.settings.algorithm = algorithm;
  },
  onCleanupChange: (cleanup) => {
    state.settings.cleanup = cleanup;
  },
  onCleanupStrengthChange: (strength) => {
    state.settings.cleanupStrength = strength;
  },
  onGridScaleChange: (gridScale) => {
    state.settings.gridScale = gridScale;
  },
});

const comparison = createComparison(comparisonEl);
const stats = createStats(statsEl);

createToolbar(toolbarEl, {
  getProcessedBytes: () => state.current?.bytes ?? null,
  getFilename: () => getOutputFilename(state.originalFile?.name ?? 'image.png'),
});

// --- Handlers ---
async function handleFile(file: File) {
  const token = ++fileSeq;

  const bytes = await fileToUint8Array(file);
  if (token !== fileSeq) return;

  state.originalFile = file;
  state.originalBytes = bytes;

  // Every cached result describes the previous image, so none of them survive.
  resultCache.clear();
  state.current = null;

  revokeObjectUrl(state.originalUrl);
  state.originalUrl = bytesToObjectUrl(state.originalBytes);

  const origDim = await getImageDimensions(state.originalUrl);
  if (token !== fileSeq) return;
  state.originalWidth = origDim.width;
  state.originalHeight = origDim.height;

  // Count unique colors in background (non-blocking)
  state.originalUniqueColors = 0;
  countUniqueColors(state.originalUrl).then((count) => {
    if (token !== fileSeq) return;
    state.originalUniqueColors = count;
    if (state.current) renderStats();
  });

  dropzone.collapse();
  controlsEl.classList.remove('hidden');
  comparisonEl.classList.add('hidden');
  statsEl.classList.add('hidden');
  toolbarEl.classList.add('hidden');

  runProcess();
}

function renderStats() {
  const entry = state.current;
  if (!entry) return;
  stats.update({
    originalWidth: state.originalWidth,
    originalHeight: state.originalHeight,
    originalUniqueColors: state.originalUniqueColors,
    outputWidth: entry.width,
    outputHeight: entry.height,
    kColors: entry.settings.kColors,
    outputUniqueColors: entry.uniqueColors,
    noResize: entry.settings.noResize,
    algorithm: entry.settings.algorithm,
    cleanup: entry.settings.cleanup,
    cleanupStrength: entry.settings.cleanupStrength,
    gridScale: entry.settings.gridScale,
  });
}

/** Show a result. Synchronous, so a cache hit swaps the preview instantly. */
function applyResult(entry: CachedResult) {
  state.current = entry;
  resultCache.pin(entry.key);

  if (comparisonEl.classList.contains('hidden')) {
    comparisonEl.classList.remove('hidden');
    comparison.setImages(state.originalUrl!, entry.url);
  } else {
    comparison.updateProcessed(entry.url);
  }

  renderStats();
  statsEl.classList.remove('hidden');
  toolbarEl.classList.remove('hidden');
}

async function runProcess() {
  if (!state.originalBytes) return;

  // Snapshot the settings this run is for; state can move on while we await.
  const settings = effectiveSettings(state.settings);
  const key = settingsKey(settings);
  const seq = ++requestSeq;

  const cached = resultCache.get(key);
  if (cached) {
    applyResult(cached);
    return;
  }

  if (state.isProcessing) {
    pendingRun = true;
    return;
  }

  const token = fileSeq;
  state.isProcessing = true;
  controls.setProcessing(true);

  // rAF + setTimeout trick to let the spinner render before blocking WASM call
  await new Promise<void>((resolve) =>
    requestAnimationFrame(() => setTimeout(resolve, 0)),
  );

  try {
    const bytes = await processImage(state.originalBytes, settings);
    const url = bytesToObjectUrl(bytes);

    const [outDim, outputUniqueColors] = await Promise.all([
      getImageDimensions(url),
      countUniqueColors(url),
    ]);

    // A new image landed mid-run: this result belongs to the old one.
    if (token !== fileSeq) {
      revokeObjectUrl(url);
      return;
    }

    const entry: CachedResult = {
      key,
      bytes,
      url,
      width: outDim.width,
      height: outDim.height,
      uniqueColors: outputUniqueColors,
      settings,
    };
    resultCache.set(entry);

    // Only paint if nothing newer has been requested in the meantime.
    if (seq === requestSeq) applyResult(entry);
  } catch (err) {
    showToast(`Processing failed: ${err}`);
  } finally {
    state.isProcessing = false;
    controls.setProcessing(false);
    if (pendingRun) {
      pendingRun = false;
      runProcess();
    }
  }
}

// --- Global drag-and-drop with overlay ---
const dropOverlay = document.getElementById('drop-overlay')!;
let dragCounter = 0;

document.addEventListener('dragenter', (e) => {
  e.preventDefault();
  dragCounter++;
  if (dragCounter === 1) dropOverlay.classList.remove('hidden');
});

document.addEventListener('dragleave', () => {
  dragCounter--;
  if (dragCounter === 0) dropOverlay.classList.add('hidden');
});

document.addEventListener('dragover', (e) => e.preventDefault());

document.addEventListener('drop', (e) => {
  e.preventDefault();
  dragCounter = 0;
  dropOverlay.classList.add('hidden');
  const file = e.dataTransfer?.files[0];
  if (!file) return;
  const err = validateFile(file);
  if (err) {
    showToast(err);
    return;
  }
  handleFile(file);
});

// --- Paste handler (Ctrl-V / Cmd-V) ---
document.addEventListener('paste', (e) => {
  const items = e.clipboardData?.items;
  if (!items) return;
  for (const item of items) {
    if (item.type.startsWith('image/')) {
      e.preventDefault();
      const file = item.getAsFile();
      if (!file) return;
      const err = validateFile(file);
      if (err) {
        showToast(err);
        return;
      }
      handleFile(file);
      return;
    }
  }
});

// --- Init WASM eagerly ---
initWasm().catch((err) => {
  showToast(`Failed to load WASM: ${err}`);
  console.error('WASM init failed:', err);
});

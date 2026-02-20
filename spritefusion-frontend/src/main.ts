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
  originalUniqueColors: number;
  processedBytes: Uint8Array | null;
  processedUrl: string | null;
  kColors: number;
  isProcessing: boolean;
}

const state: AppState = {
  originalFile: null,
  originalBytes: null,
  originalUrl: null,
  originalUniqueColors: 0,
  processedBytes: null,
  processedUrl: null,
  kColors: 16,
  isProcessing: false,
};

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
    state.kColors = k;
  },
});

const comparison = createComparison(comparisonEl);
const stats = createStats(statsEl);

createToolbar(toolbarEl, {
  getProcessedBytes: () => state.processedBytes,
  getFilename: () => getOutputFilename(state.originalFile?.name ?? 'image.png'),
});

// --- Handlers ---
async function handleFile(file: File) {
  state.originalFile = file;
  state.originalBytes = await fileToUint8Array(file);

  revokeObjectUrl(state.originalUrl);
  state.originalUrl = bytesToObjectUrl(state.originalBytes);

  // Count unique colors in background (non-blocking)
  countUniqueColors(state.originalUrl).then((count) => {
    state.originalUniqueColors = count;
  });

  // Reset processed state
  revokeObjectUrl(state.processedUrl);
  state.processedBytes = null;
  state.processedUrl = null;

  dropzone.collapse();
  controlsEl.classList.remove('hidden');
  comparisonEl.classList.add('hidden');
  statsEl.classList.add('hidden');
  toolbarEl.classList.add('hidden');

  runProcess();
}

async function runProcess() {
  if (!state.originalBytes || state.isProcessing) return;

  state.isProcessing = true;
  controls.setProcessing(true);

  // rAF + setTimeout trick to let the spinner render before blocking WASM call
  await new Promise<void>((resolve) =>
    requestAnimationFrame(() => setTimeout(resolve, 0)),
  );

  try {
    const result = await processImage(state.originalBytes, state.kColors);

    revokeObjectUrl(state.processedUrl);
    state.processedBytes = result;
    state.processedUrl = bytesToObjectUrl(result);

    if (comparisonEl.classList.contains('hidden')) {
      comparisonEl.classList.remove('hidden');
      comparison.setImages(state.originalUrl!, state.processedUrl);
    } else {
      comparison.updateProcessed(state.processedUrl);
    }

    // Compute and show stats
    const [origDim, outDim] = await Promise.all([
      getImageDimensions(state.originalUrl!),
      getImageDimensions(state.processedUrl),
    ]);
    stats.update({
      originalWidth: origDim.width,
      originalHeight: origDim.height,
      originalUniqueColors: state.originalUniqueColors,
      outputWidth: outDim.width,
      outputHeight: outDim.height,
      kColors: state.kColors,
    });
    statsEl.classList.remove('hidden');

    toolbarEl.classList.remove('hidden');
  } catch (err) {
    showToast(`Processing failed: ${err}`);
  } finally {
    state.isProcessing = false;
    controls.setProcessing(false);
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

// --- Init WASM eagerly ---
initWasm().catch((err) => {
  showToast(`Failed to load WASM: ${err}`);
  console.error('WASM init failed:', err);
});

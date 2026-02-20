import './style.css';
import { initWasm, processImage } from './wasm';
import {
  bytesToObjectUrl,
  fileToUint8Array,
  getOutputFilename,
  revokeObjectUrl,
} from './utils/image-helpers';
import { createDropzone, validateFile } from './components/dropzone';
import { createControls } from './components/controls';
import { createComparison } from './components/comparison';
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
  processedBytes: Uint8Array | null;
  processedUrl: string | null;
  kColors: number;
  isProcessing: boolean;
}

const state: AppState = {
  originalFile: null,
  originalBytes: null,
  originalUrl: null,
  processedBytes: null,
  processedUrl: null,
  kColors: 16,
  isProcessing: false,
};

// --- DOM refs ---
const controlsEl = document.getElementById('controls')!;
const comparisonEl = document.getElementById('comparison')!;
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

  // Reset processed state
  revokeObjectUrl(state.processedUrl);
  state.processedBytes = null;
  state.processedUrl = null;

  dropzone.collapse();
  controlsEl.classList.remove('hidden');
  comparisonEl.classList.add('hidden');
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

    toolbarEl.classList.remove('hidden');
  } catch (err) {
    showToast(`Processing failed: ${err}`);
  } finally {
    state.isProcessing = false;
    controls.setProcessing(false);
  }
}

// --- Global drop handler (drop anywhere on the page) ---
document.addEventListener('dragover', (e) => e.preventDefault());
document.addEventListener('drop', (e) => {
  e.preventDefault();
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

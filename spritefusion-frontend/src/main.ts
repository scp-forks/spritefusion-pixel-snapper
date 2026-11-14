import './style.css';
import { initWasm, processCandidates } from './wasm';
import {
  bytesToObjectUrl,
  countUniqueColors,
  fileToUint8Array,
  getImageDimensions,
  getOutputFilename,
  revokeObjectUrl,
} from './utils/image-helpers';
import {
  createResultCache,
  type CachedResult,
  type ResultSet,
} from './utils/result-cache';
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
import { createCandidates } from './components/candidates';
import { chooseSelection } from './selection';
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
  /** The set of results on screen, and which of them is in the comparison view. */
  current: ResultSet | null;
  selected: number;
  /** Tint the result where it is losing detail a finer size would keep. */
  showLostDetail: boolean;
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
  selected: 0,
  showLostDetail: false,
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
// Set when the user submits a width; the next set shown selects that size, whether
// it had to be computed or was already cached.
let selectTypedWidth = false;

// --- DOM refs ---
const controlsEl = document.getElementById('controls')!;
const comparisonEl = document.getElementById('comparison')!;
const statsEl = document.getElementById('stats')!;
const candidatesEl = document.getElementById('candidates')!;
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
  onGridFitterChange: (gridFitter) => {
    state.settings.gridFitter = gridFitter;
  },
});

const comparison = createComparison(comparisonEl);
const stats = createStats(statsEl);

const candidates = createCandidates(candidatesEl, {
  onSelect: (index) => {
    if (!state.current || index === state.selected) return;
    showResult(state.current, index);
  },
  onTargetWidth: (width) => {
    state.settings.targetWidth = width;
    selectTypedWidth = width > 0;
    runProcess();
  },
  onVote: (vote) => {
    state.settings.vote = vote;
    runProcess();
  },
  onOverlay: (on) => {
    state.showLostDetail = on;
    renderOverlay();
  },
});

/** Draw, or clear, the lost-detail tint for the result in the comparison view. */
function renderOverlay() {
  const result = selectedResult();
  const drawable = state.showLostDetail && result && result.cols > 0;
  comparison.setOverlay(
    drawable
      ? {
          width: result.width,
          height: result.height,
          colCuts: result.colCuts,
          rowCuts: result.rowCuts,
          values: result.lostDetail,
        }
      : null,
  );
}

function selectedResult(): CachedResult | null {
  return state.current?.results[state.selected] ?? null;
}

createToolbar(toolbarEl, {
  getProcessedBytes: () => selectedResult()?.bytes ?? null,
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

  // A typed width belongs to the image it was typed for.
  state.settings.targetWidth = 0;
  selectTypedWidth = false;

  dropzone.collapse();
  controlsEl.classList.remove('hidden');
  candidatesEl.classList.add('hidden');
  comparisonEl.classList.add('hidden');
  statsEl.classList.add('hidden');
  toolbarEl.classList.add('hidden');

  runProcess();
}

function renderStats() {
  const set = state.current;
  const result = selectedResult();
  if (!set || !result) return;
  stats.update({
    originalWidth: state.originalWidth,
    originalHeight: state.originalHeight,
    originalUniqueColors: state.originalUniqueColors,
    outputWidth: result.width,
    outputHeight: result.height,
    kColors: set.settings.kColors,
    outputUniqueColors: result.uniqueColors,
    noResize: set.settings.noResize,
    algorithm: set.settings.algorithm,
    cleanup: set.settings.cleanup,
    cleanupStrength: set.settings.cleanupStrength,
    gridCols: result.cols,
    gridRows: result.rows,
  });
}

/** Put one result of a set in the comparison view. Synchronous: switching is instant. */
function showResult(set: ResultSet, index: number) {
  const sameSet = state.current === set;
  state.current = set;
  state.selected = index;
  resultCache.pin(set.key);

  const result = set.results[index];
  if (comparisonEl.classList.contains('hidden')) {
    comparisonEl.classList.remove('hidden');
    comparison.setImages(state.originalUrl!, result.url);
  } else {
    comparison.updateProcessed(result.url);
  }

  if (sameSet) {
    candidates.select(index);
  } else {
    candidates.update(set, index);
  }
  renderOverlay();
  renderStats();
  statsEl.classList.remove('hidden');
  toolbarEl.classList.remove('hidden');
}

/** Show a set of results, choosing which of them to display. */
function applyResultSet(set: ResultSet) {
  const index = chooseSelection(set, selectedResult(), selectTypedWidth);
  selectTypedWidth = false;
  showResult(set, index);
}

async function runProcess() {
  if (!state.originalBytes) return;

  // Snapshot the settings this run is for; state can move on while we await.
  const settings = effectiveSettings(state.settings);
  const key = settingsKey(settings);
  const seq = ++requestSeq;

  const cached = resultCache.get(key);
  if (cached) {
    applyResultSet(cached);
    return;
  }

  if (state.isProcessing) {
    pendingRun = true;
    return;
  }

  const token = fileSeq;
  state.isProcessing = true;
  controls.setProcessing(true);
  candidates.setProcessing(true);

  // rAF + setTimeout trick to let the spinner render before blocking WASM call
  await new Promise<void>((resolve) =>
    requestAnimationFrame(() => setTimeout(resolve, 0)),
  );

  try {
    const produced = await processCandidates(state.originalBytes, settings);
    const results: CachedResult[] = produced.map((c) => ({
      bytes: c.png,
      url: bytesToObjectUrl(c.png),
      width: c.width,
      height: c.height,
      // Counted by the engine: decoding five megapixel images to count their
      // colours in JS cost more than producing them.
      uniqueColors: c.uniqueColors,
      cols: c.cols,
      rows: c.rows,
      gridScale: c.gridScale,
      isLegacy: c.isLegacy,
      colCuts: c.colCuts,
      rowCuts: c.rowCuts,
      lostDetail: c.lostDetail,
    }));

    // A new image landed mid-run: these results belong to the old one.
    if (token !== fileSeq) {
      for (const r of results) revokeObjectUrl(r.url);
      return;
    }

    const set: ResultSet = {
      key,
      settings,
      results,
      latticeScore: produced[0]?.latticeScore ?? 0,
    };
    // Only paint if nothing newer has been requested in the meantime. A set that is
    // about to be painted is pinned as it is stored, so that making room for it
    // cannot evict it.
    const willShow = seq === requestSeq;
    resultCache.set(set, { pin: willShow });
    if (willShow) applyResultSet(set);
  } catch (err) {
    showToast(`Processing failed: ${err}`);
    // A typed width the engine refused must not stick and fail every later run. Go
    // back to the width of the set still on screen, so its sizes stay as they are.
    if (state.settings.targetWidth === settings.targetWidth) {
      state.settings.targetWidth = state.current?.settings.targetWidth ?? 0;
    }
    selectTypedWidth = false;
  } finally {
    state.isProcessing = false;
    controls.setProcessing(false);
    candidates.setProcessing(false);
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

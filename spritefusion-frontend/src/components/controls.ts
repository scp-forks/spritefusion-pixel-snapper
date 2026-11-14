import { CLEANUP_BLOCKS } from '../settings';

export interface ControlsCallbacks {
  onProcess: () => void;
  onKColorsChange: (k: number) => void;
  onNoResizeChange: (noResize: boolean) => void;
  onAlgorithmChange: (algorithm: number) => void;
  onCleanupChange: (cleanup: number) => void;
  onGridScaleChange: (gridScale: number) => void;
}

export function createControls(
  container: HTMLElement,
  callbacks: ControlsCallbacks,
) {
  container.innerHTML = `
    <div class="controls-row">
      <label class="controls-label">
        Colors: <span class="controls-value">16</span>
      </label>
      <input type="range" class="controls-slider" min="2" max="64" value="16" />
    </div>
    <div class="controls-row-options">
      <label class="controls-checkbox-label">
        <input type="checkbox" class="controls-checkbox" />
        <span>No Resize</span>
      </label>
      <select class="controls-select controls-algorithm hidden" title="Palette algorithm">
        <option value="0">K-Means (Perceptual)</option>
        <option value="1">Median Cut</option>
        <option value="2">Wu's Optimal</option>
        <option value="3">Spatial (Pixel Art)</option>
        <option value="4">Grey Wolf Optimization</option>
        <option value="5">Differentiable Palette</option>
      </select>
      <select class="controls-select controls-cleanup hidden"
        title="Cleanup removes the speckle that palette reduction leaves. Blocks gives every grid cell one colour, at the original size.">
        <option value="0">Cleanup: Off</option>
        <option value="1">Cleanup: Blocks</option>
      </select>
      <select class="controls-select controls-grid-scale hidden"
        title="Blocks works on the detected pixel grid. Choose a finer grid if details are being lost.">
        <option value="1">Grid: Auto</option>
        <option value="1.5">Grid: 1.5x finer</option>
        <option value="2">Grid: 2x finer</option>
        <option value="3">Grid: 3x finer</option>
      </select>
    </div>
    <button class="controls-btn" type="button">
      <span class="btn-text">Process</span>
      <span class="btn-spinner hidden"></span>
    </button>
  `;

  const slider = container.querySelector<HTMLInputElement>('.controls-slider')!;
  const valueLabel = container.querySelector<HTMLSpanElement>('.controls-value')!;
  const checkbox = container.querySelector<HTMLInputElement>('.controls-checkbox')!;
  const algorithmSelect = container.querySelector<HTMLSelectElement>('.controls-algorithm')!;
  const cleanupSelect = container.querySelector<HTMLSelectElement>('.controls-cleanup')!;
  const gridScaleSelect = container.querySelector<HTMLSelectElement>('.controls-grid-scale')!;

  // Each control is shown only where it has an effect.
  function syncVisibility() {
    const noResize = checkbox.checked;
    const blocks = noResize && parseInt(cleanupSelect.value, 10) === CLEANUP_BLOCKS;
    algorithmSelect.classList.toggle('hidden', !noResize);
    cleanupSelect.classList.toggle('hidden', !noResize);
    gridScaleSelect.classList.toggle('hidden', !blocks);
  }
  const btn = container.querySelector<HTMLButtonElement>('.controls-btn')!;
  const btnText = btn.querySelector<HTMLSpanElement>('.btn-text')!;
  const btnSpinner = btn.querySelector<HTMLSpanElement>('.btn-spinner')!;

  // Only long enough to swallow keyboard auto-repeat (arrow-key stepping fires
  // one `change` per step). A committed value shouldn't feel laggy beyond that.
  const COMMIT_DELAY_MS = 150;

  let debounceTimer: ReturnType<typeof setTimeout> | null = null;

  function cancelScheduled() {
    if (debounceTimer) clearTimeout(debounceTimer);
    debounceTimer = null;
  }

  function scheduleProcess() {
    cancelScheduled();
    debounceTimer = setTimeout(() => {
      debounceTimer = null;
      callbacks.onProcess();
    }, COMMIT_DELAY_MS);
  }

  // `input` fires continuously while dragging. Update the readout only — the
  // expensive algorithms would otherwise run once per pixel of travel, and
  // since the WASM call blocks the main thread it would stutter the drag too.
  slider.addEventListener('input', () => {
    const k = parseInt(slider.value, 10);
    valueLabel.textContent = String(k);
    callbacks.onKColorsChange(k);
    cancelScheduled();
  });

  // `change` fires when the value is committed: on pointer release after a
  // drag, or per step when using the keyboard.
  slider.addEventListener('change', () => {
    callbacks.onKColorsChange(parseInt(slider.value, 10));
    scheduleProcess();
  });

  checkbox.addEventListener('change', () => {
    syncVisibility();
    callbacks.onNoResizeChange(checkbox.checked);
    scheduleProcess();
  });

  cleanupSelect.addEventListener('change', () => {
    syncVisibility();
    callbacks.onCleanupChange(parseInt(cleanupSelect.value, 10));
    scheduleProcess();
  });

  gridScaleSelect.addEventListener('change', () => {
    callbacks.onGridScaleChange(parseFloat(gridScaleSelect.value));
    scheduleProcess();
  });

  algorithmSelect.addEventListener('change', () => {
    callbacks.onAlgorithmChange(parseInt(algorithmSelect.value, 10));
    scheduleProcess();
  });

  btn.addEventListener('click', () => {
    cancelScheduled();
    callbacks.onProcess();
  });

  return {
    setProcessing(processing: boolean) {
      btn.disabled = processing;
      slider.disabled = processing;
      checkbox.disabled = processing;
      algorithmSelect.disabled = processing;
      cleanupSelect.disabled = processing;
      gridScaleSelect.disabled = processing;
      btnText.textContent = processing ? 'Processing...' : 'Process';
      btnSpinner.classList.toggle('hidden', !processing);
    },
    getKColors(): number {
      return parseInt(slider.value, 10);
    },
  };
}

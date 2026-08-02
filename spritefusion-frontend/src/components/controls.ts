export interface ControlsCallbacks {
  onProcess: () => void;
  onKColorsChange: (k: number) => void;
  onNoResizeChange: (noResize: boolean) => void;
  onAlgorithmChange: (algorithm: number) => void;
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
      <select class="controls-algorithm hidden">
        <option value="0">K-Means (Perceptual)</option>
        <option value="1">Median Cut</option>
        <option value="2">Wu's Optimal</option>
        <option value="3">Spatial (Pixel Art)</option>
        <option value="4">Grey Wolf Optimization</option>
        <option value="5">Differentiable Palette</option>
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
    const checked = checkbox.checked;
    algorithmSelect.classList.toggle('hidden', !checked);
    callbacks.onNoResizeChange(checked);
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
      btnText.textContent = processing ? 'Processing...' : 'Process';
      btnSpinner.classList.toggle('hidden', !processing);
    },
    getKColors(): number {
      return parseInt(slider.value, 10);
    },
  };
}

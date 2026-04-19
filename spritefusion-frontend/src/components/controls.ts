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

  let debounceTimer: ReturnType<typeof setTimeout> | null = null;

  slider.addEventListener('input', () => {
    const k = parseInt(slider.value, 10);
    valueLabel.textContent = String(k);
    callbacks.onKColorsChange(k);

    if (debounceTimer) clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
      callbacks.onProcess();
    }, 300);
  });

  checkbox.addEventListener('change', () => {
    const checked = checkbox.checked;
    algorithmSelect.classList.toggle('hidden', !checked);
    callbacks.onNoResizeChange(checked);

    if (debounceTimer) clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
      callbacks.onProcess();
    }, 300);
  });

  algorithmSelect.addEventListener('change', () => {
    callbacks.onAlgorithmChange(parseInt(algorithmSelect.value, 10));

    if (debounceTimer) clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
      callbacks.onProcess();
    }, 300);
  });

  btn.addEventListener('click', () => {
    if (debounceTimer) clearTimeout(debounceTimer);
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

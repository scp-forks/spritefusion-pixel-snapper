export interface ControlsCallbacks {
  onProcess: () => void;
  onKColorsChange: (k: number) => void;
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
    <button class="controls-btn" type="button">
      <span class="btn-text">Process</span>
      <span class="btn-spinner hidden"></span>
    </button>
  `;

  const slider = container.querySelector<HTMLInputElement>('.controls-slider')!;
  const valueLabel = container.querySelector<HTMLSpanElement>('.controls-value')!;
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

  btn.addEventListener('click', () => {
    if (debounceTimer) clearTimeout(debounceTimer);
    callbacks.onProcess();
  });

  return {
    setProcessing(processing: boolean) {
      btn.disabled = processing;
      slider.disabled = processing;
      btnText.textContent = processing ? 'Processing...' : 'Process';
      btnSpinner.classList.toggle('hidden', !processing);
    },
    getKColors(): number {
      return parseInt(slider.value, 10);
    },
  };
}

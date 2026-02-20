export function createComparison(container: HTMLElement) {
  container.innerHTML = `
    <div class="comparison-wrapper">
      <img class="comparison-original" alt="Original" />
      <div class="comparison-clip">
        <img class="comparison-processed" alt="Processed" />
      </div>
      <div class="comparison-divider">
        <div class="comparison-handle"></div>
      </div>
      <span class="comparison-label comparison-label-left">Original</span>
      <span class="comparison-label comparison-label-right">Processed</span>
      <button class="comparison-size-toggle" type="button" title="Toggle view size">Fit</button>
    </div>
  `;

  const wrapper = container.querySelector<HTMLDivElement>('.comparison-wrapper')!;
  const originalImg = container.querySelector<HTMLImageElement>('.comparison-original')!;
  const processedImg = container.querySelector<HTMLImageElement>('.comparison-processed')!;
  const clipDiv = container.querySelector<HTMLDivElement>('.comparison-clip')!;
  const divider = container.querySelector<HTMLDivElement>('.comparison-divider')!;
  const toggleBtn = container.querySelector<HTMLButtonElement>('.comparison-size-toggle')!;

  let pct = 50;
  let nativeMode = false;

  function updatePosition() {
    divider.style.left = `${pct}%`;
    clipDiv.style.width = `${100 - pct}%`;
    processedImg.style.width = `${wrapper.offsetWidth}px`;
  }

  function applyMode() {
    if (nativeMode) {
      // Size wrapper to the processed image's native dimensions
      const w = processedImg.naturalWidth;
      const h = processedImg.naturalHeight;
      if (w && h) {
        wrapper.style.maxWidth = `${w}px`;
        wrapper.style.aspectRatio = `${w} / ${h}`;
      }
      toggleBtn.textContent = '1:1';
    } else {
      wrapper.style.maxWidth = '';
      wrapper.style.aspectRatio = '';
      toggleBtn.textContent = 'Fit';
    }
    updatePosition();
  }

  toggleBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    nativeMode = !nativeMode;
    applyMode();
  });

  function onPointerDown(e: PointerEvent) {
    // Don't start drag if clicking the toggle button
    if ((e.target as HTMLElement).closest('.comparison-size-toggle')) return;

    e.preventDefault();
    wrapper.setPointerCapture(e.pointerId);
    divider.classList.add('comparison-divider-active');

    // Jump to click position immediately
    const rect = wrapper.getBoundingClientRect();
    pct = Math.max(0, Math.min(100, ((e.clientX - rect.left) / rect.width) * 100));
    updatePosition();

    const onMove = (ev: PointerEvent) => {
      const r = wrapper.getBoundingClientRect();
      pct = Math.max(0, Math.min(100, ((ev.clientX - r.left) / r.width) * 100));
      updatePosition();
    };

    const onUp = () => {
      divider.classList.remove('comparison-divider-active');
      wrapper.removeEventListener('pointermove', onMove);
      wrapper.removeEventListener('pointerup', onUp);
    };

    wrapper.addEventListener('pointermove', onMove);
    wrapper.addEventListener('pointerup', onUp);
  }

  wrapper.addEventListener('pointerdown', onPointerDown);

  const resizeObserver = new ResizeObserver(() => updatePosition());
  resizeObserver.observe(wrapper);

  return {
    setImages(originalUrl: string, processedUrl: string) {
      originalImg.src = originalUrl;
      processedImg.src = processedUrl;
      // Reset to fit mode on new image
      nativeMode = false;
      applyMode();
      originalImg.onload = () => updatePosition();
    },
    updateProcessed(processedUrl: string) {
      processedImg.src = processedUrl;
      processedImg.onload = () => applyMode();
    },
  };
}

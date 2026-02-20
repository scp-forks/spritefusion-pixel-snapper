export interface StatsData {
  originalWidth: number;
  originalHeight: number;
  originalUniqueColors: number;
  outputWidth: number;
  outputHeight: number;
  kColors: number;
}

function formatNumber(n: number): string {
  return n.toLocaleString();
}

export function createStats(container: HTMLElement) {
  container.innerHTML = `<div class="stats-row"></div>`;
  const row = container.querySelector<HTMLDivElement>('.stats-row')!;

  return {
    update(data: StatsData) {
      const cellW = data.originalWidth / data.outputWidth;
      const cellH = data.originalHeight / data.outputHeight;
      const cellSize =
        Math.abs(cellW - cellH) < 0.3
          ? `~${cellW.toFixed(1)}px`
          : `~${cellW.toFixed(1)} x ${cellH.toFixed(1)}px`;

      row.innerHTML = `
        <div class="stat-group">
          <span class="stat-group-label">Original</span>
          <div class="stat">
            <span class="stat-value">${data.originalWidth} x ${data.originalHeight}</span>
            <span class="stat-label">Resolution</span>
          </div>
          <div class="stat">
            <span class="stat-value">${formatNumber(data.originalUniqueColors)}</span>
            <span class="stat-label">Unique colors</span>
          </div>
        </div>
        <div class="stat-divider"></div>
        <div class="stat-group">
          <span class="stat-group-label">Snapped</span>
          <div class="stat">
            <span class="stat-value">${data.outputWidth} x ${data.outputHeight}</span>
            <span class="stat-label">Resolution</span>
          </div>
          <div class="stat">
            <span class="stat-value">${data.kColors}</span>
            <span class="stat-label">Palette colors</span>
          </div>
        </div>
        <div class="stat-divider"></div>
        <div class="stat-group">
          <span class="stat-group-label">Grid</span>
          <div class="stat">
            <span class="stat-value">${cellSize}</span>
            <span class="stat-label">Detected cell size</span>
          </div>
        </div>
      `;
    },
  };
}

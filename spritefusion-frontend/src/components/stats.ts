import {
  CLEANUP_BLOCKS,
  CLEANUP_NAMES,
  CLEANUP_SMOOTH,
  STRENGTH_NAMES,
} from '../settings';

export interface StatsData {
  originalWidth: number;
  originalHeight: number;
  originalUniqueColors: number;
  outputWidth: number;
  outputHeight: number;
  kColors: number;
  outputUniqueColors: number;
  noResize: boolean;
  algorithm: number;
  cleanup: number;
  cleanupStrength: number;
  gridScale: number;
}

const ALGORITHM_NAMES = [
  'K-Means (Perceptual)',
  'Median Cut',
  "Wu's Optimal",
  'Spatial (Pixel Art)',
  'Grey Wolf Optimization',
  'Differentiable Palette',
];

function formatNumber(n: number): string {
  return n.toLocaleString();
}

export function createStats(container: HTMLElement) {
  container.innerHTML = `<div class="stats-row"></div>`;
  const row = container.querySelector<HTMLDivElement>('.stats-row')!;

  return {
    update(data: StatsData) {
      const outputLabel = data.noResize ? 'Output' : 'Snapped';

      let thirdGroup: string;
      if (data.noResize) {
        const cleanupName = CLEANUP_NAMES[data.cleanup] ?? 'Off';
        // The engine falls back to a coarser grid when cells would go under 2 px, and
        // does not report which grid it used, so this is what was asked for.
        let cleanupValue = cleanupName;
        if (data.cleanup === CLEANUP_BLOCKS && data.gridScale !== 1) {
          cleanupValue = `${cleanupName}, ${data.gridScale}x finer grid requested`;
        } else if (data.cleanup === CLEANUP_SMOOTH) {
          cleanupValue = `${cleanupName}, ${STRENGTH_NAMES[data.cleanupStrength] ?? 'light'}`;
        }
        thirdGroup = `
          <div class="stat-group">
            <span class="stat-group-label">Mode</span>
            <div class="stat">
              <span class="stat-value">${ALGORITHM_NAMES[data.algorithm] ?? 'K-Means'}</span>
              <span class="stat-label">Algorithm</span>
            </div>
            <div class="stat">
              <span class="stat-value">${cleanupValue}</span>
              <span class="stat-label">Cleanup</span>
            </div>
          </div>`;
      } else {
        const cellW = data.originalWidth / data.outputWidth;
        const cellH = data.originalHeight / data.outputHeight;
        const cellSize =
          Math.abs(cellW - cellH) < 0.3
            ? `~${cellW.toFixed(1)}px`
            : `~${cellW.toFixed(1)} x ${cellH.toFixed(1)}px`;
        thirdGroup = `
          <div class="stat-group">
            <span class="stat-group-label">Grid</span>
            <div class="stat">
              <span class="stat-value">${cellSize}</span>
              <span class="stat-label">Detected cell size</span>
            </div>
          </div>`;
      }

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
          <span class="stat-group-label">${outputLabel}</span>
          <div class="stat">
            <span class="stat-value">${data.outputWidth} x ${data.outputHeight}</span>
            <span class="stat-label">Resolution</span>
          </div>
          <div class="stat">
            <span class="stat-value">${formatNumber(data.outputUniqueColors)}</span>
            <span class="stat-label">Unique colors</span>
          </div>
        </div>
        <div class="stat-divider"></div>
        ${thirdGroup}
      `;
    },
  };
}

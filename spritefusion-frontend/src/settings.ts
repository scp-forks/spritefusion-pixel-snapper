/** Everything that determines a processing result. */
export interface ProcessSettings {
  kColors: number;
  /** Keep the original dimensions instead of snapping to one pixel per cell. */
  noResize: boolean;
  /** Quantizer, 0-5. Only used in no-resize mode. */
  algorithm: number;
  /** Spatial cleanup after palette reduction. Only used in no-resize mode. */
  cleanup: number;
  /** How much Smooth may remove: 1 light, 2 medium, 3 strong. Only used by Smooth. */
  cleanupStrength: number;
  /** How many times finer than the detected grid to work. Only used by Blocks. */
  gridScale: number;
  /**
   * How the pixel grid is found: 0 the original walker, 1 the elastic fit on the
   * source image. Used wherever a grid is: snap mode, Blocks, and Smooth's scale.
   */
  gridFitter: number;
}

export const CLEANUP_OFF = 0;
export const CLEANUP_BLOCKS = 1;
export const CLEANUP_SMOOTH = 2;

export const CLEANUP_NAMES = ['Off', 'Blocks', 'Smooth'];
export const STRENGTH_LIGHT = 1;
export const FITTER_WALK = 0;
export const FITTER_NAMES = ['Standard', 'Elastic'];
export const STRENGTH_NAMES = ['', 'light', 'medium', 'strong'];

export const DEFAULT_SETTINGS: ProcessSettings = {
  kColors: 16,
  noResize: false,
  algorithm: 0,
  cleanup: CLEANUP_OFF,
  cleanupStrength: STRENGTH_LIGHT,
  gridScale: 1,
  gridFitter: FITTER_WALK,
};

/**
 * The settings as the engine will actually use them. Controls that are hidden in the
 * current mode keep their value in the UI, but must not reach the engine (it rejects
 * cleanup in snap mode) and must not split the cache into identical entries.
 */
export function effectiveSettings(s: ProcessSettings): ProcessSettings {
  if (!s.noResize) {
    return {
      ...s,
      algorithm: 0,
      cleanup: CLEANUP_OFF,
      cleanupStrength: STRENGTH_LIGHT,
      gridScale: 1,
    };
  }
  return {
    ...s,
    cleanupStrength: s.cleanup === CLEANUP_SMOOTH ? s.cleanupStrength : STRENGTH_LIGHT,
    gridScale: s.cleanup === CLEANUP_BLOCKS ? s.gridScale : 1,
    // Plain palette reduction uses no grid at all.
    gridFitter: s.cleanup === CLEANUP_OFF ? FITTER_WALK : s.gridFitter,
  };
}

/** Cache identity of a result: equal keys mean byte-identical output. */
export function settingsKey(s: ProcessSettings): string {
  const e = effectiveSettings(s);
  return e.noResize
    ? `noresize:${e.kColors}:${e.algorithm}:${e.cleanup}:${e.cleanupStrength}:${e.gridScale}:${e.gridFitter}`
    : `snap:${e.kColors}:${e.gridFitter}`;
}

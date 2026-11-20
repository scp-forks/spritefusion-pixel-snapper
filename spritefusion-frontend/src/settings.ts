/** Everything that determines a set of results. */
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
  /**
   * How the pixel grid is found: 0 the original walker, 1 the elastic fit on the
   * source image. Used wherever a grid is: snap mode, Blocks, and Smooth's scale.
   */
  gridFitter: number;
  /**
   * A grid width the user typed, added to the sizes on offer; 0 = none. Only used
   * where sizes are offered: snap mode and Blocks.
   */
  targetWidth: number;
  /**
   * How a grid cell's colour is chosen: 0 the majority (clean), 1 detail (pixels that
   * differ from their surroundings count for more). Only used where sizes are offered.
   */
  vote: number;
  /**
   * Alpha (1-255) at or above which a source pixel is opaque; fainter ones become
   * fully transparent. Results never have partial alpha. Used in every mode.
   */
  alphaThreshold: number;
}

export const CLEANUP_OFF = 0;
export const CLEANUP_BLOCKS = 1;
export const CLEANUP_SMOOTH = 2;

export const CLEANUP_NAMES = ['Off', 'Blocks', 'Smooth'];
export const STRENGTH_LIGHT = 1;
export const STRENGTH_NAMES = ['', 'light', 'medium', 'strong'];
export const FITTER_WALK = 0;
export const FITTER_NAMES = ['Standard', 'Elastic'];
export const VOTE_MAJORITY = 0;
export const VOTE_DETAIL = 1;

export const DEFAULT_SETTINGS: ProcessSettings = {
  kColors: 16,
  noResize: false,
  algorithm: 0,
  cleanup: CLEANUP_OFF,
  cleanupStrength: STRENGTH_LIGHT,
  gridFitter: FITTER_WALK,
  targetWidth: 0,
  vote: VOTE_MAJORITY,
  alphaThreshold: 128,
};

/** Modes that produce several sizes to choose from. */
export function offersSizes(s: ProcessSettings): boolean {
  return !s.noResize || s.cleanup === CLEANUP_BLOCKS;
}

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
    };
  }
  return {
    ...s,
    cleanupStrength: s.cleanup === CLEANUP_SMOOTH ? s.cleanupStrength : STRENGTH_LIGHT,
    // Plain palette reduction uses no grid at all.
    gridFitter: s.cleanup === CLEANUP_OFF ? FITTER_WALK : s.gridFitter,
    targetWidth: s.cleanup === CLEANUP_BLOCKS ? s.targetWidth : 0,
    vote: s.cleanup === CLEANUP_BLOCKS ? s.vote : VOTE_MAJORITY,
  };
}

/** Cache identity of a set of results: equal keys mean byte-identical output. */
export function settingsKey(s: ProcessSettings): string {
  const e = effectiveSettings(s);
  return e.noResize
    ? `noresize:${e.kColors}:${e.algorithm}:${e.cleanup}:${e.cleanupStrength}:${e.gridFitter}:${e.targetWidth}:${e.vote}:${e.alphaThreshold}`
    : `snap:${e.kColors}:${e.gridFitter}:${e.targetWidth}:${e.vote}:${e.alphaThreshold}`;
}

/** Lattice score at or above which the image clearly has a pixel grid. */
export const CLEAR_LATTICE_SCORE = 0.2;
/** Lattice score below which no pixel grid was found. */
export const NO_LATTICE_SCORE = 0.07;

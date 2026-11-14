/** Everything that determines a processing result. */
export interface ProcessSettings {
  kColors: number;
  /** Keep the original dimensions instead of snapping to one pixel per cell. */
  noResize: boolean;
  /** Quantizer, 0-5. Only used in no-resize mode. */
  algorithm: number;
  /** Spatial cleanup after palette reduction. Only used in no-resize mode. */
  cleanup: number;
  /** How many times finer than the detected grid to work. Only used by Blocks. */
  gridScale: number;
}

export const CLEANUP_OFF = 0;
export const CLEANUP_BLOCKS = 1;

export const CLEANUP_NAMES = ['Off', 'Blocks'];

export const DEFAULT_SETTINGS: ProcessSettings = {
  kColors: 16,
  noResize: false,
  algorithm: 0,
  cleanup: CLEANUP_OFF,
  gridScale: 1,
};

/**
 * The settings as the engine will actually use them. Controls that are hidden in the
 * current mode keep their value in the UI, but must not reach the engine (it rejects
 * cleanup in snap mode) and must not split the cache into identical entries.
 */
export function effectiveSettings(s: ProcessSettings): ProcessSettings {
  if (!s.noResize) {
    return { ...s, algorithm: 0, cleanup: CLEANUP_OFF, gridScale: 1 };
  }
  if (s.cleanup !== CLEANUP_BLOCKS) {
    return { ...s, gridScale: 1 };
  }
  return { ...s };
}

/** Cache identity of a result: equal keys mean byte-identical output. */
export function settingsKey(s: ProcessSettings): string {
  const e = effectiveSettings(s);
  return e.noResize
    ? `noresize:${e.kColors}:${e.algorithm}:${e.cleanup}:${e.gridScale}`
    : `snap:${e.kColors}`;
}

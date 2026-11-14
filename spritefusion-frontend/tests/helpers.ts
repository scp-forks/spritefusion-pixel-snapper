import type { CachedResult, ResultSet } from '../src/utils/result-cache.ts';
import { DEFAULT_SETTINGS, type ProcessSettings } from '../src/settings.ts';

let serial = 0;

export function result(cols: number, extra: Partial<CachedResult> = {}): CachedResult {
  return {
    bytes: new Uint8Array(16),
    url: `blob:test-${serial++}`,
    width: cols,
    height: cols,
    uniqueColors: 16,
    cols,
    rows: cols,
    gridScale: 2,
    isLegacy: false,
    colCuts: new Uint32Array(0),
    rowCuts: new Uint32Array(0),
    lostDetail: new Uint8Array(0),
    ...extra,
  };
}

export function resultSet(
  key: string,
  results: CachedResult[],
  settings: Partial<ProcessSettings> = {},
): ResultSet {
  return { key, settings: { ...DEFAULT_SETTINGS, ...settings }, results, latticeScore: 0.3 };
}

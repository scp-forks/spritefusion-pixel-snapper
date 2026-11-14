import type { ProcessSettings } from '../settings';

/** One finished image. */
export interface CachedResult {
  bytes: Uint8Array;
  url: string;
  width: number;
  height: number;
  uniqueColors: number;
  /** Cells of the grid it was built on; 0 when the mode uses no grid. */
  cols: number;
  rows: number;
  /** How many times finer than the detected grid. 1 = the detected grid. */
  gridScale: number;
  /** Exactly what the original engine returns for this image. */
  isLegacy: boolean;
}

/** Everything one engine run produced: one result, or several sizes to choose from. */
export interface ResultSet {
  /** `settingsKey(settings)`. */
  key: string;
  /** The effective settings that produced this set. */
  settings: ProcessSettings;
  /** Coarsest first. Never empty. */
  results: CachedResult[];
  /** Is there a pixel grid in the image; 0 when the mode uses no grid. */
  latticeScore: number;
}

// Bounds so a long session of slider-dragging can't pin unbounded blob memory.
export const MAX_ENTRIES = 48;
export const MAX_BYTES = 128 * 1024 * 1024;

function sizeOf(set: ResultSet): number {
  return set.results.reduce((sum, r) => sum + r.bytes.length, 0);
}

export function createResultCache() {
  // Map iterates in insertion order, so re-inserting on read gives LRU ordering.
  const entries = new Map<string, ResultSet>();
  let totalBytes = 0;
  let pinnedKey: string | null = null;

  function drop(key: string) {
    const set = entries.get(key);
    if (!set) return;
    entries.delete(key);
    totalBytes -= sizeOf(set);
    for (const r of set.results) URL.revokeObjectURL(r.url);
  }

  function evict() {
    for (const key of entries.keys()) {
      if (entries.size <= MAX_ENTRIES && totalBytes <= MAX_BYTES) return;
      if (key === pinnedKey) continue;
      drop(key);
    }
  }

  return {
    get(key: string): ResultSet | null {
      const set = entries.get(key);
      if (!set) return null;
      // Reinsert to mark as most recently used.
      entries.delete(key);
      entries.set(key, set);
      return set;
    },

    /**
     * Store a set. With `pin`, it becomes the protected set **before** anything is
     * evicted: a set about to go on screen must not be thrown out by its own arrival
     * (several full-size images can exceed what is left of the budget while the old
     * set is still pinned).
     */
    set(set: ResultSet, options: { pin?: boolean } = {}) {
      drop(set.key);
      entries.set(set.key, set);
      totalBytes += sizeOf(set);
      if (options.pin) pinnedKey = set.key;
      evict();
    },

    /**
     * Protect the set on screen from eviction: all of it, because every one of its
     * images is visible as a thumbnail, not only the one in the comparison view.
     */
    pin(key: string) {
      pinnedKey = key;
      evict();
    },

    get totalBytes(): number {
      return totalBytes;
    },

    has(key: string): boolean {
      return entries.has(key);
    },

    /** Every entry describes one source image; a new image invalidates all of them. */
    clear() {
      for (const set of entries.values()) {
        for (const r of set.results) URL.revokeObjectURL(r.url);
      }
      entries.clear();
      totalBytes = 0;
      pinnedKey = null;
    },

    get size(): number {
      return entries.size;
    },
  };
}

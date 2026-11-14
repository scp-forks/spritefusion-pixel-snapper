import type { ProcessSettings } from '../settings';

export interface CachedResult {
  /** `settingsKey(settings)`. */
  key: string;
  bytes: Uint8Array;
  url: string;
  width: number;
  height: number;
  uniqueColors: number;
  /** The effective settings that produced this result. */
  settings: ProcessSettings;
}

// Bounds so a long session of slider-dragging can't pin unbounded blob memory.
const MAX_ENTRIES = 64;
const MAX_BYTES = 96 * 1024 * 1024;

export function createResultCache() {
  // Map iterates in insertion order, so re-inserting on read gives LRU ordering.
  const entries = new Map<string, CachedResult>();
  let totalBytes = 0;
  let pinnedKey: string | null = null;

  function drop(key: string) {
    const entry = entries.get(key);
    if (!entry) return;
    entries.delete(key);
    totalBytes -= entry.bytes.length;
    URL.revokeObjectURL(entry.url);
  }

  function evict() {
    for (const key of entries.keys()) {
      if (entries.size <= MAX_ENTRIES && totalBytes <= MAX_BYTES) return;
      if (key === pinnedKey) continue;
      drop(key);
    }
  }

  return {
    get(key: string): CachedResult | null {
      const entry = entries.get(key);
      if (!entry) return null;
      // Reinsert to mark as most recently used.
      entries.delete(key);
      entries.set(key, entry);
      return entry;
    },

    set(entry: CachedResult) {
      drop(entry.key);
      entries.set(entry.key, entry);
      totalBytes += entry.bytes.length;
      evict();
    },

    /** Protect the currently displayed result from eviction. */
    pin(key: string) {
      pinnedKey = key;
    },

    /** Every entry describes one source image; a new image invalidates all of them. */
    clear() {
      for (const entry of entries.values()) URL.revokeObjectURL(entry.url);
      entries.clear();
      totalBytes = 0;
      pinnedKey = null;
    },

    get size(): number {
      return entries.size;
    },
  };
}

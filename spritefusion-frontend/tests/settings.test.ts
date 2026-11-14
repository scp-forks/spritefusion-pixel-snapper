import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  CLEANUP_BLOCKS,
  CLEANUP_OFF,
  CLEANUP_SMOOTH,
  DEFAULT_SETTINGS,
  effectiveSettings,
  offersSizes,
  settingsKey,
  type ProcessSettings,
} from '../src/settings.ts';

const base: ProcessSettings = { ...DEFAULT_SETTINGS };

test('controls hidden in snap mode do not reach the engine or split the cache', () => {
  const hidden = { ...base, algorithm: 3, cleanup: CLEANUP_BLOCKS, cleanupStrength: 3 };
  assert.deepEqual(effectiveSettings(hidden), base);
  assert.equal(settingsKey(hidden), settingsKey(base));
});

test('everything that changes the result changes the key', () => {
  const variants: ProcessSettings[] = [
    base,
    { ...base, kColors: 8 },
    { ...base, gridFitter: 1 },
    { ...base, targetWidth: 64 },
    { ...base, noResize: true },
    { ...base, noResize: true, algorithm: 2 },
    { ...base, noResize: true, cleanup: CLEANUP_BLOCKS },
    { ...base, noResize: true, cleanup: CLEANUP_BLOCKS, gridFitter: 1 },
    { ...base, noResize: true, cleanup: CLEANUP_BLOCKS, targetWidth: 64 },
    { ...base, noResize: true, cleanup: CLEANUP_SMOOTH },
    { ...base, noResize: true, cleanup: CLEANUP_SMOOTH, cleanupStrength: 2 },
    { ...base, noResize: true, cleanup: CLEANUP_SMOOTH, gridFitter: 1 },
  ];
  const keys = variants.map(settingsKey);
  assert.equal(new Set(keys).size, keys.length, keys.join('\n'));
});

test('settings with no effect in the current mode share a key', () => {
  const plain = { ...base, noResize: true, cleanup: CLEANUP_OFF };
  // Plain palette reduction uses no grid: fitter and typed width are irrelevant.
  assert.equal(settingsKey({ ...plain, gridFitter: 1, targetWidth: 99 }), settingsKey(plain));
  // Strength belongs to Smooth, typed widths to modes that offer sizes.
  const blocks = { ...base, noResize: true, cleanup: CLEANUP_BLOCKS };
  assert.equal(settingsKey({ ...blocks, cleanupStrength: 3 }), settingsKey(blocks));
  const smooth = { ...base, noResize: true, cleanup: CLEANUP_SMOOTH };
  assert.equal(settingsKey({ ...smooth, targetWidth: 99 }), settingsKey(smooth));
});

test('sizes are offered in snap mode and Blocks only', () => {
  assert.ok(offersSizes(base));
  assert.ok(offersSizes({ ...base, noResize: true, cleanup: CLEANUP_BLOCKS }));
  assert.ok(!offersSizes({ ...base, noResize: true, cleanup: CLEANUP_OFF }));
  assert.ok(!offersSizes({ ...base, noResize: true, cleanup: CLEANUP_SMOOTH }));
});

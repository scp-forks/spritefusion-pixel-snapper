import { test } from 'node:test';
import assert from 'node:assert/strict';
import { chooseSelection, nearestByWidth } from '../src/selection.ts';
import { result, resultSet } from './helpers.ts';

/** 24 (detected), 37, 48, 72 wide. */
function ladder(targetWidth = 0) {
  return resultSet(
    'ladder',
    [result(24, { gridScale: 1, isLegacy: true }), result(37, { gridScale: 1.5 }), result(48), result(72, { gridScale: 3 })],
    { targetWidth },
  );
}

test('with nothing to go on, the detected grid is selected', () => {
  assert.equal(chooseSelection(ladder(), null, false), 0);
  // The detected grid is not always the first: a typed width can be coarser.
  const set = resultSet('s', [result(12), result(24, { gridScale: 1 }), result(48)], { targetWidth: 12 });
  assert.equal(chooseSelection(set, null, false), 1);
});

test('a width the user just submitted is selected', () => {
  assert.equal(chooseSelection(ladder(48), null, true), 2);
  // ...the nearest size, when the engine landed a few cells off.
  assert.equal(chooseSelection(ladder(50), null, true), 2);
});

test('resubmitting a width selects it again, over the size being looked at', () => {
  // Codex, Phase 5 review: submit 48, click the 72 thumbnail, submit 48 again. The
  // set comes from the cache with the same target as before; 48 must be selected.
  const set = ladder(48);
  const lookingAt = set.results[3];
  assert.equal(chooseSelection(set, lookingAt, true), 2);
  // Without a fresh submission the size being looked at is kept.
  assert.equal(chooseSelection(set, lookingAt, false), 3);
});

test('the size being looked at is kept when every size shifts a little', () => {
  const before = ladder().results[2]; // 48 wide
  const shifted = resultSet('shifted', [result(25, { gridScale: 1 }), result(38), result(50), result(75)]);
  assert.equal(chooseSelection(shifted, before, false), 2);
  // More than a tenth away is a different size: fall back to the detected grid.
  const far = resultSet('far', [result(25, { gridScale: 1 }), result(60), result(90)]);
  assert.equal(chooseSelection(far, before, false), 0);
});

test('sets without a grid have one result and it is selected', () => {
  const plain = resultSet('plain', [result(0, { cols: 0, rows: 0, gridScale: 1 })]);
  assert.equal(nearestByWidth(plain, 100), -1);
  assert.equal(chooseSelection(plain, ladder().results[2], true), 0);
});

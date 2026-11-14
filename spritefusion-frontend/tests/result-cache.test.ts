import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createResultCache, MAX_BYTES, MAX_ENTRIES } from '../src/utils/result-cache.ts';
import { result, resultSet } from './helpers.ts';

const MiB = 1024 * 1024;

/** Record every URL the cache revokes while `fn` runs. */
function revokedDuring(fn: () => void): string[] {
  const revoked: string[] = [];
  const original = URL.revokeObjectURL;
  URL.revokeObjectURL = (url: string) => void revoked.push(url);
  try {
    fn();
  } finally {
    URL.revokeObjectURL = original;
  }
  return revoked;
}

function big(key: string, mebibytes: number, images = 4) {
  return resultSet(
    key,
    Array.from({ length: images }, (_, i) =>
      result(100 + i, { bytes: new Uint8Array((mebibytes * MiB) / images) }),
    ),
  );
}

test('a set that is about to be shown survives its own arrival', () => {
  // Codex, Phase 5 review: the old set is still pinned when the new one is stored,
  // and together they exceed the budget. The new one used to be evicted at once and
  // the page was handed revoked URLs.
  assert.ok(80 * MiB + 60 * MiB > MAX_BYTES);
  const cache = createResultCache();
  const onScreen = big('old', 80);
  const incoming = big('new', 60);
  cache.set(onScreen, { pin: true });

  const revoked = revokedDuring(() => cache.set(incoming, { pin: true }));

  assert.ok(cache.has('new'), 'the incoming set was evicted');
  for (const r of incoming.results) assert.ok(!revoked.includes(r.url), 'an incoming URL was revoked');
  assert.ok(!cache.has('old'), 'the old set should have made room');
  assert.deepEqual(revoked.sort(), onScreen.results.map((r) => r.url).sort());
  assert.equal(cache.totalBytes, 60 * MiB);
});

test('a set stored without a pin does not displace the one on screen', () => {
  const cache = createResultCache();
  const onScreen = big('shown', 100);
  cache.set(onScreen, { pin: true });
  const revoked = revokedDuring(() => cache.set(big('background', 60)));
  assert.ok(cache.has('shown'));
  for (const r of onScreen.results) assert.ok(!revoked.includes(r.url));
  assert.ok(!cache.has('background'), 'over budget, and nothing else can go');
});

test('every image of a dropped set is released, exactly once', () => {
  const cache = createResultCache();
  const a = resultSet('a', [result(10), result(20), result(30)]);
  cache.set(a);
  // Replacing a key releases the images it held.
  const replaced = revokedDuring(() => cache.set(resultSet('a', [result(11)])));
  assert.deepEqual(replaced.sort(), a.results.map((r) => r.url).sort());
  // Clearing releases the rest.
  const cleared = revokedDuring(() => cache.clear());
  assert.equal(cleared.length, 1);
  assert.equal(cache.size, 0);
  assert.equal(cache.totalBytes, 0);
});

test('least recently used sets go first, and reading a set renews it', () => {
  const cache = createResultCache();
  for (let i = 0; i < MAX_ENTRIES; i++) cache.set(resultSet(`k${i}`, [result(i + 1)]));
  assert.ok(cache.get('k0')); // renew the oldest
  cache.set(resultSet('one-more', [result(999)]));
  assert.equal(cache.size, MAX_ENTRIES);
  assert.ok(cache.has('k0'), 'a set read just now was evicted');
  assert.ok(!cache.has('k1'), 'the least recently used set was kept');
  assert.equal(cache.get('missing'), null);
});

test('pinning protects a set that would otherwise be the first to go', () => {
  const cache = createResultCache();
  cache.set(resultSet('k0', [result(1)]));
  cache.pin('k0');
  for (let i = 1; i <= MAX_ENTRIES; i++) cache.set(resultSet(`k${i}`, [result(i + 1)]));
  assert.ok(cache.has('k0'));
  assert.ok(!cache.has('k1'));
});

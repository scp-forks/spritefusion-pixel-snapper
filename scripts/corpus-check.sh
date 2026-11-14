#!/usr/bin/env bash
# Regression check over the sample images in other/.
#
#   scripts/corpus-check.sh            compare against scripts/corpus-baseline.jsonl
#   scripts/corpus-check.sh --update   rewrite the baseline (do this on purpose, and
#                                      read the diff before committing it)
#
# Also builds the WASM package and runs scripts/wasm-smoke.mjs when wasm-pack and node
# are installed.
#
# Each line is the CLI's --stats output for one image in one mode: output size, number
# of colours, and how many 4-connected same-colour regions are under 4 px and under
# 16 px (the speckle counts). Lines carry no paths or timings.
#
# The baseline was recorded on macOS/arm64. The OKLab quantizers use cbrt and powf, so
# another platform's libm may move a handful of pixels; treat small differences in the
# region counts there as noise and differences in size or colour count as real.
set -euo pipefail
cd "$(dirname "$0")/.."

BASELINE=scripts/corpus-baseline.jsonl
BIN=target/release/spritefusion-pixel-snapper
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

cargo build --release --quiet

run() { # run <image> <extra args...>
  local image="$1"; shift
  "$BIN" "$image" "$OUT/out.png" 16 "$@" --stats | tail -n 1
}

{
  for image in other/*.png; do
    run "$image"
    for algorithm in 0 1 2 3 4 5; do
      run "$image" --no-resize "--algorithm=$algorithm"
    done
    run "$image" --grid-scale=2
    run "$image" --no-resize --cleanup=blocks
    run "$image" --no-resize --cleanup=blocks --grid-scale=2
    for strength in light medium strong; do
      run "$image" --no-resize --cleanup=smooth "--strength=$strength"
    done
  done
} > "$OUT/current.jsonl"

# The JavaScript side of the WASM package, when the tools to build it are present.
if command -v wasm-pack >/dev/null && command -v node >/dev/null; then
  wasm-pack build --target web --out-dir pkg --release >/dev/null 2>&1
  node scripts/wasm-smoke.mjs
else
  echo "Skipping the WASM smoke test (needs wasm-pack and node)."
fi

if [[ "${1:-}" == "--update" ]]; then
  cp "$OUT/current.jsonl" "$BASELINE"
  echo "Baseline rewritten: $(wc -l < "$BASELINE" | tr -d ' ') lines in $BASELINE"
  exit 0
fi

if [[ ! -f "$BASELINE" ]]; then
  echo "No baseline at $BASELINE. Run: scripts/corpus-check.sh --update" >&2
  exit 2
fi

if diff -u "$BASELINE" "$OUT/current.jsonl"; then
  echo "Corpus check passed: $(wc -l < "$BASELINE" | tr -d ' ') results match the baseline."
else
  echo >&2
  echo "Corpus check FAILED: output statistics differ from $BASELINE (see diff above)." >&2
  echo "If the change is intended, rerun with --update and commit the new baseline." >&2
  exit 1
fi

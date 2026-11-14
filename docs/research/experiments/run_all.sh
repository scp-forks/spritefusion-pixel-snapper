#!/usr/bin/env bash
# Reproduce every number in ../02-experiments.md.
#   uv venv /tmp/psnap/venv && uv pip install --python /tmp/psnap/venv/bin/python numpy pillow scipy scikit-image
#   PY=/tmp/psnap/venv/bin/python ./run_all.sh
set -euo pipefail
cd "$(dirname "$0")"
REPO="$(cd ../../.. && pwd)"; PY="${PY:-python3}"; export PSNAP_OUT="${PSNAP_OUT:-/tmp/psnap/out}"; mkdir -p "$PSNAP_OUT"
(cd "$REPO" && cargo build --release -q)
B="$REPO/target/release/spritefusion-pixel-snapper"
for a in 0 1 2 3 4 5; do "$B" "$REPO/other/colliseum.png" "$PSNAP_OUT/col_k16_a$a.png" 16 --no-resize --algorithm=$a >/dev/null; done
for k in 4 8 32 64; do "$B" "$REPO/other/colliseum.png" "$PSNAP_OUT/col_k${k}_a0.png" $k --no-resize --algorithm=0 >/dev/null; done
"$B" "$REPO/other/colliseum.png" "$PSNAP_OUT/col_snap_k16.png" 16 >/dev/null
"$B" "$REPO/other/pixel-art-sample-1024x1024.png" "$PSNAP_OUT/sample_k16_a0.png" 16 --no-resize --algorithm=0 >/dev/null
for s in e1_baseline e2_local_grid e3_despeckle e4_scale_sweep e5_snap_in_place e6_diagnose e7_unmix e8_tiling e9_lattice_score e10_elastic_sweep e11_detail_reducers e13_uniform_best_phase; do
  echo "=================== $s"; "$PY" "$s.py"
done

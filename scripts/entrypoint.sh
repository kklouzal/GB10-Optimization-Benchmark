#!/usr/bin/env bash
set -Eeuo pipefail

LAB_HOME="${GB10_LAB_HOME:-/opt/gb10-spark-perf-lab}"
RESULTS_ROOT="${GB10_RESULTS:-/results}"
mkdir -p "$RESULTS_ROOT"

usage() {
  cat <<'USAGE'
gb10-spark-perf-lab commands:
  all             collect + benchmark + analyze
  collect         collect host/container/GPU/firmware/kernel state
  bench           run PyTorch/CUDA/nvbandwidth/DCGM/fio/STREAM probes where available
  lowp            run only the FP8/MXFP8/NVFP4 low-precision benchmark
  tunables        generate tunability matrix only
  analyze         create report.md from an existing result directory
  tune-plan       print supported/experimental tuning candidates; does not apply
  apply-safe      apply only low-risk runtime knobs; requires GB10_APPLY=1
  shell           open bash

Environment:
  GB10_RESULTS=/results                  output root
  GB10_OUT=/results/custom-dir           fixed output dir
  RUN_DCGM=0|1                           run baked-in in-container DCGM diagnostics
  RUN_DCGM_LEVEL=1|2|3|4                 default 1; higher is heavier
  DCGM_TIMEOUT=2400                      timeout in seconds for the DCGM run
  RUN_NVBANDWIDTH=1                      run nvbandwidth if installed
  RUN_FIO=0|1                            optional storage test on /results
  RUN_STREAM=1                           run CPU STREAM-like memory test
  RUN_PROFILING_HINTS=1                  collect nsys/ncu availability
  RUN_MATMUL=1                           run dense TF32/FP32/BF16/FP16 GEMM profiling
  RUN_COPY_BENCH=1                       run internal device/pinned copy bandwidth probes
  BENCH_SECONDS=20                       telemetry benchmark duration per dtype/size family
  GB10_VBOOST_VALUES=auto                auto-sweep 0..advertised vboost max by default
  GB10_VBOOST_SETTLE_S=5                 seconds to settle after each vboost change
  RUN_TUNABLES=1                         collect tunability matrix
  RUN_LOWP=1                             run FP8/MXFP8/NVFP4 low-precision benchmarks
  LOWP_VBOOST_VALUES=current             current|auto|roundtrip|0,3,4,3,0
  LOWP_GPU_CLOCK_LOCKS=reset;2400,2600   unlocked baseline plus semicolon-separated lock ranges
  LOWP_GPU_CLOCK_SETTLE_S=5              settle time after each lock/unlock change
  LOWP_SECONDS=12                        seconds per low-precision case
  LOWP_SHAPES=...                        comma-separated N or MxNxK shapes
  GB10_APPLY=1                           required for apply-safe
USAGE
}

cmd="${1:-all}"
shift || true

case "$cmd" in
  all)
    exec "$LAB_HOME/scripts/run-all.sh" "$@"
    ;;
  collect)
    exec "$LAB_HOME/scripts/collect.sh" "$@"
    ;;
  bench)
    exec "$LAB_HOME/scripts/bench.sh" "$@"
    ;;
  lowp)
    out="${GB10_OUT:-${RESULTS_ROOT}/gb10-lowp-$(date -u +%Y%m%dT%H%M%SZ)}"
    exec python3 "$LAB_HOME/scripts/gb10-lowp-bench.py" --out "$out" "$@"
    ;;
  tunables)
    out="${GB10_OUT:-${RESULTS_ROOT}/gb10-tunables-$(date -u +%Y%m%dT%H%M%SZ)}"
    exec python3 "$LAB_HOME/scripts/gb10-tunables.py" --out "$out" "$@"
    ;;
  analyze)
    exec "$LAB_HOME/scripts/gb10-analyze.py" "$@"
    ;;
  tune-plan)
    exec "$LAB_HOME/scripts/tune-plan.sh" "$@"
    ;;
  apply-safe)
    exec "$LAB_HOME/scripts/apply-safe.sh" "$@"
    ;;
  shell|bash)
    exec /bin/bash "$@"
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    echo "Unknown command: $cmd" >&2
    usage >&2
    exit 2
    ;;
esac

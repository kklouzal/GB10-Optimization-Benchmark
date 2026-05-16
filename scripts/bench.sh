#!/usr/bin/env bash
set -Eeuo pipefail
source "${GB10_LAB_HOME:-/opt/gb10-spark-perf-lab}/scripts/common.sh"
NO_ARCHIVE=0
[[ "${1:-}" == "--no-archive" ]] && NO_ARCHIVE=1
mkdir -p "$OUT"/{bench,gpu,logs}
log "running benchmarks into $OUT"

run bench/cuda_smoke 'if command -v gb10-cuda-smoke >/dev/null; then gb10-cuda-smoke; else echo "gb10-cuda-smoke not available"; fi'

log "running PyTorch benchmark"
timeout "${BENCH_TIMEOUT:-1800}" python3 "$LAB_HOME/scripts/gb10-bench.py" --out "$OUT/bench" > "$OUT/bench/torch_bench_stdout.txt" 2> "$OUT/bench/torch_bench_stderr.txt" || true

if [[ "${RUN_NVBANDWIDTH:-1}" == "1" ]]; then
  run bench/nvbandwidth_list 'command -v nvbandwidth && nvbandwidth -l || true'
  run bench/nvbandwidth_json 'if command -v nvbandwidth >/dev/null; then nvbandwidth -j -i "${NVBANDWIDTH_SAMPLES:-5}" -b "${NVBANDWIDTH_BUFFER_MB:-512}"; else echo "nvbandwidth not installed"; fi'
fi

if [[ "${RUN_DCGM:-0}" == "1" ]]; then
  run bench/dcgm_diag 'if command -v dcgmi >/dev/null; then dcgmi discovery -l 2>/dev/null || true; dcgmi diag -r "${RUN_DCGM_LEVEL:-1}" 2>/dev/null || true; else echo "dcgmi not available in container or host"; fi'
else
  echo "Set RUN_DCGM=1 RUN_DCGM_LEVEL=1..4 to run DCGM diagnostics if dcgmi is available." > "$OUT/bench/dcgm_SKIPPED.txt"
fi

if [[ "${RUN_STREAM:-1}" == "1" ]]; then
  run bench/stream_like 'python3 - <<PY
import numpy as np, time, os, json
n_gib=float(os.environ.get("STREAM_GIB", "8"))
n=int(n_gib*1024**3/8/3)
a=np.ones(n,dtype=np.float64); b=np.ones(n,dtype=np.float64)*2; c=np.zeros(n,dtype=np.float64)
# warmup
c[:] = a + b
iters=int(os.environ.get("STREAM_ITERS","10"))
res=[]
for name,fn,bytes_per in [
  ("copy", lambda: np.copyto(c,a), 2*8*n),
  ("scale", lambda: np.multiply(a,3.0,out=b), 2*8*n),
  ("add", lambda: np.add(a,b,out=c), 3*8*n),
  ("triad", lambda: np.add(b,3.0*c,out=a), 3*8*n)]:
    times=[]
    for _ in range(iters):
        t=time.perf_counter(); fn(); times.append(time.perf_counter()-t)
    best=min(times)
    res.append({"name":name,"best_seconds":best,"GB_s":bytes_per/best/1e9,"GiB_s":bytes_per/best/1024**3})
print(json.dumps({"stream_gib":n_gib,"n":n,"iters":iters,"results":res}, indent=2))
PY'
fi

if [[ "${RUN_FIO:-0}" == "1" ]]; then
  run bench/fio_results_mount 'fio --name=gb10-lab-fio --directory="${GB10_FIO_DIR:-/results}" --size="${GB10_FIO_SIZE:-8G}" --rw=readwrite --bs=1M --iodepth=32 --numjobs=1 --direct=1 --time_based --runtime="${GB10_FIO_SECONDS:-60}" --group_reporting --output-format=json'
else
  echo "Set RUN_FIO=1 to run non-destructive temp-file fio benchmark in /results." > "$OUT/bench/fio_SKIPPED.txt"
fi

if [[ "$NO_ARCHIVE" == "0" ]]; then
  "$LAB_HOME/scripts/gb10-analyze.py" "$OUT" || true
  archive="$(archive_out)"
  log "created archive: $archive"
  echo "$archive"
fi

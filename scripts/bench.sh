#!/usr/bin/env bash
set -Eeuo pipefail
source "${GB10_LAB_HOME:-/opt/gb10-spark-perf-lab}/scripts/common.sh"
NO_ARCHIVE=0
[[ "${1:-}" == "--no-archive" ]] && NO_ARCHIVE=1
mkdir -p "$OUT"/{bench,gpu,logs,host}
log "running benchmarks into $OUT"

cat > "$OUT/bench/run_context.txt" <<EOFCTX
profile=${GB10_PROFILE:-unspecified}
cpuset=${GB10_CPUSET:-unspecified}
run_dcgm=${RUN_DCGM:-0}
run_dcgm_level=${RUN_DCGM_LEVEL:-1}
run_nvbandwidth=${RUN_NVBANDWIDTH:-1}
run_stream=${RUN_STREAM:-1}
run_fio=${RUN_FIO:-0}
bench_seconds=${BENCH_SECONDS:-20}
bench_sizes=${BENCH_SIZES:-4096,8192,12288,16384}
vboost_values=${GB10_VBOOST_VALUES:-auto}
vboost_settle_s=${GB10_VBOOST_SETTLE_S:-5}
run_tunables=${RUN_TUNABLES:-1}
run_lowp=${RUN_LOWP:-1}
lowp_vboost_values=${LOWP_VBOOST_VALUES:-current}
lowp_seconds=${LOWP_SECONDS:-12}
lowp_shapes=${LOWP_SHAPES:-default}
shm_hint=${GB10_SHM_SIZE:-unspecified}
omp_num_threads=${OMP_NUM_THREADS:-unset}
malloc_arena_max=${MALLOC_ARENA_MAX:-unset}
cuda_module_loading=${CUDA_MODULE_LOADING:-unset}
cuda_device_max_connections=${CUDA_DEVICE_MAX_CONNECTIONS:-unset}
EOFCTX

run_host host/reboot_preflight '
echo "== identity =="
date -Iseconds
uname -a
cat /etc/dgx-release 2>/dev/null || true
cat /etc/os-release | grep -E "PRETTY_NAME|VERSION_CODENAME" || true

echo
echo "== boot/perf state =="
cat /proc/cmdline
cat /sys/devices/system/cpu/vulnerabilities/* 2>/dev/null || true
cat /proc/sys/kernel/numa_balancing 2>/dev/null || true
grep -o "init_on_alloc=[01]" /proc/cmdline || true
grep -i huge /proc/meminfo || true
lscpu -e=CPU,CORE,SOCKET,NODE,MAXMHZ,MINMHZ,MHZ 2>/dev/null || true

echo
echo "== gpu =="
nvidia-smi
nvidia-smi -q -d PERFORMANCE,CLOCK,POWER,TEMPERATURE
nvidia-smi boost-slider -l 2>/dev/null || true
nvidia-smi power-profiles -l 2>/dev/null || true
nvidia-smi power-smoothing -ppd 2>/dev/null || true

echo
echo "== memory/process cleanliness =="
free -h
swapon --show
docker ps
ps -eo pid,ppid,psr,pcpu,pmem,comm,args --sort=-pcpu | head -n 40
'
run_host bench/gpu_state_before 'nvidia-smi -q -d PERFORMANCE,CLOCK,POWER,TEMPERATURE 2>/dev/null || true'
run bench/boost_slider_before 'nvidia-smi boost-slider -l 2>/dev/null || true'

# --- gb10-lowp-tunables-addon: tunables begin ---
if [[ "${RUN_TUNABLES:-1}" == "1" ]]; then
  log "collecting tunability matrix"
  mkdir -p "$OUT/tunables"
  timeout "${TUNABLES_TIMEOUT:-600}" python3 "$LAB_HOME/scripts/gb10-tunables.py" --out "$OUT/tunables" > "$OUT/tunables/tunables_stdout.txt" 2> "$OUT/tunables/tunables_stderr.txt" || true
else
  echo "Set RUN_TUNABLES=1 to collect the tunability matrix." > "$OUT/tunables_SKIPPED.txt"
fi
# --- gb10-lowp-tunables-addon: tunables end ---


run bench/cuda_smoke 'if command -v gb10-cuda-smoke >/dev/null; then gb10-cuda-smoke; else echo "gb10-cuda-smoke not available"; fi'

log "running PyTorch benchmark"
timeout "${BENCH_TIMEOUT:-1800}" python3 "$LAB_HOME/scripts/gb10-bench.py" --out "$OUT/bench" > "$OUT/bench/torch_bench_stdout.txt" 2> "$OUT/bench/torch_bench_stderr.txt" || true

# --- gb10-lowp-tunables-addon: lowp begin ---
if [[ "${RUN_LOWP:-1}" == "1" ]]; then
  log "running low-precision FP8/MXFP8/NVFP4 benchmark"
  mkdir -p "$OUT/bench/lowp"
  timeout "${LOWP_BENCH_TIMEOUT:-3600}" python3 "$LAB_HOME/scripts/gb10-lowp-bench.py" --out "$OUT/bench/lowp" > "$OUT/bench/lowp_bench_stdout.txt" 2> "$OUT/bench/lowp_bench_stderr.txt" || true
else
  echo "Set RUN_LOWP=1 to run FP8/MXFP8/NVFP4 low-precision benchmarks." > "$OUT/bench/lowp_SKIPPED.txt"
fi
# --- gb10-lowp-tunables-addon: lowp end ---


if [[ "${RUN_NVBANDWIDTH:-1}" == "1" ]]; then
  run bench/nvbandwidth_list 'command -v nvbandwidth && nvbandwidth -l || true'
  run bench/nvbandwidth_json 'if command -v nvbandwidth >/dev/null; then nvbandwidth -j -i "${NVBANDWIDTH_SAMPLES:-5}" -b "${NVBANDWIDTH_BUFFER_MB:-512}"; else echo "nvbandwidth not installed"; fi'
fi

if [[ "${RUN_DCGM:-0}" == "1" ]]; then
  log "bench/dcgm"
  GB10_OUT="$OUT" timeout "${DCGM_TIMEOUT:-2400}" "$LAB_HOME/scripts/dcgm-run.sh" > "$OUT/bench/dcgm_console.txt" 2>&1 || true
else
  echo "Set RUN_DCGM=1 RUN_DCGM_LEVEL=1..4 to run baked-in in-container DCGM diagnostics." > "$OUT/bench/dcgm_SKIPPED.txt"
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

run_host bench/gpu_state_after 'nvidia-smi -q -d PERFORMANCE,CLOCK,POWER,TEMPERATURE 2>/dev/null || true'
run bench/boost_slider_after 'nvidia-smi boost-slider -l 2>/dev/null || true'
run_host bench/system_state_after '
free -h
swapon --show
docker ps
ps -eo pid,ppid,psr,pcpu,pmem,comm,args --sort=-pcpu | head -n 40
'

if [[ "$NO_ARCHIVE" == "0" ]]; then
  "$LAB_HOME/scripts/gb10-analyze.py" "$OUT" || true
python3 "$LAB_HOME/scripts/gb10-report-append.py" "$OUT" || true
  archive="$(archive_out)"
  log "created archive: $archive"
  echo "$archive"
fi

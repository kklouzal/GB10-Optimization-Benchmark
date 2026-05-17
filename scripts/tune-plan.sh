#!/usr/bin/env bash
set -Eeuo pipefail
cat <<'PLAN'
# GB10 Spark tuning plan generator
# This script intentionally DOES NOT change the host.
# Use it as an A/B matrix. Change one variable at a time, benchmark, then revert.

## 0. Baseline snapshot
uname -a
cat /proc/cmdline
nvidia-smi -q -d CLOCK,POWER,PERFORMANCE,SUPPORTED_CLOCKS
nvidia-smi power-profiles -l 2>/dev/null || true
nvidia-smi boost-slider -l 2>/dev/null || true
nvidia-smi power-hint -l 2>/dev/null || true

## 1. Safe runtime baseline
sudo nvidia-smi -pm 1
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null | sort -u
cat /proc/sys/kernel/numa_balancing
cat /proc/cmdline | tr ' ' '\n' | grep -E 'init_on_alloc|mitigations|idle|nohz|rcu|isolcpus|hugepages' || true

## 2. GPU clock-lock A/B, if supported
# Reset first:
sudo nvidia-smi --reset-gpu-clocks || true
# Baseline: run gb10-lab bench and your real workload.
# Candidate broad lock range; adjust after reading supported clocks and max clocks:
sudo nvidia-smi --lock-gpu-clocks=2400,3003 --mode=0
# Re-run identical benchmark and compare report + live CSV.
# Revert:
sudo nvidia-smi --reset-gpu-clocks

## 3. Workload power-profile A/B, if supported
nvidia-smi power-profiles -l
nvidia-smi power-profiles -ld
# If an obvious compute/max-performance profile appears, test it by ID:
# sudo nvidia-smi power-profiles --set-requested <PROFILE_ID>
# nvidia-smi power-profiles --get-requested
# nvidia-smi power-profiles --get-enforced
# Revert:
# sudo nvidia-smi power-profiles --clear-requested <PROFILE_ID>

## 4. Host memory A/B
cat /sys/kernel/mm/transparent_hugepage/enabled
cat /sys/kernel/mm/transparent_hugepage/defrag
# Try for throughput-oriented runs:
# echo always | sudo tee /sys/kernel/mm/transparent_hugepage/enabled
# echo madvise | sudo tee /sys/kernel/mm/transparent_hugepage/enabled
# For 1G hugepages, create a separate GRUB entry; do not mix with clock tests.

## 5. Latency-heavy boot-entry experiments, separate GRUB entry only
# These are intentionally not applied by this project:
#   idle=poll
#   nohz_full=<cpuset>
#   rcu_nocbs=<cpuset>
#   isolcpus=managed_irq,domain,<cpuset>
#   irqaffinity=<housekeeping cpuset>
#   hugepagesz=1G hugepages=<N> default_hugepagesz=1G
#   preempt=none
# Benchmark with and without; keep only if your actual workload improves.

## 6. Container workload launch hygiene
# Use for real model benchmarks:
# docker run --rm -it --gpus all --ipc=host --network=host \
#   --ulimit memlock=-1 --ulimit stack=67108864 --shm-size=32g \
#   -e NVIDIA_DRIVER_CAPABILITIES=all \
#   -e CUDA_DEVICE_MAX_CONNECTIONS=1 \
#   -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
#   <ngc-image> <your-command>


## 7. Low-precision workload-relevant A/B
# BF16/FP16 GEMM exposes throttling. Actual GB10 LLM workloads should also test:
#   FP8 static GEMM
#   FP8 dynamic activation + static weight
#   Transformer Engine FP8 delayed scaling
#   Transformer Engine MXFP8 block scaling
#   Transformer Engine NVFP4 block scaling
# Example:
#   RUN_LOWP=1 LOWP_VBOOST_VALUES=current gb10-lab bench
# Sweep vboost specifically for low precision:
#   LOWP_VBOOST_VALUES=0,3,4,3,0 gb10-lab lowp

## 8. Tunability matrix
# Generate a read-only inventory of exposed GPU/kernel/sysfs/container knobs:
#   gb10-lab tunables
# or inside full runs:
#   RUN_TUNABLES=1 gb10-lab all
PLAN

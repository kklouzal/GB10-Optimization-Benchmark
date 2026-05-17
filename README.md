# GB10 Spark Perf Lab

A repo-ready diagnostics and benchmark container for NVIDIA DGX Spark / Dell Pro Max with GB10 systems. It is built on an NVIDIA NGC PyTorch container and is designed to produce a reproducible archive containing host state, DGX/NVIDIA package versions, firmware/update status, GPU clocks/throttle telemetry, CPU/memory/kernel state, CUDA/PyTorch benchmarks, nvbandwidth output, and an automatically generated tuning report.

The project has two goals:

1. Help GB10 owners identify misconfiguration, outdated kernels, firmware/update issues, thermal/power delivery constraints, and workload bottlenecks.
2. Provide a disciplined A/B framework for safe, experimental, and unsupported tuning candidates without blindly applying destructive changes.

## Build

```bash
docker buildx build --platform linux/arm64 \
  --build-arg BASE_IMAGE=nvcr.io/nvidia/pytorch:26.04-py3 \
  --build-arg BUILD_NVBANDWIDTH=1 \
  --build-arg NVBANDWIDTH_REF=v0.9 \
  --build-arg BUILD_DCGM=1 \
  --build-arg DCGM_CUDA_MAJOR=13 \
  -t gb10-spark-perf-lab:ngc .
```

The base image is configurable because NGC tags move over time. Prefer a current NGC PyTorch image that supports Blackwell/GB10 well.

## Run full collector + benchmark + analysis

Recommended official run shape for the current GB10 topology:
- performance cores only: `5-9,15-19`
- efficiency cores left free: `0-4,10-14`
- DCGM enabled at level 1
- large shared memory and explicit memlock/nofile limits

```bash
mkdir -p results

docker run --rm -it --gpus all \
  --cpuset-cpus=5-9,15-19 \
  --privileged --pid=host --net=host --ipc=host --uts=host \
  --ulimit memlock=-1 --ulimit nofile=1048576:1048576 \
  --shm-size=64g \
  --security-opt seccomp=unconfined \
  --cap-add SYS_ADMIN --cap-add SYS_PTRACE --cap-add PERFMON --cap-add IPC_LOCK --cap-add SYS_NICE \
  -e GB10_PROFILE=perf-cores-runtime-maxperf \
  -e GB10_CPUSET=5-9,15-19 \
  -e GB10_SHM_SIZE=64g \
  -e RUN_NVBANDWIDTH=1 \
  -e RUN_DCGM=1 \
  -e RUN_DCGM_LEVEL=1 \
  -e RUN_STREAM=1 \
  -e RUN_FIO=0 \
  -e OMP_NUM_THREADS=10 \
  -e MALLOC_ARENA_MAX=2 \
  -e BENCH_SECONDS=20 \
  -v /:/host:ro \
  -v /dev:/dev \
  -v /sys:/sys:ro \
  -v /proc:/host_proc:ro \
  -v "$PWD/results:/results" \
  gb10-spark-perf-lab:ngc all
```

The resulting archive now also includes an automatic reboot/preflight snapshot, run context, before/after GPU state snapshots, and a first-class per-vboost sweep that tests every advertised vboost value from 0 through the reported max.

The container prints an archive path like:

```text
/results/gb10-lab-HOST-YYYYMMDDTHHMMSSZ.tar.gz
```

Upload or share that archive for review. The most important file is `report.md`.

## Optional heavier diagnostics

DCGM is baked into the benchmark image by default. The official run above already enables a light `RUN_DCGM_LEVEL=1`. For a heavier pass, raise the level explicitly:

```bash
docker run --rm -it --gpus all \
  --cpuset-cpus=5-9,15-19 \
  --privileged --pid=host --net=host --ipc=host --uts=host \
  --ulimit memlock=-1 --ulimit nofile=1048576:1048576 \
  --shm-size=64g \
  --security-opt seccomp=unconfined \
  --cap-add SYS_ADMIN --cap-add SYS_PTRACE --cap-add PERFMON --cap-add IPC_LOCK --cap-add SYS_NICE \
  -e GB10_PROFILE=perf-cores-runtime-maxperf-dcgm2 \
  -e GB10_CPUSET=5-9,15-19 \
  -e GB10_SHM_SIZE=64g \
  -e RUN_DCGM=1 -e RUN_DCGM_LEVEL=2 \
  -e RUN_NVBANDWIDTH=1 -e RUN_STREAM=1 -e RUN_FIO=0 \
  -e OMP_NUM_THREADS=10 -e MALLOC_ARENA_MAX=2 \
  -v /:/host:ro -v /dev:/dev -v /sys:/sys:ro -v /proc:/host_proc:ro \
  -v "$PWD/results:/results" \
  gb10-spark-perf-lab:ngc all
```

For storage testing on the results mount:

```bash
-e RUN_FIO=1 -e GB10_FIO_SIZE=16G -e GB10_FIO_SECONDS=120
```

## Commands

```bash
gb10-lab all        # collect + bench + analyze + tarball
gb10-lab collect    # host/GPU/kernel/firmware inventory only
gb10-lab bench      # PyTorch/CUDA/nvbandwidth/DCGM/fio/STREAM probes
gb10-lab analyze    # generate report.md from a result directory
gb10-lab tune-plan  # print A/B tuning matrix; does not modify host
gb10-lab apply-safe # low-risk runtime knobs only; requires GB10_APPLY=1
```

## Automatic benchmark context captured in artifacts

Every run now records:
- a concise reboot/preflight snapshot (`host/reboot_preflight.txt`)
- benchmark run context including profile/cpuset/DCGM settings and vboost plan (`bench/run_context.txt`)
- GPU state before and after the benchmark (`bench/gpu_state_before.txt`, `bench/gpu_state_after.txt`)
- top-level boost-slider snapshots (`bench/boost_slider_before.txt`, `bench/boost_slider_after.txt`)
- top-level vboost summaries (`bench/vboost_summary.md`, `bench/vboost_summary.json`) that make the sweep outcome obvious at a glance
- per-vboost subdirectories with `torch_bench.json`, `torch_meta.json`, `nvidia_smi_live.csv`, and `nvidia_smi_dmon.csv`
- memory, swap, running containers, and top CPU consumers after the run

## Why privileged?

A normal GPU container can benchmark CUDA, but it cannot see enough of the host to diagnose GB10-specific performance limits. This project needs host PID/network namespaces, `/sys`, `/dev`, `nvidia-smi`, PCI/NVMe/NIC state, firmware tools, cpufreq, dmesg, and sometimes `nsenter` into PID 1's namespace. The default run command mounts host root read-only and does not apply host changes.

## Tuning philosophy

Do not mix changes. Establish a baseline, change one variable, run the same benchmark and real workload, then revert. The report separates:

- corrective fixes: update/reboot into newest kernel, firmware, package mismatches, throttling, power brake, thermal slowdown;
- safe runtime knobs: persistence mode, performance governor, NUMA balancing off, memlock limits;
- experimental A/B candidates: GPU clock locks, workload power profiles, THP/hugepages, idle/C-state settings, IRQ/NIC tuning;
- unsupported/hacky candidates: separate boot entries only, never silently applied by the container.

## What it benchmarks

- A built-in vboost sweep that tests every advertised value from `0..max` and records separate telemetry/artifacts for each setting.
- PyTorch GEMM throughput for TF32/FP32/BF16/FP16 across configurable matrix sizes.
- PyTorch device-to-device and pinned host-to-device/device-to-host copy bandwidth.
- NVIDIA nvbandwidth when built into the image.
- Native CUDA smoke/SAXPY bandwidth sanity probe.
- Optional CPU STREAM-like memory bandwidth.
- Optional fio temp-file storage benchmark.
- Optional DCGM diagnostics.
- Live `nvidia-smi` telemetry during benchmarks.

## Environment overrides

```bash
BENCH_SIZES=4096,8192,12288,16384
BENCH_SECONDS=20
BENCH_MAX_ALLOC_FRAC=0.55
GB10_VBOOST_VALUES=auto
GB10_VBOOST_SETTLE_S=5
NVBANDWIDTH_SAMPLES=5
NVBANDWIDTH_BUFFER_MB=512
GB10_PROFILE=perf-cores-runtime-maxperf
GB10_CPUSET=5-9,15-19
GB10_SHM_SIZE=64g
RUN_DCGM=1
RUN_DCGM_LEVEL=1
DCGM_TIMEOUT=2400
RUN_FIO=1
RUN_STREAM=1
REDACT=1
```

## Safe host modification mode

The default is read-only. To apply only low-risk runtime knobs:

```bash
docker run --rm -it --gpus all --privileged --pid=host --net=host --ipc=host \
  -e GB10_APPLY=1 \
  -v /dev:/dev -v /sys:/sys:rw -v /proc:/host_proc:rw \
  gb10-spark-perf-lab:ngc apply-safe
```

This sets NVIDIA persistence mode, CPU governor/EPP to performance where writable, NUMA balancing off, and swappiness to 1. It does not modify GRUB, firmware, clocks, power profiles, hugepages, or thermal controls.

## Experimental tuning

Generate the manual tuning matrix:

```bash
docker run --rm -it --gpus all gb10-spark-perf-lab:ngc tune-plan
```

The plan includes clock-lock A/B, power-profile A/B, THP/hugepage A/B, latency boot-entry experiments, and container launch hygiene.

## Privacy

By default the collector redacts common IPs, MACs, serial numbers, and token-like strings. Redaction is best-effort, not a security boundary. Review the archive before publishing.

## License

Apache-2.0.

<!-- gb10-lowp-tunables-addon -->
## Low-precision FP8 / MXFP8 / NVFP4 benchmarking

The suite can now collect workload-relevant low-precision data in addition to BF16/FP16/TF32 GEMM stress results. This is important for GB10 workloads that use MXFP8, NVFP4, FP8 KV cache, or mixed BF16/low-precision execution.

```bash
-e RUN_LOWP=1 \
-e LOWP_VBOOST_VALUES=current \
-e LOWP_SECONDS=20 \
-e LOWP_SHAPES='1024x8192x8192,2048x8192x8192,4096x8192x8192,8192x8192x8192'
```

For standalone low-precision sweeps, set `LOWP_VBOOST_VALUES=auto` or a roundtrip sequence such as `LOWP_VBOOST_VALUES=0,3,4,3,0`. Outputs are written under `bench/lowp/`; see `docs/LOWP-BENCHMARKS.md`.

## Tunability matrix

The suite can also produce a read-only inventory of exposed performance knobs: GPU boost/clock/power-profile capabilities, sysctl state, THP/hugepages, CPU idle/frequency knobs, fan/PWM exposure, PCIe power state, container limits, and tool availability.

```bash
gb10-lab tunables
# or in a full run
-e RUN_TUNABLES=1
```

Outputs are written under `tunables/`; see `docs/MAXPERF-TUNING-MATRIX.md`.

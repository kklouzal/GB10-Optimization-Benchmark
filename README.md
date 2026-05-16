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

```bash
mkdir -p results

docker run --rm -it --gpus all \
  --privileged --pid=host --net=host --ipc=host --uts=host \
  --cap-add SYS_ADMIN --cap-add SYS_PTRACE --cap-add PERFMON \
  -e RUN_NVBANDWIDTH=1 \
  -e RUN_DCGM=0 \
  -e RUN_STREAM=1 \
  -e RUN_FIO=0 \
  -e BENCH_SECONDS=20 \
  -v /:/host:ro \
  -v /dev:/dev \
  -v /sys:/sys:ro \
  -v /proc:/host_proc:ro \
  -v "$PWD/results:/results" \
  gb10-spark-perf-lab:ngc all
```

The container prints an archive path like:

```text
/results/gb10-lab-HOST-YYYYMMDDTHHMMSSZ.tar.gz
```

Upload or share that archive for review. The most important file is `report.md`.

## Optional heavier diagnostics

DCGM is baked into the benchmark image by default. It is still expensive, so keep it opt-in and run it only with the privileged/host-namespace launch shape shown below:

```bash
docker run --rm -it --gpus all \
  --privileged --pid=host --net=host --ipc=host --uts=host \
  -e RUN_DCGM=1 -e RUN_DCGM_LEVEL=2 \
  -e RUN_NVBANDWIDTH=1 \
  -v /:/host:ro -v /dev:/dev -v /sys:/sys:ro -v "$PWD/results:/results" \
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

## Why privileged?

A normal GPU container can benchmark CUDA, but it cannot see enough of the host to diagnose GB10-specific performance limits. This project needs host PID/network namespaces, `/sys`, `/dev`, `nvidia-smi`, PCI/NVMe/NIC state, firmware tools, cpufreq, dmesg, and sometimes `nsenter` into PID 1's namespace. The default run command mounts host root read-only and does not apply host changes.

## Tuning philosophy

Do not mix changes. Establish a baseline, change one variable, run the same benchmark and real workload, then revert. The report separates:

- corrective fixes: update/reboot into newest kernel, firmware, package mismatches, throttling, power brake, thermal slowdown;
- safe runtime knobs: persistence mode, performance governor, NUMA balancing off, memlock limits;
- experimental A/B candidates: GPU clock locks, workload power profiles, THP/hugepages, idle/C-state settings, IRQ/NIC tuning;
- unsupported/hacky candidates: separate boot entries only, never silently applied by the container.

## What it benchmarks

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
NVBANDWIDTH_SAMPLES=5
NVBANDWIDTH_BUFFER_MB=512
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

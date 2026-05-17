# Low-precision benchmarks: FP8, MXFP8, NVFP4

This add-on extends the GB10 Optimization Benchmark suite beyond BF16/FP16/TF32 stress GEMMs with probes that better match Blackwell/GB10 LLM workloads:

- PyTorch native FP8 `torch._scaled_mm`, where the selected NGC PyTorch build exposes it.
- Transformer Engine FP8 delayed-scaling Linear forward.
- Transformer Engine MXFP8 block-scaling Linear forward.
- Transformer Engine NVFP4 block-scaling Linear forward.
- Optional low-precision-specific vboost sweep.
- Live `nvidia-smi` telemetry and `dmon` capture for each low-precision run.

The goal is not to reproduce NVIDIA marketing sparse FP4 TOPS. The reported metric is **dense-equivalent TFLOP/s** using `2*M*N*K`, so it is useful for A/B comparisons on the same GB10 system across vboost, cooling, boot profile, IRQ placement, container flags, and framework images.

## Default behavior

`RUN_LOWP=1` runs one low-precision pass at the current vboost value. This is intentional: the main BF16/FP16 benchmark already has a first-class vboost sweep, and low-precision benchmarking can be expensive.

```bash
-e RUN_LOWP=1 \
-e LOWP_VBOOST_VALUES=current
```

To sweep every advertised vboost value specifically for FP8/MXFP8/NVFP4:

```bash
-e LOWP_VBOOST_VALUES=auto
```

To run a roundtrip sequence to reduce heat-soak/order bias:

```bash
-e LOWP_VBOOST_VALUES=roundtrip
```

To explicitly test selected values:

```bash
-e LOWP_VBOOST_VALUES=0,3,4,3,0
```

## Useful environment variables

```bash
RUN_LOWP=1
RUN_TORCH_FP8=1
RUN_TE_LOWP=1
RUN_TRTLLM_PROBE=1
LOWP_SECONDS=12
LOWP_WARMUP=12
LOWP_SHAPES='512x4096x4096,1024x8192x8192,2048x8192x8192,4096x8192x8192,8192x8192x8192'
LOWP_MAX_ALLOC_FRAC=0.55
LOWP_OUT_DTYPE=bf16
LOWP_TE_DTYPE=bf16
LOWP_DYNAMIC_QUANT=1
LOWP_VBOOST_VALUES=current
LOWP_ENABLE_DMON=1
```

Shape format is `M x N x K` for `A[M,K] @ B[K,N] -> C[M,N]`. A single integer such as `8192` is expanded to `8192x8192x8192`.

## Suggested GB10 run

```bash
docker run --rm -it --gpus all \
  --cpuset-cpus=5-9,15-19 \
  --privileged --pid=host --net=host --ipc=host --uts=host \
  --ulimit memlock=-1 --ulimit nofile=1048576:1048576 \
  --shm-size=64g \
  --security-opt seccomp=unconfined \
  --cap-add SYS_ADMIN --cap-add SYS_PTRACE --cap-add PERFMON --cap-add IPC_LOCK --cap-add SYS_NICE \
  -e GB10_PROFILE=perf-cores-vboost3-lowp \
  -e GB10_CPUSET=5-9,15-19 \
  -e RUN_LOWP=1 \
  -e LOWP_VBOOST_VALUES=current \
  -e LOWP_SECONDS=20 \
  -e LOWP_SHAPES='1024x8192x8192,2048x8192x8192,4096x8192x8192,8192x8192x8192' \
  -e OMP_NUM_THREADS=10 \
  -e MALLOC_ARENA_MAX=2 \
  -v /:/host:ro -v /dev:/dev -v /sys:/sys:ro -v /proc:/host_proc:ro \
  -v "$PWD/results:/results" \
  gb10-spark-perf-lab:ngc all
```

## Outputs

```text
bench/lowp/lowp_meta.json
bench/lowp/lowp_bench.json
bench/lowp/lowp_summary.tsv
bench/lowp/lowp_summary.md
bench/lowp/vboost-*/lowp_nvidia_smi_live.csv
bench/lowp/vboost-*/lowp_nvidia_smi_dmon.csv
```

The top-level `report.md` is augmented with the best low-precision results if `scripts/gb10-report-append.py` is installed.

## Interpretation

Questions this benchmark helps answer:

1. Does FP8/MXFP8/NVFP4 run at a different sustained clock than BF16/FP16 dense GEMM?
2. Does low precision reduce SW power-cap time or thermal slowdown?
3. Does vboost=3 remain best under workload-relevant low precision, or was that only true for dense BF16/FP16?
4. Does the chosen NGC container expose the actual Blackwell low-precision framework paths?
5. Is dynamic activation quantization much slower than static prequantized FP8 GEMM?

If all NVFP4/MXFP8 tests are skipped, the issue is usually framework/container exposure rather than the hardware. Try a newer NGC PyTorch or TensorRT-LLM container and compare `lowp_meta.json`.

# Max-performance GB10 tuning matrix

This document complements `gb10-lab tune-plan` and the generated `tunables/tunables.md` file. The goal is disciplined, reproducible A/B testing for an isolated GB10 performance host.

## Rules

1. Change one variable at a time.
2. Always capture `nvidia-smi -q -d CLOCK,POWER,PERFORMANCE,TEMPERATURE` before and after.
3. Always capture live telemetry during the workload.
4. Compare sustained throughput and p95/p99 latency, not only peak clocks.
5. Keep a rollback path for every boot-profile change.

## Highest-priority GB10 axes

### GPU policy

- `nvidia-smi boost-slider --vboost <0..max>`
- `nvidia-smi --cuda-clocks=1`, if accepted
- `nvidia-smi --lock-gpu-clocks=<min,max> --mode=0`, if accepted
- workload power profiles, if a future firmware exposes them
- power smoothing / power hint / PRM, if exposed by a future driver/firmware

### Thermal policy

- external intake airflow
- exhaust extraction and recirculation prevention
- lower ambient temperature
- avoid CPU-side heat when GPU-bound
- repeat A/B in a forward/reverse order to reduce heat-soak bias

### CPU/OS runtime

- performance cores only: `5-9,15-19`
- efficiency cores for housekeeping/BuildKit/IRQs: `0-4,10-14`
- `kernel.numa_balancing=0`
- `vm.swappiness=0` or disabled swap for golden runs
- Docker: `--ipc=host --ulimit memlock=-1 --shm-size=64g`

### Boot profile candidates

Start with low-CPU-heat GPU-throughput isolation before testing `idle=poll`:

```text
mitigations=off init_on_alloc=0 pci=realloc=off
irqaffinity=0-4,10-14 kthread_cpus=0-4,10-14
isolcpus=managed_irq,domain,5-9,15-19 nohz_full=5-9,15-19 rcu_nocbs=5-9,15-19
preempt=none psi=0 module_blacklist=nouveau
```

Then test hugepages separately:

```text
default_hugepagesz=1G hugepagesz=1G hugepages=8
```

Increase to `hugepages=16` or `24` only if model/runtime behavior benefits.

Reserve `idle=poll processor.max_cstate=0` for latency-only experiments because they can add CPU heat and reduce GPU thermal headroom.

## Workload precision matrix

Synthetic BF16/FP16 GEMM reveals throttling and dense Tensor Core behavior. For real GB10 LLM work, also benchmark:

- FP8 static GEMM
- FP8 dynamic activation + static weight
- Transformer Engine FP8 delayed scaling
- Transformer Engine MXFP8 block scaling
- Transformer Engine NVFP4 block scaling
- TensorRT-LLM FP8/FP4 model runs when an actual model is available
- vLLM FP8/MXFP8/NVFP4 configurations when supported

## What not to chase unless exposed

- `nvidia-smi -pl` when all power-limit fields are `N/A`
- laptop SMM fan hacks unless the platform exposes that control path
- CPU min=max if GPU-bound and thermally constrained
- forcing 3003 MHz if it increases SW power cap / thermal oscillation and lowers sustained throughput

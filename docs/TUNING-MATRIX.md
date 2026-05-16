# GB10 tuning matrix

This is the human A/B checklist that the container report is built around.

## Baseline

1. Official DGX/Spark update path and firmware current.
2. Newest installed `linux-image-*-nvidia` kernel is actually booted.
3. 240W PSU, direct connection, no suspect cable/dock/extension.
4. External cooling and stable ambient temperature.
5. NGC container current enough for Blackwell/GB10.
6. Benchmark actual workload and synthetic probes with live telemetry.

## Safe runtime knobs

- NVIDIA persistence mode enabled.
- CPU governor/EPP performance.
- NUMA balancing disabled.
- `init_on_alloc=0` on isolated performance boxes.
- `nv-mitigations-off` only on isolated, trusted systems.
- Docker: `--ipc=host`, `--ulimit memlock=-1`, large `--shm-size`.

## GPU exposed knobs

- `nvidia-smi --lock-gpu-clocks` / `--reset-gpu-clocks`.
- `nvidia-smi power-profiles` if GB10 exposes workload profiles.
- `nvidia-smi power-smoothing` query; do not set without A/B evidence.
- `nvidia-smi boost-slider` / vboost sweep; compare all advertised values before assuming it is irrelevant for AI compute.
- `nvidia-smi prm` Blackwell counters for diagnostics.

## Host memory/latency knobs

- THP `madvise` vs `always`.
- 1G hugepages in a separate boot entry.
- `idle=poll`, `nohz_full`, `rcu_nocbs`, `isolcpus`, `irqaffinity` only for latency-specific profiles.
- IRQ/NIC affinity and mlxconfig only when ConnectX traffic matters.

## Revert discipline

Every experimental knob must have a revert command and a benchmark delta. Keep logs.

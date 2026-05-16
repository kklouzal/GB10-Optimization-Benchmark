#!/usr/bin/env python3
import json, os, re, sys, statistics
from pathlib import Path
from datetime import datetime, timezone


def read(p: Path, max_bytes=2_000_000):
    try:
        data = p.read_bytes()[:max_bytes]
        return data.decode("utf-8", errors="replace")
    except Exception:
        return ""


def exists_text(root: Path, rel: str):
    p = root / rel
    return p.exists(), read(p)


def grep(text, pattern, flags=re.I|re.M):
    return re.findall(pattern, text, flags)


def parse_first(text, pattern, default=None, flags=re.I|re.M):
    m = re.search(pattern, text, flags)
    return m.group(1).strip() if m else default


def add_findings_from_gpu(root, findings, actions):
    _, q = exists_text(root, "gpu/nvidia_smi_q.txt")
    _, query = exists_text(root, "gpu/nvidia_smi_query_once.txt")
    _, bench_live = exists_text(root, "bench/nvidia_smi_live.csv")

    if "Persistence Mode" in q and re.search(r"Persistence Mode\s*:\s*Enabled", q):
        findings.append("GPU persistence mode is enabled.")
    elif "Persistence Mode" in q:
        findings.append("GPU persistence mode does not appear enabled.")
        actions.append("Enable persistence mode on the host: `sudo nvidia-smi -pm 1`.")

    pstate = parse_first(q, r"Performance State\s*:\s*(P\d+)") or parse_first(query, r",\s*(P\d+)\s*,")
    if pstate:
        findings.append(f"Observed GPU performance state: `{pstate}`.")
        if pstate != "P0":
            actions.append("Under a real sustained CUDA workload the GPU should normally enter P0. If it stays below P0, check workload occupancy, clocks, thermals, and persistence.")

    power_limit = parse_first(q, r"Current Power Limit\s*:\s*([^\n]+)")
    if power_limit:
        findings.append(f"Current GPU power-limit reporting: `{power_limit}`.")
        if "N/A" in power_limit:
            actions.append("The platform is not exposing a standard `nvidia-smi -pl` power budget. Focus on firmware/PD state, thermal headroom, workload occupancy, and supported clock/power-profile controls instead of assuming a settable power limit.")

    max_gfx = parse_first(q, r"Max Clocks\s*(?:\n.*)*?Graphics\s*:\s*([0-9]+) MHz")
    cur_gfx = parse_first(q, r"Clocks\s*(?:\n.*)*?Graphics\s*:\s*([0-9]+) MHz")
    app_gfx = parse_first(q, r"Applications Clocks\s*(?:\n.*)*?Graphics\s*:\s*([0-9]+) MHz")
    if cur_gfx or app_gfx or max_gfx:
        findings.append(f"Clock snapshot: current graphics={cur_gfx or 'unknown'} MHz, app graphics={app_gfx or 'unknown'} MHz, max graphics={max_gfx or 'unknown'} MHz.")
        actions.append("A/B test `nvidia-smi --lock-gpu-clocks=<min,max>` only if supported by your GB10 driver. Start with a reset baseline, then try a broad range such as `2400,3003`, and always compare sustained telemetry and throughput before keeping it.")

    if re.search(r"HW Power Brake\s*:\s*Active|hw_power_brake[^\n]*Active", q + query + bench_live, re.I):
        actions.append("HW Power Brake was observed. Check USB-C PD negotiation, the 240W adapter, cable seating, dock/extension cables, and do a full AC power-cycle before benchmarking again.")
    if re.search(r"HW Thermal Slowdown\s*:\s*Active|SW Thermal Slowdown\s*:\s*Active|thermal_slowdown[^\n]*Active", q + query + bench_live, re.I):
        actions.append("Thermal slowdown was observed. Improve intake/exhaust, ambient temperature, and sustained cooling before applying clock locks.")
    if re.search(r"SW Power Cap\s*:\s*Active|sw_power_cap[^\n]*Active", q + query + bench_live, re.I):
        actions.append("SW power cap was observed. If power-limit fields are N/A, look for workload power profiles, firmware updates, power smoothing state, and thermal/power-delivery constraints rather than `-pl`.")

    _, caps = exists_text(root, "gpu/nvidia_smi_capabilities.txt")
    if "power-profiles" in caps.lower() or "Workload Power Profiles" in q:
        actions.append("Inspect `gpu/nvidia_smi_capabilities.txt` for `nvidia-smi power-profiles -l/-ld`. If GB10 exposes a compute or maximum-performance profile, benchmark requested/enforced profile changes explicitly.")
    if "boost-slider" in caps.lower():
        actions.append("Inspect `nvidia-smi boost-slider -l`; this may be irrelevant for AI compute, but the tool records it because v580 exposes boost-slider/power-hint controls on some GPUs.")


def add_findings_from_host(root, findings, actions):
    _, platform = exists_text(root, "host/platform.txt")
    _, apt = exists_text(root, "apt/installed_versions.txt")
    _, kern = exists_text(root, "kernel/cmdline_config.txt")
    _, svc = exists_text(root, "services/dgx_nvidia.txt")
    _, cpu = exists_text(root, "cpu/cpufreq.txt")
    _, mem = exists_text(root, "mem/hugepages_thp.txt")
    _, fw = exists_text(root, "fw/fwupd.txt")

    dgx = parse_first(platform, r"DGX_OTA_VERSION=\"([^\"]+)\"") or parse_first(platform, r"dgx-release\s+([0-9.]+)")
    if dgx:
        findings.append(f"Detected DGX/Spark software version marker: `{dgx}`.")

    running_kernel = parse_first(platform, r"Linux\s+\S+\s+(\S+-nvidia)")
    if not running_kernel:
        running_kernel = parse_first(kern, r"/boot/vmlinuz-([^\s]+)")
    installed_kernels = sorted(set(re.findall(r"linux-image-([0-9][^\s]+-nvidia)\s+", apt)))
    if running_kernel:
        findings.append(f"Running kernel: `{running_kernel}`.")
    if installed_kernels:
        findings.append(f"Installed NVIDIA kernels include: `{', '.join(installed_kernels[-5:])}`.")
        # lexicographic is imperfect but catches 1014 vs 1018 class issues
        if running_kernel and running_kernel not in installed_kernels[-1:]:
            actions.append(f"Running kernel `{running_kernel}` may not be the newest installed NVIDIA kernel `{installed_kernels[-1]}`. Reboot/select the newest kernel before performance testing.")

    if "init_on_alloc=0" in kern:
        findings.append("Kernel command line includes `init_on_alloc=0`.")
    else:
        actions.append("For isolated performance systems, test `init_on_alloc=0`; NVIDIA DGX images often ship a package for this, but verify the boot command line.")
    if re.search(r"numa_balancing\s*==\s*\n?0|== numa balancing ==\s*\n0", kern, re.I) or "nvidia-disable-numa-balancing" in apt:
        findings.append("NUMA balancing appears disabled or the NVIDIA package to disable it is installed.")
    else:
        actions.append("Disable automatic NUMA balancing for deterministic latency: `kernel.numa_balancing=0` or NVIDIA's DGX package when applicable.")

    if re.search(r"scaling_governor:\s*performance", cpu) or re.search(r"\nperformance\n", cpu):
        findings.append("CPU governor appears to include `performance`.")
    else:
        actions.append("Set all CPU cpufreq policies to `performance`; verify energy_performance_preference is also `performance` where exposed.")

    if "nvidia-persistenced.service" in svc and re.search(r"nvidia-persistenced\.service.*enabled", svc):
        findings.append("nvidia-persistenced service appears enabled.")

    if "nv-mitigations-off" in apt or re.search(r"Mitigation:|Vulnerable|Not affected", kern):
        findings.append("CPU vulnerability/mitigation state was collected; review `kernel/cmdline_config.txt` for post-reboot mitigation status.")

    if "fwupdmgr get-updates" in fw or "No updatable devices" in fw or "Devices that have firmware updates" in fw:
        findings.append("Firmware inventory/update state was collected through fwupd when available.")

    if "[always]" in mem:
        findings.append("Transparent Huge Pages appear set to `always`.")
    elif "[madvise]" in mem:
        findings.append("Transparent Huge Pages appear set to `madvise`.")
    else:
        actions.append("A/B test Transparent Huge Pages (`madvise` vs `always`) and 1G hugepages for latency-sensitive host staging paths. Do not assume it improves GPU-only workloads.")


def add_bench_findings(root, findings, actions):
    p = root / "bench" / "torch_bench.json"
    if not p.exists():
        actions.append("No PyTorch benchmark JSON found. Run `gb10-lab bench` with an NGC PyTorch base image and `--gpus all`.")
        return
    try:
        data = json.loads(p.read_text())
    except Exception as e:
        actions.append(f"Could not parse torch benchmark JSON: {e!r}")
        return
    meta = data.get("meta", {})
    findings.append(f"PyTorch={meta.get('torch')} CUDA={meta.get('torch_cuda')} device={meta.get('device_name')} capability={meta.get('device_capability')}.")
    mat = [x for x in data.get("matmul", []) if x.get("median_TFLOP_s")]
    if mat:
        best = sorted(mat, key=lambda x: x.get("best_TFLOP_s") or 0)[-1]
        findings.append(f"Best observed matmul: {best.get('dtype')} n={best.get('n')} best={best.get('best_TFLOP_s'):.3f} TFLOP/s median={best.get('median_TFLOP_s'):.3f} TFLOP/s.")
        bf16 = [x for x in mat if x.get("dtype") == "bf16"]
        fp16 = [x for x in mat if x.get("dtype") == "fp16"]
        if bf16 and fp16:
            bmax=max(x.get("best_TFLOP_s") or 0 for x in bf16); fmax=max(x.get("best_TFLOP_s") or 0 for x in fp16)
            if bmax and fmax and min(bmax, fmax)/max(bmax, fmax) < 0.65:
                actions.append("BF16/FP16 performance differs materially. Verify dtype choices, Tensor Core enablement, model kernels, and whether FP4/FP8 paths are actually being used.")
    bw = [x for x in data.get("bandwidth", []) if x.get("best_GB_s")]
    if bw:
        for kind in sorted(set(x["kind"] for x in bw)):
            best = max((x.get("best_GB_s") or 0) for x in bw if x["kind"] == kind)
            findings.append(f"Best PyTorch {kind} bandwidth: {best:.3f} GB/s.")
    nv = read(root / "bench" / "nvbandwidth_json.txt", max_bytes=500_000)
    if nv and "nvbandwidth" not in nv.lower() and "error" not in nv.lower():
        findings.append("nvbandwidth output was captured for CUDA copy-path bandwidth analysis.")
    elif (root / "bench" / "nvbandwidth_json.txt").exists():
        actions.append("nvbandwidth ran but may have errored. Check `bench/nvbandwidth_json.txt`; it is the best next-level probe for memory-copy bandwidth.")


def write_report(root: Path):
    findings=[]; actions=[]
    add_findings_from_host(root, findings, actions)
    add_findings_from_gpu(root, findings, actions)
    add_bench_findings(root, findings, actions)

    conservative = [
        "Use DGX Dashboard / official DGX OS update path first; then verify kernel, driver, firmware, and `fwupdmgr` state.",
        "Keep NGC framework containers current and benchmark actual workload containers, not only synthetic GEMMs.",
        "Use `--ipc=host`, `--ulimit memlock=-1`, adequate `--shm-size`, and `NVIDIA_DRIVER_CAPABILITIES=all` for benchmark containers.",
        "Use live telemetry during real workloads; idle clocks are not enough to diagnose GB10 performance.",
    ]
    experimental = [
        "Clock-lock A/B: reset clocks, run baseline, try `nvidia-smi --lock-gpu-clocks=<min,max> --mode=0`, run identical workload, then `nvidia-smi --reset-gpu-clocks`. Keep only if sustained throughput improves without throttle reasons.",
        "Power-profile A/B: if `nvidia-smi power-profiles -l/-ld` exposes profiles, test requested/enforced compute profiles and compare telemetry.",
        "Hugepage A/B: test THP `madvise` vs `always`; for latency/RAN-style workloads, test 1G hugepages and Aerial-style kernel arguments in a separate boot entry.",
        "C-state/idle A/B: `idle=poll` can reduce latency but burns power and heat; use only for dedicated latency tests.",
        "IRQ/NIC A/B: if ConnectX traffic matters, test IRQ affinity, interrupt coalescing, relaxed ordering, and mlxconfig changes with before/after network benchmarks.",
        "Disable further security features only on isolated boxes and only when measured; do not mix security changes with clock/power changes in the same A/B run.",
    ]

    lines=[]
    lines.append("# GB10 Spark Perf Lab Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"Result directory: `{root}`")
    lines.append("")
    lines.append("## Key findings")
    lines.extend([f"- {x}" for x in findings] or ["- No findings generated; check collection completeness."])
    lines.append("")
    lines.append("## Action candidates")
    seen=set()
    for x in actions:
        if x not in seen:
            lines.append(f"- {x}"); seen.add(x)
    if not actions:
        lines.append("- No immediate corrective actions detected. Focus on workload-level profiling and A/B tuning.")
    lines.append("")
    lines.append("## Conservative performance baseline")
    lines.extend([f"- {x}" for x in conservative])
    lines.append("")
    lines.append("## Experimental / unsupported A/B inventory")
    lines.extend([f"- {x}" for x in experimental])
    lines.append("")
    lines.append("## Files to inspect first")
    for rel in [
        "host/platform.txt", "apt/installed_versions.txt", "kernel/cmdline_config.txt",
        "gpu/nvidia_smi_q.txt", "gpu/nvidia_smi_capabilities.txt", "bench/torch_bench.json",
        "bench/nvidia_smi_live.csv", "bench/nvbandwidth_json.txt", "fw/fwupd.txt",
        "logs/dmesg_power_thermal_pcie.txt", "logs/journal_warnings.txt",
    ]:
        if (root/rel).exists(): lines.append(f"- `{rel}`")
    report = "\n".join(lines) + "\n"
    (root / "report.md").write_text(report)
    print(report)


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("GB10_OUT", "/results/latest"))
    if not root.exists():
        print(f"Result directory does not exist: {root}", file=sys.stderr)
        sys.exit(2)
    write_report(root)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


def read(p: Path, max_bytes=2_000_000):
    try:
        data = p.read_bytes()[:max_bytes]
        return data.decode("utf-8", errors="replace")
    except Exception:
        return ""


def exists_text(root: Path, rel: str):
    p = root / rel
    return p.exists(), read(p)


def grep(text, pattern, flags=re.I | re.M):
    return re.findall(pattern, text, flags)


def parse_first(text, pattern, default=None, flags=re.I | re.M):
    m = re.search(pattern, text, flags)
    return m.group(1).strip() if m else default


def parse_last(text, pattern, default=None, flags=re.I | re.M):
    matches = re.findall(pattern, text, flags)
    if not matches:
        return default
    last = matches[-1]
    return last.strip() if isinstance(last, str) else last[0].strip()


def extract_json_object(text):
    match = re.search(r'(?ms)^\{.*^\}\s*(?=^### exit_status=|\Z)', text)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    start = text.find('{')
    end = text.rfind('}')
    if start < 0 or end < 0 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except Exception:
        return None


def fmt_gbs(value):
    try:
        return f"{float(value):.1f}"
    except Exception:
        return None


def summarize_nvbandwidth(text):
    parsed = extract_json_object(text)
    if not isinstance(parsed, dict):
        return None
    root = parsed.get('nvbandwidth') or {}
    tests = root.get('testcases') or []
    by_name = {t.get('name'): t for t in tests if isinstance(t, dict) and t.get('name')}

    def passed_values(*names):
        vals = []
        for name in names:
            row = by_name.get(name)
            if row and row.get('status') == 'Passed' and row.get('sum') is not None:
                vals.append(float(row.get('sum')))
        return vals

    ce_h2d = passed_values('host_to_device_memcpy_ce')
    ce_d2h = passed_values('device_to_host_memcpy_ce')
    sm_h2d = passed_values('host_to_device_memcpy_sm', 'host_to_all_memcpy_sm')
    sm_d2h = passed_values('device_to_host_memcpy_sm', 'all_to_host_memcpy_sm')
    sm_bidir = passed_values('host_to_all_bidirectional_memcpy_sm', 'all_to_host_bidirectional_memcpy_sm', 'host_to_device_bidirectional_memcpy_sm', 'device_to_host_bidirectional_memcpy_sm')
    local_copy = passed_values('device_local_copy')
    exit_status = parse_first(text, r'### exit_status=(\d+)', default=None)
    return {
        'exit_status': int(exit_status) if exit_status is not None else None,
        'ce_h2d': max(ce_h2d) if ce_h2d else None,
        'ce_d2h': max(ce_d2h) if ce_d2h else None,
        'sm_h2d': max(sm_h2d) if sm_h2d else None,
        'sm_d2h': max(sm_d2h) if sm_d2h else None,
        'sm_bidir_min': min(sm_bidir) if sm_bidir else None,
        'sm_bidir_max': max(sm_bidir) if sm_bidir else None,
        'device_local_copy': max(local_copy) if local_copy else None,
    }


def classify_lowp_issue(row):
    error_text = str(row.get('error') or row.get('reason') or '')
    suite = str(row.get('suite') or '')
    if suite.startswith('torch_scaled_mm_fp8') and 'Invalid scaling configuration' in error_text:
        return 'PyTorch FP8 failed: invalid scale dtype/configuration'
    if suite == 'te_mxfp8_block_e4m3' and 'not supported on 12.0+ architectures yet' in error_text:
        return 'TE MXFP8 failed: architecture support message'
    if suite == 'te_nvfp4_block' and 'invalid argument' in error_text.lower():
        return 'TE NVFP4 failed: CUDA invalid argument'
    if row.get('error'):
        return f"{suite or 'unknown'} failed"
    if row.get('skipped'):
        return f"{suite or 'unknown'} skipped"
    return None


def summarize_lowp_failures(records):
    counts = {}
    error_count = 0
    skipped_count = 0
    for row in records:
        if row.get('error'):
            error_count += 1
        elif row.get('skipped'):
            skipped_count += 1
        else:
            continue
        label = classify_lowp_issue(row) or 'Other low-precision failure/skip'
        counts[label] = counts.get(label, 0) + 1
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return ordered, error_count, skipped_count


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
        actions.append("VBoost is a first-class benchmark dimension in this lab now: compare the built-in sweep results before treating boost-slider as irrelevant for AI compute.")


def add_findings_from_host(root, findings, actions):
    _, platform = exists_text(root, "host/platform.txt")
    _, apt = exists_text(root, "apt/installed_versions.txt")
    _, kern = exists_text(root, "kernel/cmdline_config.txt")
    _, svc = exists_text(root, "services/dgx_nvidia.txt")
    _, cpu = exists_text(root, "cpu/cpufreq.txt")
    _, mem = exists_text(root, "mem/hugepages_thp.txt")
    _, fw = exists_text(root, "fw/fwupd.txt")

    dgx = parse_last(platform, r'DGX_OTA_VERSION="([^"]+)"') or parse_first(platform, r"dgx-release\s+([0-9.]+)")
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


def flatten_bench_rows(data, key):
    rows = []
    runs = data.get("runs") or []
    if runs:
        for run in runs:
            for row in run.get(key, []) or []:
                merged = dict(row)
                if run.get("vboost") is not None:
                    merged.setdefault("vboost", run.get("vboost"))
                rows.append(merged)
        return rows
    return [dict(row) for row in data.get(key, [])]


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

    vboost = data.get("vboost") or {}
    planned_values = vboost.get("planned_values") or []
    if planned_values:
        findings.append(f"Built-in vboost sweep planned values: `{', '.join(str(v) for v in planned_values)}`.")
    completed_runs = [run for run in (data.get("runs") or []) if run.get("status") == "ok"]
    if completed_runs:
        findings.append(f"Completed vboost runs: `{', '.join(str(run.get('vboost')) for run in completed_runs)}`.")
    restore = vboost.get("restore") or {}
    if restore and restore.get("ok") is False:
        actions.append("VBoost restore failed at the end of benchmarking. Inspect `bench/vboost_restore.txt` and `bench/vboost_final.json` before trusting later results.")

    mat = [x for x in flatten_bench_rows(data, "matmul") if x.get("median_TFLOP_s")]
    if mat:
        best = sorted(mat, key=lambda x: x.get("best_TFLOP_s") or 0)[-1]
        vboost_label = f" vboost={best.get('vboost')}" if best.get("vboost") is not None else ""
        findings.append(
            f"Best observed matmul{vboost_label}: {best.get('dtype')} n={best.get('n')} best={best.get('best_TFLOP_s'):.3f} TFLOP/s median={best.get('median_TFLOP_s'):.3f} TFLOP/s."
        )

        preferred = [x for x in mat if x.get("dtype") in {"bf16", "fp16"} and x.get("vboost") is not None]
        if preferred:
            per_vboost = {}
            for row in preferred:
                score = row.get("median_TFLOP_s") or 0
                per_vboost[row["vboost"]] = max(score, per_vboost.get(row["vboost"], 0))
            ordered = sorted(per_vboost.items())
            findings.append(
                "Best BF16/FP16 median TFLOP/s by vboost: "
                + ", ".join(f"{value}={score:.3f}" for value, score in ordered)
                + "."
            )
            best_vboost, best_score = max(ordered, key=lambda item: item[1])
            baseline = per_vboost.get(0)
            if baseline and best_vboost != 0:
                delta_pct = ((best_score - baseline) / baseline) * 100 if baseline else 0.0
                findings.append(f"Best BF16/FP16 median vboost was `{best_vboost}` ({delta_pct:+.1f}% vs vboost 0).")
                actions.append("Re-run the real workload at the winning vboost and compare sustained clocks, SW power-cap time, and slowdown reasons before keeping it.")
            elif baseline:
                findings.append("Vboost 0 remained the best BF16/FP16 median matmul setting in this sweep.")

        bf16 = [x for x in mat if x.get("dtype") == "bf16"]
        fp16 = [x for x in mat if x.get("dtype") == "fp16"]
        if bf16 and fp16:
            bmax = max(x.get("best_TFLOP_s") or 0 for x in bf16)
            fmax = max(x.get("best_TFLOP_s") or 0 for x in fp16)
            if bmax and fmax and min(bmax, fmax) / max(bmax, fmax) < 0.65:
                actions.append("BF16/FP16 performance differs materially. Verify dtype choices, Tensor Core enablement, model kernels, and whether FP4/FP8 paths are actually being used.")

    bw = [x for x in flatten_bench_rows(data, "bandwidth") if x.get("best_GB_s")]
    if bw:
        for kind in sorted(set(x["kind"] for x in bw)):
            best = max((x for x in bw if x["kind"] == kind), key=lambda x: x.get("best_GB_s") or 0)
            vboost_label = f" at vboost={best.get('vboost')}" if best.get("vboost") is not None else ""
            findings.append(f"Best PyTorch {kind} bandwidth{vboost_label}: {best.get('best_GB_s'):.3f} GB/s.")

    nv = read(root / "bench" / "nvbandwidth_json.txt", max_bytes=500_000)
    nv_summary = summarize_nvbandwidth(nv) if nv else None
    if nv_summary and (nv_summary.get("exit_status") in (None, 0)):
        parts = []
        if nv_summary.get("ce_h2d") is not None:
            parts.append(f"CE H2D ~{fmt_gbs(nv_summary['ce_h2d'])} GB/s")
        if nv_summary.get("ce_d2h") is not None:
            parts.append(f"CE D2H ~{fmt_gbs(nv_summary['ce_d2h'])} GB/s")
        if nv_summary.get("sm_h2d") is not None:
            parts.append(f"SM H2D ~{fmt_gbs(nv_summary['sm_h2d'])} GB/s")
        if nv_summary.get("sm_d2h") is not None:
            parts.append(f"SM D2H ~{fmt_gbs(nv_summary['sm_d2h'])} GB/s")
        if nv_summary.get("sm_bidir_min") is not None:
            if math.isclose(nv_summary['sm_bidir_min'], nv_summary['sm_bidir_max'], rel_tol=1e-6):
                parts.append(f"bidirectional SM ~{fmt_gbs(nv_summary['sm_bidir_max'])} GB/s")
            else:
                parts.append(f"bidirectional SM ~{fmt_gbs(nv_summary['sm_bidir_min'])}–{fmt_gbs(nv_summary['sm_bidir_max'])} GB/s")
        if nv_summary.get("device_local_copy") is not None:
            parts.append(f"device local copy ~{fmt_gbs(nv_summary['device_local_copy'])} GB/s")
        if parts:
            findings.append("nvbandwidth passed: " + "; ".join(parts) + ".")
        else:
            findings.append("nvbandwidth completed successfully; inspect `bench/nvbandwidth_json.txt` for detailed copy-path results.")
    elif (root / "bench" / "nvbandwidth_json.txt").exists():
        actions.append("nvbandwidth did not yield a clean parsed success result. Check `bench/nvbandwidth_json.txt`; it is the best next-level probe for memory-copy bandwidth.")

    lowp_path = root / "bench" / "lowp" / "lowp_bench.json"
    if lowp_path.exists():
        try:
            lowp = json.loads(lowp_path.read_text())
        except Exception as e:
            actions.append(f"Could not parse low-precision benchmark JSON: {e!r}")
        else:
            records = lowp.get('records') or []
            scored = [r for r in records if r.get('median_TFLOP_s_dense_equiv') is not None]
            failure_categories, error_count, skipped_count = summarize_lowp_failures(records)
            failed_or_error = error_count + skipped_count
            findings.append(f"Low-precision results: {len(scored)} scored record(s), {failed_or_error} failed/error record(s).")
            for label, count in failure_categories[:5]:
                findings.append(f"Low-precision issue: {count}x {label}.")


def write_report(root: Path):
    findings = []
    actions = []
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
        "Power-profile A/B: if `nvidia-smi power-profiles -l/-ld` exposes profiles, test requested/enforced profile changes and compare telemetry.",
        "Vboost A/B: treat the built-in 0..max sweep as required baseline evidence, then confirm the best value with the real workload before keeping it.",
        "Hugepage A/B: test THP `madvise` vs `always`; for latency/RAN-style workloads, test 1G hugepages and Aerial-style kernel arguments in a separate boot entry.",
        "C-state/idle A/B: `idle=poll` can reduce latency but burns power and heat; use only for dedicated latency tests.",
        "IRQ/NIC A/B: if ConnectX traffic matters, test IRQ affinity, interrupt coalescing, relaxed ordering, and mlxconfig changes with before/after network benchmarks.",
        "Disable further security features only on isolated boxes and only when measured; do not mix security changes with clock/power changes in the same A/B run.",
    ]

    lines = []
    lines.append("# GB10 Spark Perf Lab Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"Result directory: `{root}`")
    lines.append("")
    lines.append("## Key findings")
    lines.extend([f"- {x}" for x in findings] or ["- No findings generated; check collection completeness."])
    lines.append("")
    lines.append("## Action candidates")
    seen = set()
    for x in actions:
        if x not in seen:
            lines.append(f"- {x}")
            seen.add(x)
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
    inspect_paths = [
        "host/platform.txt",
        "apt/installed_versions.txt",
        "kernel/cmdline_config.txt",
        "gpu/nvidia_smi_q.txt",
        "gpu/nvidia_smi_capabilities.txt",
        "bench/torch_bench.json",
        "bench/nvidia_smi_live.csv",
        "bench/nvbandwidth_json.txt",
        "fw/fwupd.txt",
        "logs/dmesg_power_thermal_pcie.txt",
        "logs/journal_warnings.txt",
    ]
    for rel in inspect_paths:
        if (root / rel).exists():
            lines.append(f"- `{rel}`")
    if (root / "bench" / "vboost_summary.md").exists():
        lines.append("- `bench/vboost_summary.md`")
    if (root / "bench" / "vboost_summary.json").exists():
        lines.append("- `bench/vboost_summary.json`")
    if list((root / "bench").glob("vboost-*/torch_bench.json")):
        lines.append("- `bench/vboost-*/torch_bench.json`")
    if list((root / "bench").glob("vboost-*/nvidia_smi_live.csv")):
        lines.append("- `bench/vboost-*/nvidia_smi_live.csv`")
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

#!/usr/bin/env python3
"""Generate a GB10/DGX Spark tunability matrix.

This script is read-only. It catalogs performance-related controls that are
visible from the current container/host namespace and emits JSON + Markdown so
runs can be compared across firmware, kernels, boot profiles, and container
launch shapes.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def run(args: List[str], timeout: int = 30) -> Dict[str, Any]:
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return {"ok": p.returncode == 0, "returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr, "args": args}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "returncode": None, "stdout": "", "stderr": repr(e), "args": args}


def sh(cmd: str, timeout: int = 30) -> Dict[str, Any]:
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return {"ok": p.returncode == 0, "returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr, "cmd": cmd}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "returncode": None, "stdout": "", "stderr": repr(e), "cmd": cmd}


def read_text(path: str, limit: int = 1_000_000) -> str:
    try:
        return Path(path).read_bytes()[:limit].decode("utf-8", errors="replace")
    except Exception:
        return ""


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))


def parse_key_value_file(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    try:
        for line in path.read_text(errors="replace").splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip().strip('"')
    except Exception:
        pass
    return out


def first_existing(paths: Iterable[str]) -> Optional[str]:
    for p in paths:
        if Path(p).exists():
            return p
    return None


def collect_nvidia_smi() -> Dict[str, Any]:
    cmds = {
        "nvidia_smi": ["nvidia-smi"],
        "q_clock_power_perf_temp": ["nvidia-smi", "-q", "-d", "CLOCK,POWER,PERFORMANCE,TEMPERATURE"],
        "q_supported_clocks": ["nvidia-smi", "-q", "-d", "SUPPORTED_CLOCKS"],
        "boost_slider": ["nvidia-smi", "boost-slider", "-l"],
        "power_profiles_l": ["nvidia-smi", "power-profiles", "-l"],
        "power_profiles_ld": ["nvidia-smi", "power-profiles", "-ld"],
        "power_profiles_gr": ["nvidia-smi", "power-profiles", "-gr"],
        "power_profiles_ge": ["nvidia-smi", "power-profiles", "-ge"],
        "power_smoothing_q": ["nvidia-smi", "power-smoothing", "-q"],
        "power_smoothing_ppd": ["nvidia-smi", "power-smoothing", "-ppd"],
        "power_hint_l": ["nvidia-smi", "power-hint", "-l"],
        "prm_l": ["nvidia-smi", "prm", "-l"],
        "c2c_status": ["nvidia-smi", "c2c", "-s"],
        "nvlink_status": ["nvidia-smi", "nvlink", "--status"],
        "help": ["nvidia-smi", "--help"],
        "help_query_gpu": ["nvidia-smi", "--help-query-gpu"],
    }
    out: Dict[str, Any] = {}
    if not shutil.which("nvidia-smi"):
        return {"available": False, "reason": "nvidia-smi not found"}
    out["available"] = True
    for name, args in cmds.items():
        out[name] = run(args, timeout=45)
    text = "\n".join((v.get("stdout", "") + v.get("stderr", "")) for v in out.values() if isinstance(v, dict))
    out["capability_flags"] = {
        "boost_slider": "boost-slider" in text,
        "lock_gpu_clocks_help": "lock-gpu-clocks" in text or "-lgc" in text,
        "cuda_clocks_help": "cuda-clocks" in text,
        "power_limit_help": "--power-limit" in text or "-pl" in text,
        "power_profiles": "power-profiles" in text,
        "power_smoothing": "power-smoothing" in text,
        "power_hint": "power-hint" in text,
        "supported_clocks_listed": "Supported Clocks" in text and "N/A" not in (out.get("q_supported_clocks", {}).get("stdout") or ""),
    }
    return out


def collect_sysfs() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out["cmdline"] = read_text("/proc/cmdline")
    out["sysctl"] = {}
    for path in [
        "/proc/sys/kernel/numa_balancing",
        "/proc/sys/kernel/sched_autogroup_enabled",
        "/proc/sys/kernel/timer_migration",
        "/proc/sys/kernel/perf_event_paranoid",
        "/proc/sys/kernel/sched_rt_runtime_us",
        "/proc/sys/kernel/watchdog",
        "/proc/sys/vm/swappiness",
        "/proc/sys/vm/zone_reclaim_mode",
        "/proc/sys/vm/overcommit_memory",
        "/proc/sys/vm/max_map_count",
        "/proc/sys/vm/dirty_background_ratio",
        "/proc/sys/vm/dirty_ratio",
    ]:
        p = Path(path)
        if p.exists():
            out["sysctl"][path] = {"value": read_text(path).strip(), "writable": os.access(path, os.W_OK)}

    out["thp"] = {}
    for path in [
        "/sys/kernel/mm/transparent_hugepage/enabled",
        "/sys/kernel/mm/transparent_hugepage/defrag",
        "/sys/kernel/mm/transparent_hugepage/khugepaged/defrag",
        "/sys/kernel/mm/transparent_hugepage/khugepaged/max_ptes_none",
    ]:
        p = Path(path)
        if p.exists():
            out["thp"][path] = {"value": read_text(path).strip(), "writable": os.access(path, os.W_OK)}
    out["hugepages"] = read_text("/proc/meminfo")

    out["cpufreq"] = []
    for pol in sorted(Path("/sys/devices/system/cpu/cpufreq").glob("policy*")):
        item: Dict[str, Any] = {"policy": str(pol)}
        for name in [
            "scaling_driver",
            "scaling_governor",
            "scaling_available_governors",
            "scaling_cur_freq",
            "scaling_min_freq",
            "scaling_max_freq",
            "cpuinfo_min_freq",
            "cpuinfo_max_freq",
            "energy_performance_preference",
            "energy_performance_available_preferences",
            "related_cpus",
            "affected_cpus",
        ]:
            p = pol / name
            if p.exists():
                item[name] = {"value": read_text(str(p)).strip(), "writable": os.access(p, os.W_OK)}
        out["cpufreq"].append(item)

    out["cpuidle"] = []
    for state in sorted(Path("/sys/devices/system/cpu").glob("cpu*/cpuidle/state*")):
        item = {"state": str(state)}
        for name in ["name", "desc", "latency", "power", "usage", "time", "disable"]:
            p = state / name
            if p.exists():
                item[name] = {"value": read_text(str(p)).strip(), "writable": os.access(p, os.W_OK)}
        out["cpuidle"].append(item)

    out["hwmon_fan_pwm"] = []
    for h in sorted(Path("/sys/class/hwmon").glob("hwmon*")):
        item: Dict[str, Any] = {"hwmon": str(h), "name": read_text(str(h / "name")).strip()}
        files = []
        for pattern in ["fan*", "pwm*", "temp*_label", "temp*_input", "power*_input"]:
            for p in sorted(h.glob(pattern)):
                files.append({"path": str(p), "value": read_text(str(p)).strip(), "writable": os.access(p, os.W_OK)})
        if files:
            item["files"] = files
            out["hwmon_fan_pwm"].append(item)

    out["pci_power"] = []
    for dev in sorted(Path("/sys/bus/pci/devices").glob("*")):
        item: Dict[str, Any] = {"device": str(dev), "vendor": read_text(str(dev / "vendor")).strip(), "class": read_text(str(dev / "class")).strip()}
        for name in ["current_link_speed", "current_link_width", "max_link_speed", "max_link_width", "numa_node", "power/control", "power/runtime_status"]:
            p = dev / name
            if p.exists():
                item[name] = {"value": read_text(str(p)).strip(), "writable": os.access(p, os.W_OK)}
        out["pci_power"].append(item)
    return out


def collect_tools() -> Dict[str, Any]:
    commands = [
        "nvidia-smi",
        "nvidia-ctk",
        "nvidia-container-cli",
        "dcgmi",
        "nvbandwidth",
        "nsys",
        "ncu",
        "perf",
        "bpftrace",
        "trace-cmd",
        "fio",
        "mlxconfig",
        "mlxlink",
        "mst",
        "ofed_info",
        "powerprofilesctl",
        "sensors",
        "docker",
    ]
    out: Dict[str, Any] = {}
    for cmd in commands:
        path = shutil.which(cmd)
        info: Dict[str, Any] = {"path": path}
        if path:
            info["version"] = sh(f"{cmd} --version 2>&1 | head -n 8", timeout=15)
        out[cmd] = info
    return out


def collect_container_context() -> Dict[str, Any]:
    return {
        "env_subset": {k: os.environ.get(k) for k in sorted(os.environ) if k.startswith(("GB10_", "RUN_", "LOWP_", "NVIDIA_", "CUDA_", "PYTORCH_", "OMP_", "MALLOC_", "VLLM_", "TRT"))},
        "limits": read_text("/proc/self/limits"),
        "cgroup": read_text("/proc/self/cgroup"),
        "status": read_text("/proc/self/status"),
        "mounts_head": "\n".join(read_text("/proc/mounts").splitlines()[:120]),
    }


def analyze(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    nvsmi_text = json.dumps(data.get("nvidia_smi", {}))
    sysfs = data.get("sysfs", {})
    candidates: List[Dict[str, Any]] = []

    def add(category: str, name: str, state: str, priority: str, evidence: str, command: str = "", risk: str = "") -> None:
        candidates.append({"category": category, "name": name, "state": state, "priority": priority, "evidence": evidence, "command": command, "risk": risk})

    flags = data.get("nvidia_smi", {}).get("capability_flags", {})
    if flags.get("boost_slider"):
        add("gpu", "vboost", "available", "high", "nvidia-smi boost-slider appears available", "sudo nvidia-smi boost-slider --vboost <0..max>", "May increase cap/thermal oscillation; A/B only")
    else:
        add("gpu", "vboost", "absent", "info", "boost-slider not found in nvidia-smi output")
    if flags.get("lock_gpu_clocks_help"):
        add("gpu", "GPU clock lock", "possibly_available", "high", "nvidia-smi help exposes --lock-gpu-clocks", "sudo nvidia-smi --lock-gpu-clocks=<min,max> --mode=0", "Can reduce sustained performance if power/thermal cap oscillates")
    if "Current Power Limit" in nvsmi_text and "N/A" in nvsmi_text:
        add("gpu", "nvidia-smi -pl", "not_exposed", "low", "Power limit fields appear N/A", "", "Do not chase unsupported -pl path")
    elif flags.get("power_limit_help"):
        add("gpu", "power limit", "maybe_available", "medium", "nvidia-smi help exposes --power-limit", "sudo nvidia-smi -pl <watts>", "Only valid if min/max range is reported")
    if flags.get("power_profiles"):
        add("gpu", "workload power profile", "probe", "medium", "power-profiles command present", "nvidia-smi power-profiles -l -ld -gr -ge", "May be unsupported on GB10 despite help text")
    if flags.get("cuda_clocks_help"):
        add("gpu", "CUDA clocks override", "probe", "medium", "nvidia-smi help mentions cuda-clocks", "sudo nvidia-smi --cuda-clocks=1", "Experimental; compare telemetry")

    cmdline = sysfs.get("cmdline", "")
    if "mitigations=off" in cmdline:
        add("kernel", "CPU mitigations", "already_off", "info", "cmdline includes mitigations=off")
    else:
        add("kernel", "CPU mitigations", "candidate", "medium", "cmdline does not include mitigations=off", "install/use nv-mitigations-off or add mitigations=off", "Security tradeoff")
    if "init_on_alloc=0" in cmdline:
        add("kernel", "init_on_alloc", "already_zero", "info", "cmdline includes init_on_alloc=0")
    else:
        add("kernel", "init_on_alloc", "candidate", "medium", "cmdline does not include init_on_alloc=0", "add init_on_alloc=0", "Security hardening tradeoff")

    for path, item in sysfs.get("sysctl", {}).items():
        val = item.get("value", "")
        if path.endswith("numa_balancing") and val != "0":
            add("sysctl", "NUMA balancing", "candidate", "high", f"{path}={val}", "sudo sysctl kernel.numa_balancing=0")
        if path.endswith("swappiness") and val not in {"0", "1"}:
            add("sysctl", "swappiness", "candidate", "medium", f"{path}={val}", "sudo sysctl vm.swappiness=0")
        if path.endswith("sched_autogroup_enabled") and val != "0":
            add("sysctl", "sched autogroup", "candidate", "medium", f"{path}={val}", "sudo sysctl kernel.sched_autogroup_enabled=0")

    if any((x.get("name") or "").lower().find("fan") >= 0 or any("pwm" in f.get("path", "") for f in x.get("files", [])) for x in sysfs.get("hwmon_fan_pwm", [])):
        add("thermal", "fan/PWM sysfs", "probe", "high", "fan/pwm hwmon entries exist", "inspect tunables.json hwmon_fan_pwm; never write values without confirming semantics", "Wrong PWM writes can destabilize cooling")
    else:
        add("thermal", "fan/PWM sysfs", "absent", "info", "no hwmon fan/pwm entries found")

    if sysfs.get("cpuidle"):
        add("latency", "CPU idle states", "available", "medium", "cpuidle state disable files detected", "echo 1 | sudo tee /sys/devices/system/cpu/cpu*/cpuidle/state[1-9]/disable", "Raises CPU heat; may hurt GPU-bound thermal headroom")
    if sysfs.get("thp"):
        add("memory", "THP", "available", "medium", "transparent hugepage sysfs entries found", "echo always|madvise | sudo tee /sys/kernel/mm/transparent_hugepage/enabled", "A/B only")
    add("boot", "1G hugepages / nohz / IRQ isolation", "candidate", "medium", "requires separate GRUB profile", "default_hugepagesz=1G hugepagesz=1G hugepages=<N> nohz_full=<cores> rcu_nocbs=<cores> irqaffinity=<housekeeping>", "Can reduce usable memory and complicate recovery")
    add("container", "LLM launch hygiene", "candidate", "high", "always A/B with real workload", "--gpus all --ipc=host --cpuset-cpus=5-9,15-19 --ulimit memlock=-1 --shm-size=64g", "None if isolated")
    return candidates


def markdown(data: Dict[str, Any], candidates: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append("# GB10 tunability matrix")
    lines.append("")
    lines.append(f"Generated UTC: {data.get('generated_utc')}")
    lines.append("")
    lines.append("## High-priority candidates")
    high = [c for c in candidates if c.get("priority") == "high"]
    if high:
        for c in high:
            lines.append(f"- **{c['category']} / {c['name']}**: `{c['state']}` — {c['evidence']}")
            if c.get("command"):
                lines.append(f"  - Command/probe: `{c['command']}`")
            if c.get("risk"):
                lines.append(f"  - Risk: {c['risk']}")
    else:
        lines.append("- None detected")
    lines.append("")
    lines.append("## Full candidate inventory")
    for c in candidates:
        lines.append(f"- `{c['priority']}` **{c['category']} / {c['name']}**: state=`{c['state']}`; evidence={c['evidence']}")
    lines.append("")
    lines.append("## Useful files")
    lines.append("- `tunables.json` — full raw probe output")
    lines.append("- `tunables.md` — this report")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/results/tunables")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data: Dict[str, Any] = {
        "tool": "gb10-tunables",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "nvidia_smi": collect_nvidia_smi(),
        "sysfs": collect_sysfs(),
        "tools": collect_tools(),
        "container": collect_container_context(),
    }
    candidates = analyze(data)
    data["candidates"] = candidates
    write_json(out / "tunables.json", data)
    (out / "tunables.md").write_text(markdown(data, candidates))
    print(json.dumps({"wrote": str(out), "candidate_count": len(candidates)}, indent=2))


if __name__ == "__main__":
    main()

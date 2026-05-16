#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path


SAFE_NVIDIA_SMI_FIELDS = [
    "timestamp",
    "pstate",
    "temperature.gpu",
    "power.draw",
    "utilization.gpu",
    "utilization.memory",
    "clocks.current.graphics",
    "clocks.current.sm",
    "clocks.max.graphics",
    "clocks_throttle_reasons.active",
    "clocks_throttle_reasons.sw_power_cap",
    "clocks_throttle_reasons.hw_power_brake",
    "clocks_throttle_reasons.hw_slowdown",
    "clocks_throttle_reasons.sw_thermal_slowdown",
    "clocks_throttle_reasons.hw_thermal_slowdown",
]
DEFAULT_NVIDIA_SMI_FIELDS = SAFE_NVIDIA_SMI_FIELDS


def sh(cmd, timeout=20):
    try:
        return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT, timeout=timeout)
    except Exception as e:
        return f"ERROR: {e!r}"


def run_cmd(args, timeout=20):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return {
            "ok": p.returncode == 0,
            "stdout": p.stdout,
            "stderr": p.stderr,
            "returncode": p.returncode,
            "args": list(args),
        }
    except Exception as e:
        return {
            "ok": False,
            "stdout": "",
            "stderr": repr(e),
            "returncode": None,
            "args": list(args),
        }


def command_output(res):
    stdout = res.get("stdout") or ""
    stderr = res.get("stderr") or ""
    if stdout and stderr:
        return f"{stdout}\n{stderr}".strip()
    return (stdout or stderr).strip()


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def probe_nvsmi_fields(candidates=None):
    requested = list(candidates or SAFE_NVIDIA_SMI_FIELDS)
    supported = []
    unsupported = []
    for field in requested:
        res = run_cmd(["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader,nounits"], timeout=5)
        if res["ok"]:
            supported.append(field)
        else:
            unsupported.append({"field": field, "error": command_output(res)})
    return {"requested": requested, "supported": supported, "unsupported": unsupported}


def nvsmi_query(fields):
    if not fields:
        return {"fields": [], "row": {}, "error": "no supported nvidia-smi query fields"}
    res = run_cmd(["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"], timeout=5)
    if not res["ok"]:
        return {"fields": list(fields), "row": {}, "error": command_output(res)}
    line = next((ln for ln in res["stdout"].splitlines() if ln.strip()), "").strip()
    values = [v.strip() for v in line.split(",")]
    if len(values) != len(fields):
        return {"fields": list(fields), "row": {}, "error": f"field/value mismatch: expected {len(fields)} values got {len(values)} from {line!r}"}
    return {"fields": list(fields), "row": dict(zip(fields, values)), "error": None}


class Telemetry:
    def __init__(self, path: Path, interval: float = 0.5, query_info=None, dmon_path: Path | None = None):
        self.path = path
        self.interval = interval
        self.query_info = query_info or probe_nvsmi_fields()
        self.fields = list(self.query_info.get("supported", []))
        self.dmon_path = dmon_path
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.dmon_proc = None

    def __enter__(self):
        if self.dmon_path and shutil.which("nvidia-smi"):
            try:
                self.dmon_proc = subprocess.Popen([
                    "nvidia-smi", "dmon", "-s", os.environ.get("TELEMETRY_DMON_SETS", "pucvmt"),
                    "-d", os.environ.get("TELEMETRY_DMON_INTERVAL", "1"),
                    "-o", "DT", "-f", str(self.dmon_path),
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                self.dmon_proc = None
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=5)
        if self.dmon_proc is not None:
            self.dmon_proc.terminate()
            try:
                self.dmon_proc.wait(timeout=5)
            except Exception:
                self.dmon_proc.kill()

    def _run(self):
        with self.path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["unix_time", *self.fields, "error"])
            while not self.stop.is_set():
                sample = nvsmi_query(self.fields)
                row = [time.time(), *[sample.get("row", {}).get(field, "") for field in self.fields], sample.get("error") or ""]
                w.writerow(row)
                f.flush()
                self.stop.wait(self.interval)


VBOOST_MAX_RE = re.compile(r"vboost\s+max\s+value\s*:\s*(\d+)", re.I)
VBOOST_CURRENT_RE = re.compile(r"current\s+value\s*:\s*(\d+)", re.I)
VBOOST_TABLE_RE = re.compile(r"\|\s*\d+\s+vboost\s+(\d+)\s+(\d+)\s*\|", re.I)


def query_vboost_state():
    res = run_cmd(["nvidia-smi", "boost-slider", "-l"], timeout=10)
    raw = command_output(res)
    info = {
        "query": res,
        "raw": raw,
        "available": res["ok"],
        "max_value": None,
        "current_value": None,
        "supported_values": [],
        "error": None,
    }
    if not res["ok"]:
        info["error"] = raw or "boost-slider query failed"
        return info
    max_match = VBOOST_MAX_RE.search(raw)
    cur_match = VBOOST_CURRENT_RE.search(raw)
    table_match = VBOOST_TABLE_RE.search(raw)
    if max_match:
        info["max_value"] = int(max_match.group(1))
    if cur_match:
        info["current_value"] = int(cur_match.group(1))
    if table_match:
        info["max_value"] = info["max_value"] if info["max_value"] is not None else int(table_match.group(1))
        info["current_value"] = info["current_value"] if info["current_value"] is not None else int(table_match.group(2))
    if info["max_value"] is not None:
        info["supported_values"] = list(range(info["max_value"] + 1))
    if info["max_value"] is None and info["current_value"] is None:
        info["error"] = "unable to parse boost-slider output"
    return info


def parse_vboost_values(raw: str):
    values = []
    for part in (raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    unique = []
    seen = set()
    for value in values:
        if value not in seen:
            unique.append(value)
            seen.add(value)
    return unique


def plan_vboost_values(info):
    raw = os.environ.get("GB10_VBOOST_VALUES", "auto").strip()
    if not raw or raw.lower() == "auto":
        if info.get("supported_values"):
            return list(info["supported_values"]), "auto"
        cur = info.get("current_value")
        return ([cur] if cur is not None else [0]), "fallback"
    values = parse_vboost_values(raw)
    max_value = info.get("max_value")
    if max_value is not None:
        invalid = [value for value in values if value < 0 or value > max_value]
        if invalid:
            raise ValueError(f"GB10_VBOOST_VALUES includes unsupported values {invalid}; advertised max is {max_value}")
    return values, "env"


def set_vboost(value: int):
    return run_cmd(["nvidia-smi", "boost-slider", "--vboost", str(value)], timeout=20)


def percentile(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return xs[int(k)]
    return xs[f] * (c - k) + xs[c] * (k - f)


def bench_torch(out: Path, query_info=None, *, vboost_value=None, vboost_state=None):
    import torch

    query_info = query_info or probe_nvsmi_fields()
    meta = {
        "python": sys.version,
        "torch": getattr(torch, "__version__", None),
        "torch_cuda": getattr(torch.version, "cuda", None),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "env": {
            k: os.environ.get(k)
            for k in [
                "CUDA_VISIBLE_DEVICES",
                "NVIDIA_VISIBLE_DEVICES",
                "OMP_NUM_THREADS",
                "NCCL_DEBUG",
                "TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT",
                "PYTORCH_CUDA_ALLOC_CONF",
            ]
        },
        "vboost_value": vboost_value,
        "vboost_state": vboost_state,
        "nvidia_smi_start": sh("nvidia-smi", timeout=15),
        "nvidia_smi_q_clock_power": sh("nvidia-smi -q -d CLOCK,POWER,PERFORMANCE 2>/dev/null", timeout=30),
        "nvidia_smi_query_fields_requested": query_info.get("requested", []),
        "nvidia_smi_query_fields_supported": query_info.get("supported", []),
        "nvidia_smi_query_fields_unsupported": query_info.get("unsupported", []),
    }
    if not torch.cuda.is_available():
        write_json(out / "torch_meta.json", meta)
        return {"meta": meta, "matmul": [], "bandwidth": [], "allocator": []}

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    dev = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(0)
    meta["device_name"] = torch.cuda.get_device_name(0)
    meta["device_capability"] = torch.cuda.get_device_capability(0)
    meta["device_properties"] = {k: str(getattr(props, k)) for k in dir(props) if not k.startswith("_") and k not in ("__class__",)}
    meta["transformer_engine"] = None
    try:
        import transformer_engine as te

        meta["transformer_engine"] = getattr(te, "__version__", "present")
    except Exception as e:
        meta["transformer_engine"] = f"not importable: {e!r}"
    try:
        import triton

        meta["triton"] = getattr(triton, "__version__", "present")
    except Exception as e:
        meta["triton"] = f"not importable: {e!r}"

    results = {"meta": meta, "matmul": [], "bandwidth": [], "allocator": []}
    write_json(out / "torch_meta.json", meta)

    sizes = [int(x) for x in os.environ.get("BENCH_SIZES", "4096,8192,12288,16384").split(",") if x.strip()]
    dtypes = [("tf32", torch.float32), ("fp32", torch.float32), ("bf16", torch.bfloat16), ("fp16", torch.float16)]

    matmul_seconds = float(os.environ.get("BENCH_SECONDS", "20"))
    max_alloc_frac = float(os.environ.get("BENCH_MAX_ALLOC_FRAC", "0.55"))
    total_mem = props.total_memory
    supported_fields = query_info.get("supported", [])

    for dtype_name, dtype in dtypes:
        for n in sizes:
            bytes_needed = 3 * n * n * torch.tensor([], dtype=dtype).element_size()
            if bytes_needed > total_mem * max_alloc_frac:
                results["matmul"].append({
                    "vboost": vboost_value,
                    "dtype": dtype_name,
                    "n": n,
                    "skipped": True,
                    "reason": f"needs {bytes_needed} bytes > fraction of total mem",
                })
                continue
            try:
                torch.backends.cuda.matmul.allow_tf32 = dtype_name != "fp32"
                a = torch.randn((n, n), device=dev, dtype=dtype)
                b = torch.randn((n, n), device=dev, dtype=dtype)
                for _ in range(8):
                    c = a @ b
                torch.cuda.synchronize()
                times = []
                t_end = time.perf_counter() + matmul_seconds
                while time.perf_counter() < t_end:
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    c = a @ b
                    end.record()
                    end.synchronize()
                    times.append(start.elapsed_time(end) / 1000.0)
                flops = 2.0 * n * n * n
                rec = {
                    "vboost": vboost_value,
                    "dtype": dtype_name,
                    "torch_dtype": str(dtype),
                    "n": n,
                    "iterations": len(times),
                    "median_seconds": percentile(times, 50),
                    "p05_seconds": percentile(times, 5),
                    "p95_seconds": percentile(times, 95),
                    "best_seconds": min(times) if times else None,
                    "median_TFLOP_s": flops / percentile(times, 50) / 1e12 if times else None,
                    "best_TFLOP_s": flops / min(times) / 1e12 if times else None,
                    "nvidia_smi_after": nvsmi_query(supported_fields),
                }
                print(json.dumps({"matmul": rec}, sort_keys=True), flush=True)
                results["matmul"].append(rec)
                del a, b, c
                torch.cuda.empty_cache()
            except Exception as e:
                results["matmul"].append({"vboost": vboost_value, "dtype": dtype_name, "n": n, "error": repr(e)})
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

    for mib in [256, 1024, 4096, 8192]:
        numel = mib * 1024 * 1024 // 4
        try:
            d0 = torch.empty(numel, device=dev, dtype=torch.float32)
            d1 = torch.empty_like(d0)
            for _ in range(8):
                d1.copy_(d0)
            torch.cuda.synchronize()
            times = []
            for _ in range(30):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                d1.copy_(d0)
                end.record()
                end.synchronize()
                times.append(start.elapsed_time(end) / 1000.0)
            rec = {
                "vboost": vboost_value,
                "kind": "device_to_device_copy",
                "MiB": mib,
                "median_GB_s": (mib * 1024 * 1024) / percentile(times, 50) / 1e9,
                "best_GB_s": (mib * 1024 * 1024) / min(times) / 1e9,
                "nvidia_smi_after": nvsmi_query(supported_fields),
            }
            results["bandwidth"].append(rec)
            print(json.dumps({"bandwidth": rec}, sort_keys=True), flush=True)
            del d0, d1
            torch.cuda.empty_cache()
        except Exception as e:
            results["bandwidth"].append({"vboost": vboost_value, "kind": "device_to_device_copy", "MiB": mib, "error": repr(e)})

        try:
            h = torch.empty(numel, device="cpu", dtype=torch.float32, pin_memory=True)
            d = torch.empty(numel, device=dev, dtype=torch.float32)
            stream = torch.cuda.Stream()
            for direction in ["h2d", "d2h"]:
                times = []
                for _ in range(20):
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    with torch.cuda.stream(stream):
                        start.record(stream)
                        if direction == "h2d":
                            d.copy_(h, non_blocking=True)
                        else:
                            h.copy_(d, non_blocking=True)
                        end.record(stream)
                    end.synchronize()
                    times.append(start.elapsed_time(end) / 1000.0)
                rec = {
                    "vboost": vboost_value,
                    "kind": direction,
                    "MiB": mib,
                    "median_GB_s": (mib * 1024 * 1024) / percentile(times, 50) / 1e9,
                    "best_GB_s": (mib * 1024 * 1024) / min(times) / 1e9,
                    "nvidia_smi_after": nvsmi_query(supported_fields),
                }
                results["bandwidth"].append(rec)
                print(json.dumps({"bandwidth": rec}, sort_keys=True), flush=True)
            del h, d
            torch.cuda.empty_cache()
        except Exception as e:
            results["bandwidth"].append({"vboost": vboost_value, "kind": "pinned_h2d_d2h", "MiB": mib, "error": repr(e)})

    try:
        results["allocator"].append({"vboost": vboost_value, "memory_summary": torch.cuda.memory_summary()})
    except Exception as e:
        results["allocator"].append({"vboost": vboost_value, "error": repr(e)})

    return results


def append_flattened(aggregate, run_result):
    aggregate.setdefault("matmul", []).extend(run_result.get("matmul", []))
    aggregate.setdefault("bandwidth", []).extend(run_result.get("bandwidth", []))
    aggregate.setdefault("allocator", []).extend(run_result.get("allocator", []))


def best_rows(rows, key, *, require_vboost=True):
    filtered = [row for row in rows if row.get(key) is not None and (not require_vboost or row.get("vboost") is not None)]
    grouped = {}
    for row in filtered:
        label = row.get("dtype") or row.get("kind") or "unknown"
        prev = grouped.get(label)
        if prev is None or (row.get(key) or 0) > (prev.get(key) or 0):
            grouped[label] = row
    return grouped


def build_vboost_summary(aggregate):
    runs = aggregate.get("runs") or []
    planned = (aggregate.get("vboost") or {}).get("planned_values") or []
    mat_rows = aggregate.get("matmul") or []
    bw_rows = aggregate.get("bandwidth") or []
    preferred = [row for row in mat_rows if row.get("dtype") in {"bf16", "fp16"} and row.get("median_TFLOP_s") is not None and row.get("vboost") is not None]
    best_pref_by_vboost = {}
    for row in preferred:
        value = row["vboost"]
        score = row.get("median_TFLOP_s") or 0
        if score > best_pref_by_vboost.get(value, {}).get("median_TFLOP_s", -1):
            best_pref_by_vboost[value] = {
                "vboost": value,
                "dtype": row.get("dtype"),
                "n": row.get("n"),
                "median_TFLOP_s": score,
                "best_TFLOP_s": row.get("best_TFLOP_s"),
            }
    ordered_pref = [best_pref_by_vboost[v] for v in sorted(best_pref_by_vboost)]
    winner = None
    if ordered_pref:
        winner = max(ordered_pref, key=lambda row: row.get("median_TFLOP_s") or 0)
    summary = {
        "planned_values": planned,
        "completed_values": [run.get("vboost") for run in runs if run.get("status") == "ok"],
        "winner_by_best_bf16_fp16_median": winner,
        "best_bf16_fp16_median_by_vboost": ordered_pref,
        "best_matmul_by_dtype": best_rows(mat_rows, "median_TFLOP_s"),
        "best_bandwidth_by_kind": best_rows(bw_rows, "best_GB_s"),
        "run_statuses": [
            {
                "vboost": run.get("vboost"),
                "status": run.get("status"),
                "set_ok": (run.get("set_result") or {}).get("ok"),
                "before_current": (run.get("boost_slider_before") or {}).get("current_value"),
                "after_current": (run.get("boost_slider_after") or {}).get("current_value"),
            }
            for run in runs
        ],
    }
    return summary


def write_vboost_summary(out: Path, aggregate):
    summary = build_vboost_summary(aggregate)
    write_json(out / "vboost_summary.json", summary)
    lines = []
    lines.append("# VBoost Summary")
    lines.append("")
    lines.append(f"Planned values: {', '.join(str(v) for v in summary.get('planned_values') or []) or 'none'}")
    lines.append(f"Completed values: {', '.join(str(v) for v in summary.get('completed_values') or []) or 'none'}")
    winner = summary.get("winner_by_best_bf16_fp16_median")
    if winner:
        lines.append(
            f"Winner by BF16/FP16 median TFLOP/s: vboost={winner.get('vboost')} dtype={winner.get('dtype')} n={winner.get('n')} median={winner.get('median_TFLOP_s'):.3f} best={winner.get('best_TFLOP_s'):.3f}"
        )
    else:
        lines.append("Winner by BF16/FP16 median TFLOP/s: unavailable")
    lines.append("")
    lines.append("## Best BF16/FP16 median TFLOP/s by vboost")
    best_pref = summary.get("best_bf16_fp16_median_by_vboost") or []
    if best_pref:
        for row in best_pref:
            lines.append(
                f"- vboost={row.get('vboost')}: dtype={row.get('dtype')} n={row.get('n')} median={row.get('median_TFLOP_s'):.3f} best={row.get('best_TFLOP_s'):.3f}"
            )
    else:
        lines.append("- unavailable")
    lines.append("")
    lines.append("## Run status")
    for row in summary.get("run_statuses") or []:
        lines.append(
            f"- vboost={row.get('vboost')}: status={row.get('status')} set_ok={row.get('set_ok')} before={row.get('before_current')} after={row.get('after_current')}"
        )
    write_text(out / "vboost_summary.md", "\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/results/bench")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    query_info = probe_nvsmi_fields()
    initial_vboost = query_vboost_state()
    planned_values, plan_source = plan_vboost_values(initial_vboost)
    settle_seconds = float(os.environ.get("GB10_VBOOST_SETTLE_S", "5"))
    restore_target = initial_vboost.get("current_value")
    if restore_target is None and planned_values:
        restore_target = planned_values[0]
    if restore_target is None:
        restore_target = 0

    write_json(out / "nvidia_smi_live.meta.json", query_info)
    write_json(out / "vboost_initial.json", initial_vboost)
    write_json(
        out / "vboost_plan.json",
        {
            "plan_source": plan_source,
            "planned_values": planned_values,
            "settle_seconds": settle_seconds,
            "restore_target": restore_target,
        },
    )

    aggregate = {
        "meta": {
            "nvidia_smi_query_fields_requested": query_info.get("requested", []),
            "nvidia_smi_query_fields_supported": query_info.get("supported", []),
            "nvidia_smi_query_fields_unsupported": query_info.get("unsupported", []),
        },
        "vboost": {
            "initial": initial_vboost,
            "plan_source": plan_source,
            "planned_values": planned_values,
            "settle_seconds": settle_seconds,
            "restore_target": restore_target,
        },
        "runs": [],
        "matmul": [],
        "bandwidth": [],
        "allocator": [],
    }

    try:
        for value in planned_values:
            run_dir = out / f"vboost-{value}"
            run_dir.mkdir(parents=True, exist_ok=True)
            set_result = None
            if initial_vboost.get("available") and (initial_vboost.get("max_value") is not None or plan_source == "env"):
                set_result = set_vboost(value)
                write_text(run_dir / "boost_slider_set.txt", command_output(set_result))
                if not set_result["ok"]:
                    rec = {
                        "vboost": value,
                        "status": "vboost_set_failed",
                        "set_result": set_result,
                        "boost_slider_after_set": query_vboost_state(),
                    }
                    aggregate["runs"].append(rec)
                    write_json(out / "torch_bench.json", aggregate)
                    write_vboost_summary(out, aggregate)
                    continue
                time.sleep(settle_seconds)
            else:
                set_result = {
                    "ok": False,
                    "stdout": "",
                    "stderr": initial_vboost.get("error") or "boost-slider unavailable; benchmarking current state only",
                    "returncode": None,
                    "args": ["nvidia-smi", "boost-slider", "--vboost", str(value)],
                    "skipped": True,
                }

            boost_state_before = query_vboost_state()
            write_json(run_dir / "vboost_state_before.json", boost_state_before)
            write_json(run_dir / "nvidia_smi_live.meta.json", {"query_info": query_info, "vboost": boost_state_before})

            with Telemetry(
                run_dir / "nvidia_smi_live.csv",
                interval=float(os.environ.get("TELEMETRY_INTERVAL", "0.5")),
                query_info=query_info,
                dmon_path=(run_dir / "nvidia_smi_dmon.csv") if os.environ.get("TELEMETRY_ENABLE_DMON", "1") == "1" else None,
            ):
                run_result = bench_torch(run_dir, query_info=query_info, vboost_value=value, vboost_state=boost_state_before)

            write_json(run_dir / "torch_bench.json", run_result)
            boost_state_after = query_vboost_state()
            write_json(run_dir / "vboost_state_after.json", boost_state_after)

            rec = {
                "vboost": value,
                "status": "ok",
                "set_result": set_result,
                "boost_slider_before": boost_state_before,
                "boost_slider_after": boost_state_after,
                **run_result,
            }
            aggregate["runs"].append(rec)
            append_flattened(aggregate, run_result)
            if not aggregate["meta"].get("python") and run_result.get("meta"):
                aggregate["meta"].update(run_result["meta"])
            write_json(out / "torch_bench.json", aggregate)
            write_vboost_summary(out, aggregate)
    finally:
        restore_result = None
        if initial_vboost.get("available") and (initial_vboost.get("max_value") is not None or plan_source == "env"):
            restore_result = set_vboost(int(restore_target))
            write_text(out / "vboost_restore.txt", command_output(restore_result))
        final_vboost = query_vboost_state()
        aggregate["vboost"]["restore"] = restore_result
        aggregate["vboost"]["final"] = final_vboost
        write_json(out / "vboost_final.json", final_vboost)
        write_json(out / "torch_bench.json", aggregate)
        write_vboost_summary(out, aggregate)

    print(json.dumps({"wrote": str(out / "torch_bench.json")}, indent=2))


if __name__ == "__main__":
    main()

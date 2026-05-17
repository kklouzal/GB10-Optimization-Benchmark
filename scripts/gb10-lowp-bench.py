#!/usr/bin/env python3
"""
GB10 / DGX Spark low-precision benchmark suite.

Purpose
-------
The existing GB10 lab BF16/FP16/TF32 GEMM tests are excellent for exposing
clock, power-cap, and thermal behavior, but many real GB10 workloads spend much
of their time in FP8, MXFP8, and NVFP4 paths. This script adds workload-relevant
low-precision probes while staying safe for public benchmarking:

* PyTorch native FP8 torch._scaled_mm probes, where supported.
* Transformer Engine Linear forward probes under FP8, MXFP8, and NVFP4 recipes.
* Optional vboost sweep for low-precision workloads specifically.
* Live NVIDIA telemetry and dmon capture for each run.
* Structured skip/error records so missing framework support is visible without
  failing the entire benchmark.

The reported TFLOP/s values use dense-equivalent GEMM math: 2*M*N*K. For FP4
this is an A/B metric, not NVIDIA's sparse FP4 marketing TOPS/PFLOPS number.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import inspect
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

# Intentionally omit clocks_throttle_reasons.hw_power_brake from query fields:
# on GB10 it is visible in `nvidia-smi -q` counters but rejected by the query API.
DEFAULT_NVIDIA_SMI_FIELDS = [
    "timestamp",
    "pstate",
    "temperature.gpu",
    "power.draw",
    "utilization.gpu",
    "utilization.memory",
    "clocks.current.graphics",
    "clocks.current.sm",
    "clocks.applications.graphics",
    "clocks.max.graphics",
    "clocks_throttle_reasons.active",
    "clocks_throttle_reasons.sw_power_cap",
    "clocks_throttle_reasons.hw_slowdown",
    "clocks_throttle_reasons.sw_thermal_slowdown",
    "clocks_throttle_reasons.hw_thermal_slowdown",
]

VBOOST_MAX_RE = re.compile(r"vboost\s+max\s+value\s*:\s*(\d+)", re.I)
VBOOST_CURRENT_RE = re.compile(r"current\s+value\s*:\s*(\d+)", re.I)
VBOOST_TABLE_RE = re.compile(r"\|\s*\d+\s+vboost\s+(\d+)\s+(\d+)\s*\|", re.I)


def run_cmd(args: Sequence[str], timeout: int = 20) -> Dict[str, Any]:
    try:
        p = subprocess.run(list(args), capture_output=True, text=True, timeout=timeout)
        return {
            "ok": p.returncode == 0,
            "returncode": p.returncode,
            "stdout": p.stdout,
            "stderr": p.stderr,
            "args": list(args),
        }
    except Exception as e:  # noqa: BLE001 - diagnostic tool should not hard-fail
        return {"ok": False, "returncode": None, "stdout": "", "stderr": repr(e), "args": list(args)}


def sh(cmd: str, timeout: int = 20) -> str:
    try:
        return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT, timeout=timeout)
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {type(e).__name__}: {e}"


def command_output(res: Dict[str, Any]) -> str:
    out = (res.get("stdout") or "").strip()
    err = (res.get("stderr") or "").strip()
    return "\n".join(x for x in [out, err] if x).strip()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def parse_shape_list(spec: str) -> List[Tuple[int, int, int]]:
    shapes: List[Tuple[int, int, int]] = []
    for raw in (spec or "").split(","):
        raw = raw.strip().lower().replace("*", "x")
        if not raw:
            continue
        parts = [int(x) for x in raw.split("x") if x]
        if len(parts) == 1:
            n = parts[0]
            parts = [n, n, n]
        if len(parts) != 3:
            raise ValueError(f"invalid LOWP_SHAPES entry {raw!r}; expected N or MxNxK")
        shapes.append((parts[0], parts[1], parts[2]))
    if not shapes:
        raise ValueError("LOWP_SHAPES produced no valid shapes")
    return shapes


def percentile(xs: Sequence[float], p: float) -> Optional[float]:
    if not xs:
        return None
    ys = sorted(xs)
    k = (len(ys) - 1) * p / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return ys[int(k)]
    return ys[f] * (c - k) + ys[c] * (k - f)


def tflops_dense_equiv(m: int, n: int, k: int, seconds: Optional[float]) -> Optional[float]:
    if seconds is None or seconds <= 0:
        return None
    return (2.0 * m * n * k) / seconds / 1e12


def getenv_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def getenv_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default


def getenv_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return default


def import_version(module_name: str) -> Dict[str, Any]:
    try:
        mod = importlib.import_module(module_name)
        return {"present": True, "version": getattr(mod, "__version__", None), "file": getattr(mod, "__file__", None)}
    except Exception as e:  # noqa: BLE001
        return {"present": False, "error": repr(e)}


def probe_nvsmi_fields(candidates: Optional[List[str]] = None) -> Dict[str, Any]:
    requested = list(candidates or DEFAULT_NVIDIA_SMI_FIELDS)
    supported: List[str] = []
    unsupported: List[Dict[str, str]] = []
    for field in requested:
        res = run_cmd(["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader,nounits"], timeout=6)
        if res["ok"]:
            supported.append(field)
        else:
            unsupported.append({"field": field, "error": command_output(res)})
    return {"requested": requested, "supported": supported, "unsupported": unsupported}


def nvsmi_query(fields: Sequence[str]) -> Dict[str, Any]:
    if not fields:
        return {"fields": [], "row": {}, "error": "no supported fields"}
    res = run_cmd(["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"], timeout=6)
    if not res["ok"]:
        return {"fields": list(fields), "row": {}, "error": command_output(res)}
    line = next((ln.strip() for ln in res["stdout"].splitlines() if ln.strip()), "")
    values = [v.strip() for v in line.split(",")]
    if len(values) != len(fields):
        return {"fields": list(fields), "row": {}, "error": f"mismatch: {line!r}"}
    return {"fields": list(fields), "row": dict(zip(fields, values)), "error": None}


class Telemetry:
    def __init__(self, path: Path, fields: Sequence[str], interval: float, dmon_path: Optional[Path] = None):
        self.path = path
        self.fields = list(fields)
        self.interval = interval
        self.dmon_path = dmon_path
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.dmon_proc: Optional[subprocess.Popen[Any]] = None

    def __enter__(self) -> "Telemetry":
        if self.dmon_path and shutil.which("nvidia-smi"):
            try:
                self.dmon_proc = subprocess.Popen(
                    [
                        "nvidia-smi",
                        "dmon",
                        "-s",
                        os.environ.get("LOWP_DMON_SETS", "pucvmt"),
                        "-d",
                        os.environ.get("LOWP_DMON_INTERVAL", "1"),
                        "-o",
                        "DT",
                        "-f",
                        str(self.dmon_path),
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception:
                self.dmon_proc = None
        self.thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop.set()
        self.thread.join(timeout=5)
        if self.dmon_proc is not None:
            self.dmon_proc.terminate()
            try:
                self.dmon_proc.wait(timeout=5)
            except Exception:
                self.dmon_proc.kill()

    def _run(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["unix_time", *self.fields, "error"])
            while not self.stop.is_set():
                sample = nvsmi_query(self.fields)
                row = [time.time(), *[sample.get("row", {}).get(field, "") for field in self.fields], sample.get("error") or ""]
                w.writerow(row)
                f.flush()
                self.stop.wait(self.interval)


def summarize_telemetry(csv_path: Path) -> Dict[str, Any]:
    if not csv_path.exists():
        return {"present": False}
    try:
        rows = list(csv.DictReader(csv_path.open()))
    except Exception as e:  # noqa: BLE001
        return {"present": False, "error": repr(e)}
    numeric_fields = [
        "temperature.gpu",
        "power.draw",
        "utilization.gpu",
        "utilization.memory",
        "clocks.current.graphics",
        "clocks.current.sm",
        "clocks.applications.graphics",
        "clocks.max.graphics",
    ]
    summary: Dict[str, Any] = {"present": True, "samples": len(rows)}
    for field in numeric_fields:
        vals: List[float] = []
        for row in rows:
            raw = (row.get(field) or "").strip().replace("W", "").replace("%", "")
            if raw in {"", "N/A", "[N/A]"}:
                continue
            try:
                vals.append(float(raw))
            except Exception:
                pass
        if vals:
            summary[field] = {
                "avg": sum(vals) / len(vals),
                "min": min(vals),
                "max": max(vals),
                "p50": percentile(vals, 50),
                "p95": percentile(vals, 95),
            }
    for field in [
        "clocks_throttle_reasons.active",
        "clocks_throttle_reasons.sw_power_cap",
        "clocks_throttle_reasons.hw_slowdown",
        "clocks_throttle_reasons.sw_thermal_slowdown",
        "clocks_throttle_reasons.hw_thermal_slowdown",
    ]:
        values = [(row.get(field) or "").strip().lower() for row in rows]
        denom = max(1, len(values))
        summary[field] = {
            "active_count": sum(1 for v in values if v == "active"),
            "active_fraction": sum(1 for v in values if v == "active") / denom,
        }
    return summary


def query_vboost_state() -> Dict[str, Any]:
    res = run_cmd(["nvidia-smi", "boost-slider", "-l"], timeout=10)
    raw = command_output(res)
    info: Dict[str, Any] = {
        "available": bool(res.get("ok")),
        "raw": raw,
        "max_value": None,
        "current_value": None,
        "supported_values": [],
        "query": res,
    }
    if not res.get("ok"):
        info["error"] = raw or "boost-slider unavailable"
        return info
    for rx, key in [(VBOOST_MAX_RE, "max_value"), (VBOOST_CURRENT_RE, "current_value")]:
        m = rx.search(raw)
        if m:
            info[key] = int(m.group(1))
    m = VBOOST_TABLE_RE.search(raw)
    if m:
        info["max_value"] = info["max_value"] if info["max_value"] is not None else int(m.group(1))
        info["current_value"] = info["current_value"] if info["current_value"] is not None else int(m.group(2))
    if info["max_value"] is not None:
        info["supported_values"] = list(range(int(info["max_value"]) + 1))
    return info


def set_vboost(value: int) -> Dict[str, Any]:
    return run_cmd(["nvidia-smi", "boost-slider", "--vboost", str(value)], timeout=20)


def reset_gpu_clocks() -> Dict[str, Any]:
    return run_cmd(["nvidia-smi", "--reset-gpu-clocks"], timeout=20)


def set_gpu_clock_lock(range_spec: str) -> Dict[str, Any]:
    return run_cmd(["nvidia-smi", f"--lock-gpu-clocks={range_spec}", "--mode=0"], timeout=20)


def parse_gpu_clock_lock_values(raw: str) -> Tuple[List[Optional[str]], str]:
    raw = (raw or "").strip()
    if not raw or raw.lower() in {"none", "off", "disabled"}:
        return [None], "unlocked-only"
    values: List[Optional[str]] = []
    for part in re.split(r"[;\n]+", raw):
        part = part.strip()
        if not part:
            continue
        lowered = part.lower()
        if lowered in {"none", "off", "reset", "unlocked", "default"}:
            values.append(None)
            continue
        if not re.fullmatch(r"\d+,\d+", part):
            raise ValueError(f"invalid LOWP_GPU_CLOCK_LOCKS entry {part!r}; expected min,max or reset/off")
        values.append(part)
    deduped: List[Optional[str]] = []
    for value in values:
        if value not in deduped:
            deduped.append(value)
    if None not in deduped:
        deduped.insert(0, None)
    return deduped or [None], "explicit"


def gpu_clock_lock_label(range_spec: Optional[str]) -> str:
    return "reset" if range_spec is None else range_spec.replace(",", "-")


def parse_vboost_values(raw: str, initial: Dict[str, Any]) -> Tuple[List[Optional[int]], str]:
    raw = (raw or "current").strip().lower()
    if raw in {"", "current", "none"}:
        return [None], "current"
    if raw == "inherit":
        return parse_vboost_values(os.environ.get("GB10_VBOOST_VALUES", "current"), initial)
    if raw == "auto":
        values = initial.get("supported_values") or []
        return [int(v) for v in values] if values else [None], "auto"
    if raw in {"roundtrip", "auto-roundtrip"}:
        values = [int(v) for v in (initial.get("supported_values") or [])]
        if not values:
            return [None], "current"
        return values + list(reversed(values)), raw
    values: List[Optional[int]] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    return values or [None], "explicit"


@dataclass
class Config:
    seconds: float
    warmup: int
    shapes: List[Tuple[int, int, int]]
    max_alloc_frac: float
    telemetry_interval: float
    out_dtype: str
    te_dtype: str
    dynamic_quant: bool
    run_torch_fp8: bool
    run_te: bool
    run_trtllm_probe: bool
    gpu_clock_locks: List[Optional[str]]
    gpu_clock_lock_source: str
    gpu_clock_settle_s: float


def build_config() -> Config:
    default_shapes = "512x4096x4096,1024x8192x8192,2048x8192x8192,4096x8192x8192,8192x8192x8192"
    gpu_clock_locks, gpu_clock_lock_source = parse_gpu_clock_lock_values(os.environ.get("LOWP_GPU_CLOCK_LOCKS", ""))
    return Config(
        seconds=getenv_float("LOWP_SECONDS", 12.0),
        warmup=getenv_int("LOWP_WARMUP", 12),
        shapes=parse_shape_list(os.environ.get("LOWP_SHAPES", default_shapes)),
        max_alloc_frac=getenv_float("LOWP_MAX_ALLOC_FRAC", 0.55),
        telemetry_interval=getenv_float("LOWP_TELEMETRY_INTERVAL", 0.5),
        out_dtype=os.environ.get("LOWP_OUT_DTYPE", "bf16").strip().lower(),
        te_dtype=os.environ.get("LOWP_TE_DTYPE", os.environ.get("LOWP_TE_INPUT_DTYPE", "bf16")).strip().lower(),
        dynamic_quant=getenv_bool("LOWP_DYNAMIC_QUANT", True),
        run_torch_fp8=getenv_bool("RUN_TORCH_FP8", True),
        run_te=getenv_bool("RUN_TE_LOWP", True),
        run_trtllm_probe=getenv_bool("RUN_TRTLLM_PROBE", True),
        gpu_clock_locks=gpu_clock_locks,
        gpu_clock_lock_source=gpu_clock_lock_source,
        gpu_clock_settle_s=getenv_float("LOWP_GPU_CLOCK_SETTLE_S", getenv_float("LOWP_VBOOST_SETTLE_S", getenv_float("GB10_VBOOST_SETTLE_S", 5.0))),
    )


def torch_dtype_from_name(name: str) -> Any:
    import torch

    n = name.lower()
    if n in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if n in {"fp16", "float16", "half"}:
        return torch.float16
    if n in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"unsupported dtype name {name!r}")


def should_skip_shape(total_mem: int, frac: float, m: int, n: int, k: int, bytes_per_value: float, multiplier: float) -> Optional[str]:
    approx = multiplier * (m * k + k * n + m * n) * bytes_per_value
    if approx > total_mem * frac:
        return f"approx {approx:.0f} bytes > {frac:.2f} * CUDA visible memory {total_mem}"
    return None


def cuda_event_bench(op: Callable[[], Any], seconds: float, warmup: int) -> Dict[str, Any]:
    import torch

    for _ in range(warmup):
        op()
    torch.cuda.synchronize()
    times: List[float] = []
    end_time = time.perf_counter() + seconds
    while time.perf_counter() < end_time:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        op()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) / 1000.0)
    return {
        "iterations": len(times),
        "median_seconds": percentile(times, 50),
        "p05_seconds": percentile(times, 5),
        "p95_seconds": percentile(times, 95),
        "best_seconds": min(times) if times else None,
        "worst_seconds": max(times) if times else None,
    }


def quantize_to_float8(x: Any, dtype: Any) -> Tuple[Any, Any]:
    import torch

    finfo = torch.finfo(dtype)
    amax = torch.amax(torch.abs(x)).clamp(min=1e-12)
    scale_q = (float(finfo.max) * 0.95) / amax
    scale_dq = (1.0 / scale_q).reshape(())
    return (x * scale_q).to(dtype), scale_dq


def scaled_mm_call(a8: Any, b8: Any, scale_a: Any, scale_b: Any, out_dtype: Any) -> Any:
    import torch

    fn = getattr(torch, "_scaled_mm")
    # PyTorch/NGC builds have changed this signature. Try known variants.
    candidates = [
        lambda: fn(a8, b8, scale_a=scale_a, scale_b=scale_b, out_dtype=out_dtype, use_fast_accum=True),
        lambda: fn(a8, b8, scale_a, scale_b, out_dtype=out_dtype, use_fast_accum=True),
        lambda: fn(a8, b8, scale_a=scale_a, scale_b=scale_b, out_dtype=out_dtype),
        lambda: fn(a8, b8, scale_a, scale_b, out_dtype),
    ]
    last: Optional[BaseException] = None
    for cand in candidates:
        try:
            return cand()
        except TypeError as e:
            last = e
    assert last is not None
    raise last


def available_float8_dtypes() -> List[Tuple[str, Any]]:
    import torch

    out: List[Tuple[str, Any]] = []
    for name in ["float8_e4m3fn", "float8_e4m3fnuz", "float8_e5m2", "float8_e5m2fnuz"]:
        dtype = getattr(torch, name, None)
        if dtype is not None:
            out.append((name.replace("float8_", ""), dtype))
    return out


def run_torch_fp8(cfg: Config, total_mem: int, vboost: Optional[int]) -> List[Dict[str, Any]]:
    import torch

    rows: List[Dict[str, Any]] = []
    if not cfg.run_torch_fp8:
        return [{"suite": "torch_scaled_mm_fp8", "skipped": True, "reason": "RUN_TORCH_FP8=0", "vboost": vboost}]
    if not hasattr(torch, "_scaled_mm"):
        return [{"suite": "torch_scaled_mm_fp8", "skipped": True, "reason": "torch._scaled_mm not available", "vboost": vboost}]
    dtypes = available_float8_dtypes()
    if not dtypes:
        return [{"suite": "torch_scaled_mm_fp8", "skipped": True, "reason": "no torch.float8_* dtype exposed", "vboost": vboost}]
    out_dtype = torch_dtype_from_name(cfg.out_dtype)
    source_dtype = torch.bfloat16
    dev = torch.device("cuda:0")

    for m, n, k in cfg.shapes:
        skip = should_skip_shape(total_mem, cfg.max_alloc_frac, m, n, k, bytes_per_value=2.0, multiplier=5.0)
        if skip:
            rows.append({"suite": "torch_scaled_mm_fp8", "vboost": vboost, "m": m, "n": n, "k": k, "skipped": True, "reason": skip})
            continue
        a_src = torch.randn((m, k), device=dev, dtype=source_dtype)
        b_src = torch.randn((k, n), device=dev, dtype=source_dtype)
        for dtype_name, dtype in dtypes:
            try:
                a8, scale_a = quantize_to_float8(a_src, dtype)
                b8, scale_b = quantize_to_float8(b_src, dtype)
                # One immediate call catches unsupported shape/kernel combinations.
                _ = scaled_mm_call(a8, b8, scale_a, scale_b, out_dtype)
                torch.cuda.synchronize()
                rec = cuda_event_bench(lambda: scaled_mm_call(a8, b8, scale_a, scale_b, out_dtype), cfg.seconds, cfg.warmup)
                rec.update(
                    {
                        "suite": "torch_scaled_mm_fp8_static",
                        "vboost": vboost,
                        "dtype": dtype_name,
                        "out_dtype": cfg.out_dtype,
                        "m": m,
                        "n": n,
                        "k": k,
                        "median_TFLOP_s_dense_equiv": tflops_dense_equiv(m, n, k, rec.get("median_seconds")),
                        "best_TFLOP_s_dense_equiv": tflops_dense_equiv(m, n, k, rec.get("best_seconds")),
                        "nvidia_smi_after": nvsmi_query(DEFAULT_NVIDIA_SMI_FIELDS),
                    }
                )
                rows.append(rec)
                print(json.dumps({"lowp": rec}, sort_keys=True), flush=True)
            except Exception as e:  # noqa: BLE001
                rows.append({"suite": "torch_scaled_mm_fp8_static", "vboost": vboost, "dtype": dtype_name, "m": m, "n": n, "k": k, "error": repr(e)})
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

            if cfg.dynamic_quant:
                try:
                    # Prequantized weight + dynamic activation quantization, closer to inference.
                    b8, scale_b = quantize_to_float8(b_src, dtype)

                    def op() -> Any:
                        a8_dyn, scale_a_dyn = quantize_to_float8(a_src, dtype)
                        return scaled_mm_call(a8_dyn, b8, scale_a_dyn, scale_b, out_dtype)

                    _ = op()
                    torch.cuda.synchronize()
                    rec = cuda_event_bench(op, cfg.seconds, cfg.warmup)
                    rec.update(
                        {
                            "suite": "torch_scaled_mm_fp8_dynamic_a_static_w",
                            "vboost": vboost,
                            "dtype": dtype_name,
                            "out_dtype": cfg.out_dtype,
                            "m": m,
                            "n": n,
                            "k": k,
                            "median_TFLOP_s_dense_equiv": tflops_dense_equiv(m, n, k, rec.get("median_seconds")),
                            "best_TFLOP_s_dense_equiv": tflops_dense_equiv(m, n, k, rec.get("best_seconds")),
                            "nvidia_smi_after": nvsmi_query(DEFAULT_NVIDIA_SMI_FIELDS),
                        }
                    )
                    rows.append(rec)
                    print(json.dumps({"lowp": rec}, sort_keys=True), flush=True)
                except Exception as e:  # noqa: BLE001
                    rows.append({"suite": "torch_scaled_mm_fp8_dynamic_a_static_w", "vboost": vboost, "dtype": dtype_name, "m": m, "n": n, "k": k, "error": repr(e)})
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
        del a_src, b_src
        torch.cuda.empty_cache()
    return rows


def make_te_recipes() -> List[Tuple[str, Any]]:
    recipes: List[Tuple[str, Any]] = []
    try:
        from transformer_engine.common.recipe import DelayedScaling, Format

        recipes.append(("te_fp8_delayed_hybrid", DelayedScaling(fp8_format=Format.HYBRID, amax_history_len=16, amax_compute_algo="max")))
    except Exception as e:  # noqa: BLE001
        recipes.append(("te_fp8_delayed_hybrid", {"unavailable": repr(e)}))
    try:
        from transformer_engine.common.recipe import Format, MXFP8BlockScaling

        recipes.append(("te_mxfp8_block_e4m3", MXFP8BlockScaling(fp8_format=Format.E4M3)))
    except Exception as e:  # noqa: BLE001
        recipes.append(("te_mxfp8_block_e4m3", {"unavailable": repr(e)}))
    try:
        from transformer_engine.common.recipe import NVFP4BlockScaling

        recipes.append(("te_nvfp4_block", NVFP4BlockScaling()))
    except Exception as e:  # noqa: BLE001
        recipes.append(("te_nvfp4_block", {"unavailable": repr(e)}))
    return recipes


def te_autocast_context(te_mod: Any, recipe: Any) -> Any:
    # TE has exposed both autocast and fp8_autocast across releases.
    if hasattr(te_mod, "autocast"):
        return te_mod.autocast(enabled=True, recipe=recipe)
    if hasattr(te_mod, "fp8_autocast"):
        return te_mod.fp8_autocast(enabled=True, fp8_recipe=recipe)
    raise RuntimeError("Transformer Engine autocast/fp8_autocast not found")


def te_linear(in_features: int, out_features: int, dtype: Any) -> Any:
    import transformer_engine.pytorch as te

    try:
        return te.Linear(in_features, out_features, bias=False, params_dtype=dtype).cuda()
    except TypeError:
        layer = te.Linear(in_features, out_features, bias=False).cuda()
        return layer.to(dtype=dtype)


def run_te_lowp(cfg: Config, total_mem: int, vboost: Optional[int]) -> List[Dict[str, Any]]:
    import torch

    rows: List[Dict[str, Any]] = []
    if not cfg.run_te:
        return [{"suite": "transformer_engine_lowp", "skipped": True, "reason": "RUN_TE_LOWP=0", "vboost": vboost}]
    try:
        import transformer_engine.pytorch as te
    except Exception as e:  # noqa: BLE001
        return [{"suite": "transformer_engine_lowp", "skipped": True, "reason": f"transformer_engine.pytorch import failed: {e!r}", "vboost": vboost}]

    input_dtype = torch_dtype_from_name(cfg.te_dtype)
    recipes = make_te_recipes()
    for m, n, k in cfg.shapes:
        if m % 16 or n % 16 or k % 16:
            rows.append({"suite": "transformer_engine_lowp", "vboost": vboost, "m": m, "n": n, "k": k, "skipped": True, "reason": "TE low precision generally requires dims divisible by 16"})
            continue
        skip = should_skip_shape(total_mem, cfg.max_alloc_frac, m, n, k, bytes_per_value=2.0, multiplier=5.0)
        if skip:
            rows.append({"suite": "transformer_engine_lowp", "vboost": vboost, "m": m, "n": n, "k": k, "skipped": True, "reason": skip})
            continue
        x = torch.randn((m, k), device="cuda", dtype=input_dtype)
        for recipe_name, recipe in recipes:
            if isinstance(recipe, dict) and recipe.get("unavailable"):
                rows.append({"suite": recipe_name, "vboost": vboost, "m": m, "n": n, "k": k, "skipped": True, "reason": recipe["unavailable"]})
                continue
            try:
                layer = te_linear(k, n, input_dtype)
                # First call catches recipe/device/kernel support problems.
                with te_autocast_context(te, recipe):
                    _ = layer(x)
                torch.cuda.synchronize()

                def op() -> Any:
                    with te_autocast_context(te, recipe):
                        return layer(x)

                rec = cuda_event_bench(op, cfg.seconds, cfg.warmup)
                rec.update(
                    {
                        "suite": recipe_name,
                        "vboost": vboost,
                        "dtype": cfg.te_dtype,
                        "m": m,
                        "n": n,
                        "k": k,
                        "median_TFLOP_s_dense_equiv": tflops_dense_equiv(m, n, k, rec.get("median_seconds")),
                        "best_TFLOP_s_dense_equiv": tflops_dense_equiv(m, n, k, rec.get("best_seconds")),
                        "nvidia_smi_after": nvsmi_query(DEFAULT_NVIDIA_SMI_FIELDS),
                    }
                )
                rows.append(rec)
                print(json.dumps({"lowp": rec}, sort_keys=True), flush=True)
                del layer
                torch.cuda.empty_cache()
            except Exception as e:  # noqa: BLE001
                rows.append({"suite": recipe_name, "vboost": vboost, "m": m, "n": n, "k": k, "error": repr(e)})
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
        del x
        torch.cuda.empty_cache()
    return rows


def trtllm_probe() -> Dict[str, Any]:
    probes = {
        "tensorrt": import_version("tensorrt"),
        "tensorrt_llm": import_version("tensorrt_llm"),
        "modelopt": import_version("modelopt"),
        "vllm": import_version("vllm"),
    }
    commands = {}
    for cmd in ["trtllm-bench", "trtllm-serve", "trtllm-build", "python3"]:
        path = shutil.which(cmd)
        commands[cmd] = {"path": path}
        if path and cmd.startswith("trtllm"):
            commands[cmd]["help_head"] = sh(f"{cmd} --help 2>&1 | head -n 80", timeout=30)
    return {"modules": probes, "commands": commands}


def run_one_vboost(out: Path, cfg: Config, vboost: Optional[int], nvsmi_fields: Sequence[str]) -> Dict[str, Any]:
    import torch

    label = "current" if vboost is None else str(vboost)
    run_dir = out / f"vboost-{label}"
    run_dir.mkdir(parents=True, exist_ok=True)
    set_result = None
    if vboost is not None:
        set_result = set_vboost(vboost)
        write_text(run_dir / "vboost_set.txt", command_output(set_result))
        time.sleep(getenv_float("LOWP_VBOOST_SETTLE_S", getenv_float("GB10_VBOOST_SETTLE_S", 5.0)))
    before = query_vboost_state()
    write_json(run_dir / "vboost_before.json", before)
    write_text(run_dir / "nvidia_smi_q_before.txt", sh("nvidia-smi -q -d CLOCK,POWER,PERFORMANCE,TEMPERATURE", timeout=30))

    total_mem = torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0
    clock_lock_runs: List[Dict[str, Any]] = []
    records: List[Dict[str, Any]] = []
    for clock_lock in cfg.gpu_clock_locks:
        lock_label = gpu_clock_lock_label(clock_lock)
        lock_dir = run_dir / f"lock-{lock_label}"
        lock_dir.mkdir(parents=True, exist_ok=True)
        reset_before = reset_gpu_clocks()
        write_text(lock_dir / "gpu_clock_reset_before.txt", command_output(reset_before))
        lock_result = None
        if clock_lock is not None:
            lock_result = set_gpu_clock_lock(clock_lock)
            write_text(lock_dir / "gpu_clock_lock_set.txt", command_output(lock_result))
            if not lock_result.get("ok"):
                rec = {
                    "suite": "all",
                    "vboost": vboost,
                    "vboost_label": label,
                    "gpu_clock_lock": clock_lock,
                    "gpu_clock_lock_label": lock_label,
                    "skipped": True,
                    "reason": f"lock-gpu-clocks failed: {command_output(lock_result)}",
                }
                lock_run = {
                    "gpu_clock_lock": clock_lock,
                    "gpu_clock_lock_label": lock_label,
                    "set_result": lock_result,
                    "records": [rec],
                }
                write_json(lock_dir / "lowp_bench.json", lock_run)
                clock_lock_runs.append(lock_run)
                records.append(rec)
                continue
        time.sleep(cfg.gpu_clock_settle_s)
        write_text(lock_dir / "nvidia_smi_q_before.txt", sh("nvidia-smi -q -d CLOCK,POWER,PERFORMANCE,TEMPERATURE", timeout=30))
        lock_records: List[Dict[str, Any]] = []
        live_path = lock_dir / "lowp_nvidia_smi_live.csv"
        dmon_path = lock_dir / "lowp_nvidia_smi_dmon.csv"
        with Telemetry(
            live_path,
            nvsmi_fields,
            interval=cfg.telemetry_interval,
            dmon_path=dmon_path if getenv_bool("LOWP_ENABLE_DMON", True) else None,
        ):
            if torch.cuda.is_available():
                lock_records.extend(run_torch_fp8(cfg, total_mem, vboost))
                lock_records.extend(run_te_lowp(cfg, total_mem, vboost))
            else:
                lock_records.append({"suite": "all", "vboost": vboost, "skipped": True, "reason": "torch.cuda.is_available() is False"})
        for rec in lock_records:
            rec.setdefault("vboost", vboost)
            rec.setdefault("vboost_label", label)
            rec["gpu_clock_lock"] = clock_lock
            rec["gpu_clock_lock_label"] = lock_label
        write_text(lock_dir / "nvidia_smi_q_after.txt", sh("nvidia-smi -q -d CLOCK,POWER,PERFORMANCE,TEMPERATURE", timeout=30))
        tel_summary = summarize_telemetry(live_path)
        reset_after = reset_gpu_clocks()
        write_text(lock_dir / "gpu_clock_reset_after.txt", command_output(reset_after))
        lock_run = {
            "gpu_clock_lock": clock_lock,
            "gpu_clock_lock_label": lock_label,
            "set_result": lock_result,
            "reset_before": reset_before,
            "reset_after": reset_after,
            "telemetry_summary": tel_summary,
            "records": lock_records,
        }
        write_json(lock_dir / "lowp_bench.json", lock_run)
        clock_lock_runs.append(lock_run)
        records.extend(lock_records)

    after = query_vboost_state()
    write_json(run_dir / "vboost_after.json", after)
    write_text(run_dir / "nvidia_smi_q_after.txt", sh("nvidia-smi -q -d CLOCK,POWER,PERFORMANCE,TEMPERATURE", timeout=30))
    run = {
        "vboost": vboost,
        "label": label,
        "set_result": set_result,
        "vboost_before": before,
        "vboost_after": after,
        "clock_lock_runs": clock_lock_runs,
        "records": records,
    }
    write_json(run_dir / "lowp_bench.json", run)
    return run


def classify_lowp_issue(row: Dict[str, Any]) -> Optional[str]:
    error_text = str(row.get("error") or row.get("reason") or "")
    suite = str(row.get("suite") or "")
    if suite.startswith("torch_scaled_mm_fp8") and "Invalid scaling configuration" in error_text:
        return "PyTorch FP8 failed: invalid scale dtype/configuration"
    if suite == "te_mxfp8_block_e4m3" and "not supported on 12.0+ architectures yet" in error_text:
        return "TE MXFP8 failed: architecture support message"
    if suite == "te_nvfp4_block" and "invalid argument" in error_text.lower():
        return "TE NVFP4 failed: CUDA invalid argument"
    if row.get("error"):
        return f"{suite or 'unknown'} failed: {error_text.splitlines()[0][:140]}"
    if row.get("skipped"):
        return f"{suite or 'unknown'} skipped: {str(row.get('reason') or 'unspecified')[:140]}"
    return None


def summarize_lowp_failures(records: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], int, int]:
    failure_counts: Dict[str, int] = {}
    error_count = 0
    skipped_count = 0
    for row in records:
        if row.get("error"):
            error_count += 1
        elif row.get("skipped"):
            skipped_count += 1
        else:
            continue
        label = classify_lowp_issue(row) or "Other low-precision failure/skip"
        failure_counts[label] = failure_counts.get(label, 0) + 1
    ordered = sorted(failure_counts.items(), key=lambda item: (-item[1], item[0]))
    return ([{"label": label, "count": count} for label, count in ordered], error_count, skipped_count)


def summarize_runs(out: Path, meta: Dict[str, Any], runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []
    for run in runs:
        for rec in run.get("records", []):
            row = dict(rec)
            row.setdefault("vboost", run.get("vboost"))
            row.setdefault("vboost_label", run.get("label"))
            row.setdefault("gpu_clock_lock", None)
            row.setdefault("gpu_clock_lock_label", "reset")
            records.append(row)
    scored = [r for r in records if r.get("median_TFLOP_s_dense_equiv") is not None]
    failure_categories, error_count, skipped_count = summarize_lowp_failures(records)
    failed_or_error_count = error_count + skipped_count
    best_by_suite: Dict[str, Dict[str, Any]] = {}
    for row in scored:
        key = str(row.get("suite") or "unknown")
        prev = best_by_suite.get(key)
        if prev is None or (row.get("median_TFLOP_s_dense_equiv") or 0) > (prev.get("median_TFLOP_s_dense_equiv") or 0):
            best_by_suite[key] = row
    best_by_vboost: Dict[str, Dict[str, Any]] = {}
    for row in scored:
        key = str(row.get("vboost_label"))
        prev = best_by_vboost.get(key)
        if prev is None or (row.get("median_TFLOP_s_dense_equiv") or 0) > (prev.get("median_TFLOP_s_dense_equiv") or 0):
            best_by_vboost[key] = row
    best_by_lock: Dict[str, Dict[str, Any]] = {}
    for row in scored:
        key = str(row.get("gpu_clock_lock_label"))
        prev = best_by_lock.get(key)
        if prev is None or (row.get("median_TFLOP_s_dense_equiv") or 0) > (prev.get("median_TFLOP_s_dense_equiv") or 0):
            best_by_lock[key] = row
    summary = {
        "meta": meta,
        "record_count": len(records),
        "scored_count": len(scored),
        "failed_or_error_count": failed_or_error_count,
        "error_count": error_count,
        "skipped_count": skipped_count,
        "failure_categories": failure_categories,
        "best_by_suite": best_by_suite,
        "best_by_vboost": best_by_vboost,
        "best_by_lock": best_by_lock,
        "runs": runs,
        "records": records,
    }
    write_json(out / "lowp_bench.json", summary)

    tsv_fields = [
        "vboost_label",
        "suite",
        "dtype",
        "out_dtype",
        "m",
        "n",
        "k",
        "iterations",
        "gpu_clock_lock_label",
        "median_seconds",
        "p95_seconds",
        "best_seconds",
        "median_TFLOP_s_dense_equiv",
        "best_TFLOP_s_dense_equiv",
        "skipped",
        "reason",
        "error",
    ]
    with (out / "lowp_summary.tsv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=tsv_fields, extrasaction="ignore", delimiter="	")
        w.writeheader()
        for row in records:
            w.writerow(row)

    lines = ["# Low-precision FP8 / MXFP8 / NVFP4 benchmark summary", ""]
    lines.append(f"Records: {len(records)}")
    lines.append(f"Scored records: {len(scored)}")
    lines.append(f"Failed/error records: {failed_or_error_count}")
    if skipped_count:
        lines.append(f"Skipped records: {skipped_count}")
    lines.append("")
    if failure_categories:
        lines.append("## Failure/error summary")
        for item in failure_categories:
            lines.append(f"- {item['count']}x {item['label']}")
        lines.append("")
    lines.append("## Best result by suite")
    if best_by_suite:
        for suite, row in sorted(best_by_suite.items()):
            lines.append(
                f"- `{suite}` vboost={row.get('vboost_label')} lock={row.get('gpu_clock_lock_label')} shape={row.get('m')}x{row.get('n')}x{row.get('k')} "
                f"median={row.get('median_TFLOP_s_dense_equiv'):.3f} dense-equiv TFLOP/s "
                f"best={row.get('best_TFLOP_s_dense_equiv'):.3f}"
            )
    else:
        lines.append("- No scored low-precision result; inspect error/skipped records in `lowp_bench.json`.")
    lines.append("")
    lines.append("## Best result by vboost")
    if best_by_vboost:
        for vb, row in sorted(best_by_vboost.items(), key=lambda x: str(x[0])):
            lines.append(
                f"- vboost={vb}: `{row.get('suite')}` lock={row.get('gpu_clock_lock_label')} shape={row.get('m')}x{row.get('n')}x{row.get('k')} "
                f"median={row.get('median_TFLOP_s_dense_equiv'):.3f} dense-equiv TFLOP/s"
            )
    else:
        lines.append("- unavailable")
    lines.append("")
    lines.append("## Best result by GPU clock lock")
    if best_by_lock:
        for lock, row in sorted(best_by_lock.items(), key=lambda x: str(x[0])):
            lines.append(
                f"- lock={lock}: `{row.get('suite')}` vboost={row.get('vboost_label')} shape={row.get('m')}x{row.get('n')}x{row.get('k')} "
                f"median={row.get('median_TFLOP_s_dense_equiv'):.3f} dense-equiv TFLOP/s"
            )
    else:
        lines.append("- unavailable")
    lines.append("")
    lines.append("## Telemetry files")
    for run in runs:
        for lock_run in run.get("clock_lock_runs", []):
            lines.append(f"- `vboost-{run.get('label')}/lock-{lock_run.get('gpu_clock_lock_label')}/lowp_nvidia_smi_live.csv` and `vboost-{run.get('label')}/lock-{lock_run.get('gpu_clock_lock_label')}/lowp_nvidia_smi_dmon.csv`")
    write_text(out / "lowp_summary.md", "\n".join(lines) + "\n")
    return summary


def load_existing_summary(out: Path) -> tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    p = out / "lowp_bench.json"
    if not p.exists():
        return None, []
    try:
        data = json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return None, []
    if not isinstance(data, dict):
        return None, []
    runs = data.get("runs")
    if not isinstance(runs, list):
        runs = []
    meta = data.get("meta") if isinstance(data.get("meta"), dict) else None
    return meta, runs


def upsert_run(runs: List[Dict[str, Any]], run: Dict[str, Any]) -> List[Dict[str, Any]]:
    label = str(run.get("label"))
    kept = [r for r in runs if str(r.get("label")) != label]
    kept.append(run)

    def sort_key(row: Dict[str, Any]) -> tuple[int, Any]:
        value = row.get("vboost")
        if value is None:
            return (1, str(row.get("label")))
        return (0, value)

    kept.sort(key=sort_key)
    return kept


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/results/bench/lowp")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    append_existing = getenv_bool("LOWP_APPEND_RESULTS", False)
    existing_meta, runs = load_existing_summary(out) if append_existing else (None, [])

    cfg = build_config()
    nvsmi_probe = probe_nvsmi_fields()
    initial_vboost = query_vboost_state()
    vboost_values, vboost_source = parse_vboost_values(os.environ.get("LOWP_VBOOST_VALUES", "current"), initial_vboost)
    restore_target = initial_vboost.get("current_value")

    current_meta: Dict[str, Any] = {
        "tool": "gb10-lowp-bench",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version,
        "platform": platform.platform(),
        "config": cfg.__dict__ | {"shapes": cfg.shapes, "gpu_clock_locks": cfg.gpu_clock_locks},
        "env": {k: os.environ.get(k) for k in sorted(os.environ) if k.startswith(("LOWP_", "RUN_", "GB10_", "CUDA_", "NVIDIA_", "PYTORCH_", "OMP_", "MALLOC_"))},
        "nvidia_smi_query": nvsmi_probe,
        "nvidia_smi_start": sh("nvidia-smi", timeout=20),
        "nvidia_smi_q_start": sh("nvidia-smi -q -d CLOCK,POWER,PERFORMANCE,TEMPERATURE", timeout=30),
        "vboost_initial": initial_vboost,
        "vboost_values": vboost_values,
        "vboost_source": vboost_source,
        "module_versions": {
            "torch": import_version("torch"),
            "transformer_engine": import_version("transformer_engine"),
            "transformer_engine.pytorch": import_version("transformer_engine.pytorch"),
            "triton": import_version("triton"),
            "tensorrt": import_version("tensorrt"),
            "tensorrt_llm": import_version("tensorrt_llm"),
            "modelopt": import_version("modelopt"),
            "vllm": import_version("vllm"),
        },
        "trtllm_probe": trtllm_probe() if cfg.run_trtllm_probe else {"skipped": True, "reason": "RUN_TRTLLM_PROBE=0"},
    }
    try:
        import torch

        current_meta["torch_cuda_available"] = torch.cuda.is_available()
        current_meta["torch_version"] = getattr(torch, "__version__", None)
        current_meta["torch_cuda_version"] = getattr(torch.version, "cuda", None)
        current_meta["device_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if torch.cuda.is_available():
            current_meta["device_name"] = torch.cuda.get_device_name(0)
            current_meta["device_capability"] = torch.cuda.get_device_capability(0)
            props = torch.cuda.get_device_properties(0)
            current_meta["device_total_memory"] = props.total_memory
    except Exception as e:  # noqa: BLE001
        current_meta["torch_probe_error"] = repr(e)

    meta: Dict[str, Any] = existing_meta or current_meta
    if append_existing:
        meta["append_mode"] = True
        invocations = list(meta.get("append_invocations") or [])
        invocations.append({
            "started_utc": current_meta.get("started_utc"),
            "vboost_values": vboost_values,
            "vboost_source": vboost_source,
            "config": current_meta.get("config"),
        })
        meta["append_invocations"] = invocations
        meta["latest_invocation"] = invocations[-1]
        meta.setdefault("module_versions", current_meta.get("module_versions"))
        meta.setdefault("trtllm_probe", current_meta.get("trtllm_probe"))
    write_json(out / "lowp_meta.json", meta)

    try:
        for value in vboost_values:
            runs = upsert_run(runs, run_one_vboost(out, cfg, value, nvsmi_probe.get("supported", DEFAULT_NVIDIA_SMI_FIELDS)))
            summary = summarize_runs(out, meta, runs)
            write_json(out / "lowp_bench.json", summary)
    finally:
        if restore_target is not None and any(v is not None for v in vboost_values):
            res = set_vboost(int(restore_target))
            write_text(out / "vboost_restore.txt", command_output(res))
        write_json(out / "vboost_final.json", query_vboost_state())

    summary = summarize_runs(out, meta, runs)
    print(json.dumps({"wrote": str(out / "lowp_bench.json"), "records": summary.get("record_count")}, indent=2))


if __name__ == "__main__":
    main()

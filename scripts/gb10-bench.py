#!/usr/bin/env python3
import argparse, csv, json, math, os, shutil, subprocess, sys, threading, time
from pathlib import Path


DEFAULT_NVIDIA_SMI_FIELDS = [
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
        }
    except Exception as e:
        return {
            "ok": False,
            "stdout": "",
            "stderr": repr(e),
            "returncode": None,
        }


def probe_nvsmi_fields(candidates=None):
    requested = list(candidates or DEFAULT_NVIDIA_SMI_FIELDS)
    supported = []
    unsupported = []
    for field in requested:
        res = run_cmd(["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader,nounits"], timeout=5)
        if res["ok"]:
            supported.append(field)
        else:
            unsupported.append({"field": field, "error": (res["stderr"] or res["stdout"]).strip()})
    return {"requested": requested, "supported": supported, "unsupported": unsupported}


def nvsmi_query(fields):
    if not fields:
        return {"fields": [], "row": {}, "error": "no supported nvidia-smi query fields"}
    res = run_cmd(["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"], timeout=5)
    if not res["ok"]:
        return {"fields": list(fields), "row": {}, "error": (res["stderr"] or res["stdout"]).strip()}
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


def percentile(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100
    f = math.floor(k); c = math.ceil(k)
    if f == c: return xs[int(k)]
    return xs[f] * (c - k) + xs[c] * (k - f)


def bench_torch(out: Path, query_info=None):
    import torch
    query_info = query_info or probe_nvsmi_fields()
    meta = {
        "python": sys.version,
        "torch": getattr(torch, "__version__", None),
        "torch_cuda": getattr(torch.version, "cuda", None),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "env": {k: os.environ.get(k) for k in ["CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "NCCL_DEBUG", "TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT", "PYTORCH_CUDA_ALLOC_CONF"]},
        "nvidia_smi_start": sh("nvidia-smi", timeout=15),
        "nvidia_smi_q_clock_power": sh("nvidia-smi -q -d CLOCK,POWER,PERFORMANCE 2>/dev/null", timeout=30),
        "nvidia_smi_query_fields_requested": query_info.get("requested", []),
        "nvidia_smi_query_fields_supported": query_info.get("supported", []),
        "nvidia_smi_query_fields_unsupported": query_info.get("unsupported", []),
    }
    if not torch.cuda.is_available():
        (out / "torch_meta.json").write_text(json.dumps(meta, indent=2))
        return {"meta": meta, "matmul": [], "bandwidth": []}

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
    (out / "torch_meta.json").write_text(json.dumps(meta, indent=2, default=str))

    # Matrix sizes can be overridden with BENCH_SIZES=4096,8192,12288,16384
    sizes = [int(x) for x in os.environ.get("BENCH_SIZES", "4096,8192,12288,16384").split(",") if x.strip()]
    dtypes = []
    for name, dtype in [("tf32", torch.float32), ("fp32", torch.float32), ("bf16", torch.bfloat16), ("fp16", torch.float16)]:
        dtypes.append((name, dtype))

    matmul_seconds = float(os.environ.get("BENCH_SECONDS", "20"))
    max_alloc_frac = float(os.environ.get("BENCH_MAX_ALLOC_FRAC", "0.55"))
    total_mem = props.total_memory

    for dtype_name, dtype in dtypes:
        for n in sizes:
            bytes_needed = 3 * n * n * torch.tensor([], dtype=dtype).element_size()
            if bytes_needed > total_mem * max_alloc_frac:
                results["matmul"].append({"dtype": dtype_name, "n": n, "skipped": True, "reason": f"needs {bytes_needed} bytes > fraction of total mem"})
                continue
            try:
                if dtype_name == "fp32":
                    torch.backends.cuda.matmul.allow_tf32 = False
                else:
                    torch.backends.cuda.matmul.allow_tf32 = True
                a = torch.randn((n, n), device=dev, dtype=dtype)
                b = torch.randn((n, n), device=dev, dtype=dtype)
                # Warmup.
                for _ in range(8):
                    c = a @ b
                torch.cuda.synchronize()
                times=[]
                t_end = time.perf_counter() + matmul_seconds
                while time.perf_counter() < t_end:
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record(); c = a @ b; end.record(); end.synchronize()
                    times.append(start.elapsed_time(end) / 1000.0)
                flops = 2.0 * n * n * n
                rec = {
                    "dtype": dtype_name,
                    "torch_dtype": str(dtype),
                    "n": n,
                    "iterations": len(times),
                    "median_seconds": percentile(times, 50),
                    "p05_seconds": percentile(times, 5),
                    "p95_seconds": percentile(times, 95),
                    "best_seconds": min(times) if times else None,
                    "median_TFLOP_s": flops / percentile(times,50) / 1e12 if times else None,
                    "best_TFLOP_s": flops / min(times) / 1e12 if times else None,
                    "nvidia_smi_after": nvsmi_query(query_info.get("supported", [])),
                }
                print(json.dumps({"matmul": rec}, sort_keys=True), flush=True)
                results["matmul"].append(rec)
                del a,b,c
                torch.cuda.empty_cache()
            except Exception as e:
                results["matmul"].append({"dtype": dtype_name, "n": n, "error": repr(e)})
                try: torch.cuda.empty_cache()
                except Exception: pass

    # Memory bandwidth probes via PyTorch. These are not a replacement for nvbandwidth,
    # but they catch pinned-memory and allocator problems in real Python workloads.
    for mib in [256, 1024, 4096, 8192]:
        numel = mib * 1024 * 1024 // 4
        try:
            d0 = torch.empty(numel, device=dev, dtype=torch.float32)
            d1 = torch.empty_like(d0)
            for _ in range(8): d1.copy_(d0)
            torch.cuda.synchronize()
            times=[]
            for _ in range(30):
                start=torch.cuda.Event(enable_timing=True); end=torch.cuda.Event(enable_timing=True)
                start.record(); d1.copy_(d0); end.record(); end.synchronize(); times.append(start.elapsed_time(end)/1000.0)
            rec={"kind":"device_to_device_copy", "MiB":mib, "median_GB_s":(mib*1024*1024)/percentile(times,50)/1e9, "best_GB_s":(mib*1024*1024)/min(times)/1e9, "nvidia_smi_after":nvsmi_query()}
            results["bandwidth"].append(rec); print(json.dumps({"bandwidth": rec}, sort_keys=True), flush=True)
            del d0,d1
            torch.cuda.empty_cache()
        except Exception as e:
            results["bandwidth"].append({"kind":"device_to_device_copy", "MiB":mib, "error":repr(e)})

        try:
            h = torch.empty(numel, device="cpu", dtype=torch.float32, pin_memory=True)
            d = torch.empty(numel, device=dev, dtype=torch.float32)
            stream = torch.cuda.Stream()
            for direction in ["h2d", "d2h"]:
                times=[]
                for _ in range(20):
                    start=torch.cuda.Event(enable_timing=True); end=torch.cuda.Event(enable_timing=True)
                    with torch.cuda.stream(stream):
                        start.record(stream)
                        if direction == "h2d": d.copy_(h, non_blocking=True)
                        else: h.copy_(d, non_blocking=True)
                        end.record(stream)
                    end.synchronize(); times.append(start.elapsed_time(end)/1000.0)
                rec={"kind":direction, "MiB":mib, "median_GB_s":(mib*1024*1024)/percentile(times,50)/1e9, "best_GB_s":(mib*1024*1024)/min(times)/1e9, "nvidia_smi_after":nvsmi_query(query_info.get("supported", []))}
                results["bandwidth"].append(rec); print(json.dumps({"bandwidth": rec}, sort_keys=True), flush=True)
            del h,d
            torch.cuda.empty_cache()
        except Exception as e:
            results["bandwidth"].append({"kind":"pinned_h2d_d2h", "MiB":mib, "error":repr(e)})

    try:
        results["allocator"].append({"memory_summary": torch.cuda.memory_summary()})
    except Exception as e:
        results["allocator"].append({"error": repr(e)})

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/results/bench")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    query_info = probe_nvsmi_fields()
    (out / "nvidia_smi_live.meta.json").write_text(json.dumps(query_info, indent=2))
    with Telemetry(
        out / "nvidia_smi_live.csv",
        interval=float(os.environ.get("TELEMETRY_INTERVAL", "0.5")),
        query_info=query_info,
        dmon_path=(out / "nvidia_smi_dmon.csv") if os.environ.get("TELEMETRY_ENABLE_DMON", "1") == "1" else None,
    ):
        result = bench_torch(out, query_info=query_info)

    (out / "torch_bench.json").write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps({"wrote": str(out / "torch_bench.json")}, indent=2))


if __name__ == "__main__":
    main()

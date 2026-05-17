#!/usr/bin/env python3
"""Append optional low-precision/tunables sections to report.md.

This keeps the existing analyzer simple while allowing add-on results to appear
in the top-level report when present.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List


def read(path: Path, limit: int = 2_000_000) -> str:
    try:
        return path.read_bytes()[:limit].decode("utf-8", errors="replace")
    except Exception:
        return ""


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def fmt_num(x: Any, digits: int = 3) -> str:
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return "n/a"


def classify_lowp_issue(row: Dict[str, Any]) -> str | None:
    error_text = str(row.get("error") or row.get("reason") or "")
    suite = str(row.get("suite") or "")
    if suite.startswith("torch_scaled_mm_fp8") and "Invalid scaling configuration" in error_text:
        return "PyTorch FP8 failed: invalid scale dtype/configuration"
    if suite == "te_mxfp8_block_e4m3" and "not supported on 12.0+ architectures yet" in error_text:
        return "TE MXFP8 failed: architecture support message"
    if suite == "te_nvfp4_block" and "invalid argument" in error_text.lower():
        return "TE NVFP4 failed: CUDA invalid argument"
    if row.get("error"):
        return f"{suite or 'unknown'} failed"
    if row.get("skipped"):
        return f"{suite or 'unknown'} skipped"
    return None


def summarize_lowp_failures(records: List[Dict[str, Any]]) -> tuple[List[tuple[str, int]], int, int]:
    counts: Dict[str, int] = {}
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
        counts[label] = counts.get(label, 0) + 1
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return ordered, error_count, skipped_count


def lowp_section(root: Path) -> str:
    p = root / "bench" / "lowp" / "lowp_bench.json"
    if not p.exists():
        return ""
    data = load_json(p) or {}
    lines: List[str] = ["", "## Low-precision FP8 / MXFP8 / NVFP4 results", ""]
    records = data.get("records") or []
    scored = [r for r in records if r.get("median_TFLOP_s_dense_equiv") is not None]
    failure_categories, error_count, skipped_count = summarize_lowp_failures(records)
    failed_or_error_count = error_count + skipped_count
    lines.append(f"Low-precision records: `{len(records)}` total.")
    lines.append(f"Scored records: `{len(scored)}`.")
    lines.append(f"Failed/error records: `{failed_or_error_count}`.")
    if skipped_count:
        lines.append(f"Skipped records: `{skipped_count}`.")
    if failure_categories:
        lines.append("")
        lines.append("Main low-precision failure modes:")
        for label, count in failure_categories[:8]:
            lines.append(f"- `{count}x` {label}")
    best_by_suite = data.get("best_by_suite") or {}
    if best_by_suite:
        lines.append("")
        lines.append("Best dense-equivalent median TFLOP/s by low-precision suite:")
        for suite, row in sorted(best_by_suite.items()):
            lines.append(
                f"- `{suite}` vboost=`{row.get('vboost_label')}` lock=`{row.get('gpu_clock_lock_label')}` shape=`{row.get('m')}x{row.get('n')}x{row.get('k')}` "
                f"median=`{fmt_num(row.get('median_TFLOP_s_dense_equiv'))}` best=`{fmt_num(row.get('best_TFLOP_s_dense_equiv'))}`"
            )
    best_by_vboost = data.get("best_by_vboost") or {}
    if best_by_vboost:
        lines.append("")
        lines.append("Best low-precision result by vboost:")
        for vb, row in sorted(best_by_vboost.items(), key=lambda x: str(x[0])):
            lines.append(
                f"- vboost=`{vb}` suite=`{row.get('suite')}` lock=`{row.get('gpu_clock_lock_label')}` shape=`{row.get('m')}x{row.get('n')}x{row.get('k')}` "
                f"median=`{fmt_num(row.get('median_TFLOP_s_dense_equiv'))}`"
            )
    if not scored:
        lines.append("No scored low-precision cases were produced; inspect `bench/lowp/lowp_bench.json` for framework/kernel support errors.")
    lines.append("")
    lines.append("Inspect `bench/lowp/lowp_summary.md`, `bench/lowp/lowp_summary.tsv`, and per-vboost telemetry under `bench/lowp/vboost-*`.")
    return "\n".join(lines) + "\n"


def tunables_section(root: Path) -> str:
    p = root / "tunables" / "tunables.json"
    if not p.exists():
        return ""
    data = load_json(p) or {}
    candidates = data.get("candidates") or []
    high = [c for c in candidates if c.get("priority") == "high"]
    lines: List[str] = ["", "## Tunability matrix", ""]
    lines.append(f"Tunability candidates detected: `{len(candidates)}` total, `{len(high)}` high priority.")
    for c in high[:12]:
        lines.append(f"- **{c.get('category')} / {c.get('name')}**: state=`{c.get('state')}` — {c.get('evidence')}")
    lines.append("")
    lines.append("Inspect `tunables/tunables.md` and `tunables/tunables.json` for the full A/B inventory.")
    return "\n".join(lines) + "\n"


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "/results/latest")
    report = root / "report.md"
    if not root.exists():
        print(f"missing root: {root}", file=sys.stderr)
        sys.exit(2)
    text = read(report) if report.exists() else "# GB10 Spark Perf Lab Report\n"
    marker = "<!-- gb10-lowp-tunables-append -->"
    text = text.split(marker)[0].rstrip() + "\n\n" + marker + "\n"
    text += lowp_section(root)
    text += tunables_section(root)
    report.write_text(text)
    print(f"updated {report}")


if __name__ == "__main__":
    main()

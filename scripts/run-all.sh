#!/usr/bin/env bash
set -Eeuo pipefail
source "${GB10_LAB_HOME:-/opt/gb10-spark-perf-lab}/scripts/common.sh"
export GB10_OUT="$OUT"
export GB10_TS="$TS"
log "GB10 Spark Perf Lab starting: $OUT"
"$LAB_HOME/scripts/collect.sh" --no-archive
"$LAB_HOME/scripts/bench.sh" --no-archive
"$LAB_HOME/scripts/gb10-analyze.py" "$OUT" || true
python3 "$LAB_HOME/scripts/gb10-report-append.py" "$OUT" || true
archive="$(archive_out)"
log "created archive: $archive"
printf '\nCreated archive:\n%s\nReport:\n%s/report.md\n' "$archive" "$OUT"

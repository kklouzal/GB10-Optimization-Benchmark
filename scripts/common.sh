#!/usr/bin/env bash
set -Eeuo pipefail

LAB_HOME="${GB10_LAB_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RESULTS_ROOT="${GB10_RESULTS:-/results}"
TS="${GB10_TS:-$(date -u +%Y%m%dT%H%M%SZ)}"
HOSTNAME_SAFE="$(hostname -s 2>/dev/null | tr -c 'A-Za-z0-9_.-' '_' || echo host)"
OUT="${GB10_OUT:-${RESULTS_ROOT}/gb10-lab-${HOSTNAME_SAFE}-${TS}}"
mkdir -p "$OUT"

log() { printf '[%s] %s\n' "$(date -Iseconds)" "$*" | tee -a "$OUT/_index.log" >&2; }

have() { command -v "$1" >/dev/null 2>&1; }

run() {
  local rel="$1"; shift
  local file="$OUT/${rel}.txt"
  mkdir -p "$(dirname "$file")"
  log "$rel"
  (
    echo "### $rel"
    echo "### started=$(date -Iseconds)"
    echo "### command=$*"
    echo
    timeout "${CMD_TIMEOUT:-240}" bash -lc "$*"
    ec=$?
    echo
    echo "### exit_status=$ec"
    echo "### ended=$(date -Iseconds)"
    exit "$ec"
  ) >"$file" 2>&1 || true
}

run_json() {
  local rel="$1"; shift
  local file="$OUT/${rel}.json"
  mkdir -p "$(dirname "$file")"
  log "$rel"
  timeout "${CMD_TIMEOUT:-240}" bash -lc "$*" >"$file" 2>"${file%.json}.stderr.txt" || true
}

hostrun() {
  if [[ "${GB10_DISABLE_NSENTER:-0}" != "1" ]] && have nsenter && [[ -e /proc/1/ns/mnt ]]; then
    nsenter -t 1 -m -u -i -n -p -- bash -lc "$*"
  elif [[ -d /host/bin ]]; then
    chroot /host bash -lc "$*"
  else
    bash -lc "$*"
  fi
}

run_host() {
  local rel="$1"; shift
  local file="$OUT/${rel}.txt"
  mkdir -p "$(dirname "$file")"
  log "$rel [host]"
  (
    echo "### $rel [host namespace]"
    echo "### started=$(date -Iseconds)"
    echo "### command=$*"
    echo
    hostrun "$*"
    ec=$?
    echo
    echo "### exit_status=$ec"
    echo "### ended=$(date -Iseconds)"
    exit "$ec"
  ) >"$file" 2>&1 || true
}

copy_host_file() {
  local src="$1" rel="$2"
  mkdir -p "$(dirname "$OUT/$rel")"
  if [[ -r "/host$src" ]]; then
    cp -a "/host$src" "$OUT/$rel" 2>/dev/null || true
  elif [[ -r "$src" ]]; then
    cp -a "$src" "$OUT/$rel" 2>/dev/null || true
  fi
}

redact_tree() {
  [[ "${REDACT:-1}" == "1" ]] || return 0
  log "redacting common serial/IP/MAC/token fields"
  python3 "$LAB_HOME/scripts/redact-results.py" "$OUT"
}

archive_out() {
  local tar_path="${OUT}.tar"
  local gz_path="${tar_path}.gz"
  redact_tree
  rm -f "$tar_path" "$gz_path"
  tar -C "$(dirname "$OUT")" -cf "$tar_path" "$(basename "$OUT")"
  gzip -f "$tar_path"
  echo "$gz_path"
}

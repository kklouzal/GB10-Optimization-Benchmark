#!/usr/bin/env bash
set -Euo pipefail

LAB_HOME="${GB10_LAB_HOME:-/opt/gb10-spark-perf-lab}"
source "$LAB_HOME/scripts/common.sh"

level="${RUN_DCGM_LEVEL:-1}"
out_dir="$OUT/bench"
summary="$out_dir/dcgm_summary.txt"
discovery_out="$out_dir/dcgm_discovery.txt"
diag_out="$out_dir/dcgm_diag_level${level}.txt"
version_out="$out_dir/dcgmi_version.txt"
hostengine_log="$out_dir/dcgm_hostengine.log"
mkdir -p "$out_dir"

owned_hostengine=0
hostengine_pid=""

diag_ec=0
discovery_ec=0

cleanup() {
  local ec=$?
  if [[ "$owned_hostengine" == "1" && -n "$hostengine_pid" ]] && kill -0 "$hostengine_pid" 2>/dev/null; then
    kill "$hostengine_pid" 2>/dev/null || true
    wait "$hostengine_pid" 2>/dev/null || true
  fi
  exit "$ec"
}
trap cleanup EXIT

capture() {
  local file="$1"
  shift
  set +e
  "$@" >"$file" 2>&1
  local ec=$?
  set -e
  return "$ec"
}

{
  echo "### started=$(date -Iseconds)"
  echo "### RUN_DCGM_LEVEL=$level"

  if ! command -v dcgmi >/dev/null 2>&1; then
    echo "dcgmi not available in container"
    exit 127
  fi
  if ! command -v nv-hostengine >/dev/null 2>&1; then
    echo "nv-hostengine not available in container"
    exit 127
  fi

  dcgmi --version >"$version_out" 2>&1 || true
  echo "### dcgmi=$(command -v dcgmi)"
  echo "### nv_hostengine=$(command -v nv-hostengine)"

  if pgrep -x nv-hostengine >/dev/null 2>&1; then
    echo "### reusing_existing_nv_hostengine=1"
  else
    echo "### starting_nv_hostengine=1"
    : >"$hostengine_log"
    nv-hostengine -n >"$hostengine_log" 2>&1 &
    hostengine_pid="$!"
    owned_hostengine=1
    for _ in $(seq 1 20); do
      if capture "$discovery_out" dcgmi discovery -l; then
        discovery_ec=0
        break
      fi
      discovery_ec=$?
      sleep 1
    done
  fi

  if [[ ! -s "$discovery_out" ]]; then
    if capture "$discovery_out" dcgmi discovery -l; then
      discovery_ec=0
    else
      discovery_ec=$?
    fi
  fi
  echo "### discovery_exit_status=$discovery_ec"

  if capture "$diag_out" dcgmi diag -r "$level"; then
    diag_ec=0
  else
    diag_ec=$?
  fi
  echo "### diag_exit_status=$diag_ec"

  if [[ -s "$hostengine_log" ]]; then
    tail -n 80 "$hostengine_log" > "$out_dir/dcgm_hostengine_tail.txt" || true
  fi

  echo "### discovery_output=$discovery_out"
  echo "### diag_output=$diag_out"
  echo "### hostengine_log=$hostengine_log"
  echo "### ended=$(date -Iseconds)"

  exit "$diag_ec"
} | tee "$summary"

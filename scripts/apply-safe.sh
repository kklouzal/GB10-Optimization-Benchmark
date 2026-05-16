#!/usr/bin/env bash
set -Eeuo pipefail
if [[ "${GB10_APPLY:-0}" != "1" ]]; then
  echo "Refusing to modify host without GB10_APPLY=1."
  echo "This mode only applies low-risk runtime knobs: nvidia persistence, CPU governor=performance, numa_balancing=0 if writable."
  exit 2
fi

echo "Applying low-risk runtime performance knobs. Reboot may reset some of these."
if command -v nvidia-smi >/dev/null; then
  nvidia-smi -pm 1 || true
fi

for p in /sys/devices/system/cpu/cpufreq/policy*; do
  [[ -w "$p/scaling_governor" ]] && echo performance > "$p/scaling_governor" || true
  [[ -w "$p/energy_performance_preference" ]] && echo performance > "$p/energy_performance_preference" || true
done

[[ -w /proc/sys/kernel/numa_balancing ]] && echo 0 > /proc/sys/kernel/numa_balancing || true
[[ -w /proc/sys/vm/swappiness ]] && echo 1 > /proc/sys/vm/swappiness || true

cat <<'DONE'
Applied safe runtime knobs where writable. Now run:
  gb10-lab collect
  gb10-lab bench
For clock locks, power profiles, hugepages, CPU isolation, or kernel command line changes, use tune-plan and A/B test manually.
DONE

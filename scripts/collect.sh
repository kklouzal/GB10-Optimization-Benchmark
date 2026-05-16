#!/usr/bin/env bash
set -Eeuo pipefail
source "${GB10_LAB_HOME:-/opt/gb10-spark-perf-lab}/scripts/common.sh"
NO_ARCHIVE=0
[[ "${1:-}" == "--no-archive" ]] && NO_ARCHIVE=1

mkdir -p "$OUT"/{host,kernel,gpu,cpu,mem,pci,nvme,nic,thermal,fw,apt,docker,services,logs,sysfs,profiling}
log "collecting system state into $OUT"

cat > "$OUT/manifest.json" <<EOFJSON
{
  "tool": "gb10-spark-perf-lab",
  "started_utc": "$(date -u -Iseconds)",
  "container_hostname": "$(hostname)",
  "out": "$OUT",
  "redact": "${REDACT:-1}",
  "notes": "Run with --privileged --pid=host --net=host --ipc=host --gpus all and -v /:/host:ro for best coverage."
}
EOFJSON

run host/container_env 'set -o posix; set | sort | grep -E "^(GB10|NVIDIA|CUDA|TORCH|TRANSFORMERS|VLLM|TRT|NCCL|CUBLAS|CUDNN|LD_|PATH|PYTHON|OMP|MKL|OPENBLAS)" || true'

run_host host/platform '
echo "== time =="; date -Iseconds; uptime
printf "\n== uname ==\n"; uname -a; uname -r; dpkg --print-architecture || true
printf "\n== os-release ==\n"; cat /etc/os-release 2>/dev/null || true
printf "\n== dgx-release ==\n"; cat /etc/dgx-release 2>/dev/null || true
printf "\n== nv_tegra_release ==\n"; cat /etc/nv_tegra_release 2>/dev/null || true
printf "\n== bootctl ==\n"; bootctl status 2>/dev/null || true
printf "\n== secure boot ==\n"; mokutil --sb-state 2>/dev/null || true
'

run_host host/hardware '
printf "== dmidecode system/baseboard/processor/memory ==\n"; dmidecode -t system -t baseboard -t processor -t memory 2>/dev/null || true
printf "\n== lshw short ==\n"; lshw -short 2>/dev/null || true
printf "\n== lsusb ==\n"; lsusb -tv 2>/dev/null || true; lsusb 2>/dev/null || true
'

run_host apt/installed_versions '
dpkg-query -W "dgx*" "nvidia*" "cuda*" "libnvidia*" "linux-image*" "linux-headers*" "linux-modules*" "docker*" "containerd*" "runc*" "libnccl*" "cudnn*" "tensorrt*" "nsight*" "dcgm*" 2>/dev/null | sort
'
run_host apt/policy '
apt-cache policy dgx-release dgx-spark-ota-update-meta dgx-repo linux-image-nvidia-hwe-24.04 linux-headers-nvidia-hwe-24.04 linux-nvidia-hwe-24.04 nvidia-driver-580 nvidia-driver-580-open nvidia-utils-580 nvidia-container-toolkit cuda-toolkit-13-0 datacenter-gpu-manager dcgm nvidia-dcgm nv-mitigations-off 2>/dev/null || true
printf "\n== upgradable ==\n"; apt list --upgradable 2>/dev/null || true
printf "\n== holds ==\n"; apt-mark showhold 2>/dev/null || true
'
run_host apt/simulated_dist_upgrade 'apt-get -s dist-upgrade 2>/dev/null || true'
run_host apt/sources 'find /etc/apt -maxdepth 4 -type f \( -name "*.list" -o -name "*.sources" -o -name "*.pref" -o -name "*.conf" \) -print -exec sed -n "1,220p" {} \; 2>/dev/null'

run_host fw/fwupd '
command -v fwupdmgr || true
fwupdmgr --version 2>/dev/null || true
printf "\n== devices ==\n"; fwupdmgr get-devices --show-all 2>/dev/null || true
printf "\n== updates ==\n"; fwupdmgr get-updates 2>/dev/null || true
printf "\n== history ==\n"; fwupdmgr get-history 2>/dev/null || true
printf "\n== remotes ==\n"; fwupdmgr get-remotes 2>/dev/null || true
'

run_host kernel/cmdline_config '
printf "== cmdline ==\n"; cat /proc/cmdline
printf "\n== boot image owner ==\n"; dpkg -S "/boot/vmlinuz-$(uname -r)" 2>/dev/null || true
printf "\n== kernel config ==\n"; grep -E "CONFIG_PREEMPT_RT=|CONFIG_PREEMPT_DYNAMIC=|CONFIG_PREEMPT_NONE=|CONFIG_PREEMPT_VOLUNTARY=|CONFIG_PREEMPT=y|CONFIG_HZ=|CONFIG_HZ_[0-9]+=|CONFIG_TRANSPARENT_HUGEPAGE|CONFIG_NUMA|CONFIG_CGROUP|CONFIG_CPU_FREQ|CONFIG_ARM64_4K_PAGES|CONFIG_ARM64_16K_PAGES|CONFIG_ARM64_64K_PAGES" "/boot/config-$(uname -r)" 2>/dev/null || true
printf "\n== mitigations ==\n"; cat /sys/devices/system/cpu/vulnerabilities/* 2>/dev/null || true
printf "\n== numa balancing ==\n"; cat /proc/sys/kernel/numa_balancing 2>/dev/null || true
printf "\n== init_on_alloc ==\n"; grep -o "init_on_alloc=[01]" /proc/cmdline || true
'
run_host kernel/grub '
printf "== grub defaults ==\n"; grep -R -n -E "GRUB_DEFAULT|GRUB_CMDLINE|GRUB_DISABLE|GRUB_TIMEOUT" /etc/default/grub /etc/default/grub.d 2>/dev/null || true
printf "\n== grub menu entries ==\n"; grep -nE "menuentry |submenu " /boot/grub/grub.cfg 2>/dev/null | head -n 160 || true
'
run_host kernel/modules '
printf "== lsmod filtered ==\n"; lsmod | grep -Ei "nvidia|nv_|mlx|mlnx|nvme|gdr|rdma|ib_|vfio|kvm|docker|overlay|zram|cpufreq|governor" || true
for m in nvidia nvidia_uvm nvidia_drm nvidia_modeset nvidia_fs mlx5_core nvme; do echo "===== modinfo $m ====="; modinfo "$m" 2>/dev/null || true; done
'
run_host kernel/sysctl_perf '
sysctl -a 2>/dev/null | grep -E "^(kernel.sched|kernel.numa|kernel.timer|kernel.perf|kernel.watchdog|kernel.softlockup|kernel.nmi|kernel.randomize_va_space|vm.swappiness|vm.overcommit|vm.dirty|vm.zone_reclaim|vm.max_map_count|vm.compaction|vm.watermark|vm.nr_hugepages|vm.nr_overcommit_hugepages|net.core|net.ipv4.tcp|fs.aio|fs.file-max)" || true
'

run_host services/dgx_nvidia '
systemctl list-unit-files | grep -Ei "dgx|nvidia|nv-|mlx|mellanox|docker|container|power|governor|thermal|fwupd" || true
printf "\n== active units ==\n"; systemctl --no-pager --type=service --state=running | grep -Ei "dgx|nvidia|nv-|mlx|mellanox|docker|container|power|governor|thermal|fwupd" || true
for u in nvidia-persistenced nv-cpu-governor nvidia-disable-init-on-alloc nvidia-disable-numa-balancing nvidia-enable-power-meter-cap nvidia-nvme-interrupt-coalescing nvidia-dgx-telemetry dgx-dashboard fwupd docker containerd; do echo "===== $u ====="; systemctl status "$u" --no-pager 2>/dev/null || true; done
'

run gpu/nvidia_smi_summary 'nvidia-smi; nvidia-smi -L; nvidia-smi topo -m 2>/dev/null || true; nvidia-smi topo -p2p r 2>/dev/null || true; nvidia-smi topo -nvme 2>/dev/null || true'
run gpu/nvidia_smi_q 'nvidia-smi -q'
run gpu/nvidia_smi_xml 'nvidia-smi -q -x'
run gpu/nvidia_smi_query_once 'nvidia-smi --query-gpu=index,timestamp,name,uuid,pci.bus_id,driver_version,pstate,temperature.gpu,power.draw,power.limit,clocks.current.graphics,clocks.current.sm,clocks.current.memory,clocks.applications.graphics,clocks.max.graphics,utilization.gpu,utilization.memory,clocks_throttle_reasons.active,clocks_throttle_reasons.sw_power_cap,clocks_throttle_reasons.hw_power_brake,clocks_throttle_reasons.hw_slowdown,clocks_throttle_reasons.sw_thermal_slowdown,clocks_throttle_reasons.hw_thermal_slowdown --format=csv 2>&1 || true'
run gpu/nvidia_smi_capabilities '
printf "== supported clocks ==\n"; nvidia-smi -q -d SUPPORTED_CLOCKS 2>/dev/null || true
printf "\n== power ==\n"; nvidia-smi -q -d POWER 2>/dev/null || true
printf "\n== clocks ==\n"; nvidia-smi -q -d CLOCK 2>/dev/null || true
printf "\n== performance ==\n"; nvidia-smi -q -d PERFORMANCE 2>/dev/null || true
printf "\n== power profiles ==\n"; nvidia-smi power-profiles -l 2>/dev/null || true; nvidia-smi power-profiles -ld 2>/dev/null || true; nvidia-smi power-profiles -gr 2>/dev/null || true; nvidia-smi power-profiles -ge 2>/dev/null || true
printf "\n== power smoothing ==\n"; nvidia-smi power-smoothing -h 2>/dev/null | head -n 120 || true; nvidia-smi power-smoothing -q 2>/dev/null || true; nvidia-smi power-smoothing -ppd 2>/dev/null || true
printf "\n== boost slider ==\n"; nvidia-smi boost-slider -l 2>/dev/null || true
printf "\n== power hint ==\n"; nvidia-smi power-hint -l 2>/dev/null || true
printf "\n== rusd ==\n"; nvidia-smi rusd -h 2>/dev/null | head -n 120 || true
printf "\n== prm ==\n"; nvidia-smi prm -l 2>/dev/null || true
'
run gpu/nvidia_idle_loop_60s 'for i in $(seq 1 60); do printf "%s," "$(date +%s.%N)"; nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,utilization.memory,power.draw,pstate,clocks.current.graphics,clocks.current.sm,clocks.applications.graphics,clocks.max.graphics,clocks_throttle_reasons.active,clocks_throttle_reasons.sw_power_cap,clocks_throttle_reasons.hw_power_brake,clocks_throttle_reasons.hw_slowdown --format=csv,noheader,nounits 2>/dev/null || true; sleep 1; done'
run gpu/nvidia_dmon_idle_60s 'timeout 65s nvidia-smi dmon -s pucvmt -d 1 2>&1 || true'
run gpu/nvlink_c2c 'nvidia-smi nvlink --status 2>/dev/null || true; nvidia-smi nvlink --info 2>/dev/null || true; nvidia-smi nvlink --capabilities 2>/dev/null || true; nvidia-smi c2c -s 2>/dev/null || true'

run cpu/lscpu 'lscpu; lscpu -e=CPU,ONLINE,CORE,SOCKET,NODE,MAXMHZ,MINMHZ,MHZ 2>/dev/null || true'
run_host cpu/cpufreq '
for p in /sys/devices/system/cpu/cpufreq/policy*; do [[ -d "$p" ]] || continue; echo "===== $p ====="; for f in scaling_governor scaling_available_governors scaling_driver scaling_cur_freq scaling_min_freq scaling_max_freq cpuinfo_min_freq cpuinfo_max_freq energy_performance_preference energy_performance_available_preferences related_cpus affected_cpus; do [[ -e "$p/$f" ]] && printf "%s: " "$f" && cat "$p/$f"; done; done
printf "\n== cpu capacities ==\n"; for f in /sys/devices/system/cpu/cpu*/cpu_capacity; do [[ -e "$f" ]] && echo "$f: $(cat "$f")"; done
command -v cpupower >/dev/null && cpupower frequency-info || true
'
run_host cpu/cpuidle '
for d in /sys/devices/system/cpu/cpu*/cpuidle/state*; do [[ -d "$d" ]] || continue; echo "===== $d ====="; for f in name desc latency power usage time disable; do [[ -e "$d/$f" ]] && printf "%s: " "$f" && cat "$d/$f"; done; done
'
run_host cpu/interrupts_threads '
printf "== interrupts ==\n"; cat /proc/interrupts
printf "\n== softirqs ==\n"; cat /proc/softirqs
printf "\n== thread placement ==\n"; ps -eLo pid,tid,psr,rtprio,ni,pri,policy,stat,comm --sort=psr,pid | head -n 700
'
run_host cpu/irq_affinity '
for f in /proc/irq/*/smp_affinity_list; do [[ -e "$f" ]] || continue; irq="${f%/*}"; irq="${irq##*/}"; printf "IRQ %s affinity=%s " "$irq" "$(cat "$f" 2>/dev/null)"; grep -m1 "^ *$irq:" /proc/interrupts || true; done
'

run_host mem/basic 'free -h; cat /proc/meminfo; numactl --hardware 2>/dev/null || true; lsmem 2>/dev/null || true; swapon --show --bytes 2>/dev/null || true; zramctl 2>/dev/null || true'
run_host mem/hugepages_thp '
printf "== transparent hugepage ==\n"; find /sys/kernel/mm/transparent_hugepage -maxdepth 2 -type f -print -exec cat {} \; 2>/dev/null || true
printf "\n== hugetlb ==\n"; grep -i huge /proc/meminfo || true; find /sys/devices/system/node -maxdepth 3 -type f -name "*huge*" -print -exec cat {} \; 2>/dev/null || true
'
run_host mem/limits '
printf "== ulimit ==\n"; ulimit -a
printf "\n== prlimit pid1 ==\n"; prlimit --pid 1 2>/dev/null || true
printf "\n== limits files ==\n"; cat /etc/security/limits.conf 2>/dev/null || true; for f in /etc/security/limits.d/*; do [[ -f "$f" ]] && echo "===== $f =====" && cat "$f"; done
printf "\n== systemd defaults ==\n"; systemctl show --property=DefaultLimitMEMLOCK,DefaultLimitNOFILE,DefaultLimitNPROC,DefaultTasksMax 2>/dev/null || true
'

run_host pci/lspci 'lspci -nn; printf "\n== tree ==\n"; lspci -tv; printf "\n== verbose ==\n"; lspci -nnvvv'
run_host pci/sysfs_link_power '
for d in /sys/bus/pci/devices/*; do [[ -d "$d" ]] || continue; echo "===== $d ====="; for f in vendor device class subsystem_vendor subsystem_device current_link_speed current_link_width max_link_speed max_link_width numa_node power/control power/runtime_status power/runtime_active_time power/runtime_suspended_time aer_dev_correctable aer_dev_fatal aer_dev_nonfatal; do [[ -e "$d/$f" ]] && printf "%s: " "$f" && cat "$d/$f" 2>/dev/null; done; done
'

run_host nvme/storage '
lsblk -o NAME,MODEL,SERIAL,SIZE,ROTA,DISC-MAX,DISC-GRAN,TRAN,FSTYPE,MOUNTPOINTS
printf "\n== findmnt ==\n"; findmnt -no TARGET,SOURCE,FSTYPE,OPTIONS
printf "\n== block queues ==\n"; for b in /sys/block/nvme* /sys/block/sd*; do [[ -d "$b" ]] || continue; echo "===== $b ====="; for f in queue/scheduler queue/nr_requests queue/read_ahead_kb queue/nomerges queue/rq_affinity queue/io_poll queue/io_poll_delay queue/write_cache queue/discard_max_bytes queue/max_sectors_kb queue/io_timeout device/model device/state; do [[ -e "$b/$f" ]] && printf "%s: " "$f" && cat "$b/$f" 2>/dev/null; done; done
printf "\n== nvme cli ==\n"; command -v nvme >/dev/null && nvme list || true; if command -v nvme >/dev/null; then for d in /dev/nvme[0-9]; do [[ -e "$d" ]] || continue; echo "===== $d id-ctrl ====="; nvme id-ctrl "$d" 2>/dev/null || true; echo "===== $d smart-log ====="; nvme smart-log "$d" 2>/dev/null || true; done; fi
'

run_host nic/basic '
ip -br addr; ip -br link; ip route
for i in /sys/class/net/*; do iface="${i##*/}"; echo "===== $iface ====="; ethtool -i "$iface" 2>/dev/null || true; ethtool "$iface" 2>/dev/null || true; ethtool -k "$iface" 2>/dev/null || true; ethtool -c "$iface" 2>/dev/null || true; ethtool -g "$iface" 2>/dev/null || true; ethtool -l "$iface" 2>/dev/null || true; done
'
run_host nic/mellanox '
printf "== ofed ==\n"; ofed_info -s 2>/dev/null || true; ofed_info 2>/dev/null || true
printf "\n== mst ==\n"; mst version 2>/dev/null || true; mst status -v 2>/dev/null || true
printf "\n== mlxconfig ==\n"; if command -v mlxconfig >/dev/null; then for dev in /dev/mst/*pciconf*; do [[ -e "$dev" ]] || continue; echo "===== mlxconfig $dev ====="; mlxconfig -d "$dev" q 2>/dev/null || true; done; fi
printf "\n== mlxlink ==\n"; if command -v mlxlink >/dev/null; then for pci in $(lspci -D 2>/dev/null | awk "/Mellanox|ConnectX|NVIDIA.*Ethernet/ {print \$1}"); do echo "===== mlxlink $pci ====="; mlxlink -d "$pci" 2>/dev/null || true; done; fi
'

run_host thermal/sensors '
printf "== sensors ==\n"; command -v sensors >/dev/null && sensors || true
printf "\n== thermal zones ==\n"; for z in /sys/class/thermal/thermal_zone*; do [[ -d "$z" ]] || continue; echo "===== $z ====="; for f in type temp mode policy trip_point_*_type trip_point_*_temp; do for x in "$z"/$f; do [[ -e "$x" ]] && printf "%s: " "${x##*/}" && cat "$x"; done; done; done
printf "\n== hwmon ==\n"; for h in /sys/class/hwmon/hwmon*; do [[ -d "$h" ]] || continue; echo "===== $h ====="; [[ -e "$h/name" ]] && cat "$h/name"; for f in "$h"/temp*_label "$h"/temp*_input "$h"/fan*_input "$h"/pwm* "$h"/power*_input "$h"/in*_input; do [[ -e "$f" ]] && printf "%s: " "${f##*/}" && cat "$f" 2>/dev/null; done; done
printf "\n== power profiles ctl ==\n"; powerprofilesctl get 2>/dev/null || true; powerprofilesctl list 2>/dev/null || true
'

run_host docker/info '
docker version 2>/dev/null || true; docker info 2>/dev/null || true
printf "\n== nvidia container tools ==\n"; nvidia-ctk --version 2>/dev/null || true; nvidia-container-cli --version 2>/dev/null || true; nvidia-container-cli info 2>/dev/null || true
printf "\n== configs ==\n"; cat /etc/docker/daemon.json 2>/dev/null || true; cat /etc/nvidia-container-runtime/config.toml 2>/dev/null || true; cat /etc/containerd/config.toml 2>/dev/null || true
printf "\n== images ==\n"; docker images --digests 2>/dev/null || true
'

run sysfs/writable_knobs '
for f in /sys/module/pcie_aspm/parameters/policy /sys/devices/system/cpu/smt/control /sys/kernel/mm/transparent_hugepage/enabled /sys/kernel/mm/transparent_hugepage/defrag /sys/kernel/mm/transparent_hugepage/khugepaged/defrag /proc/sys/kernel/numa_balancing /proc/sys/vm/swappiness /proc/sys/vm/overcommit_memory /proc/sys/vm/dirty_ratio /proc/sys/vm/dirty_background_ratio /proc/sys/vm/max_map_count; do [[ -e "$f" ]] || continue; printf "%-85s value=" "$f"; cat "$f" 2>/dev/null || true; [[ -w "$f" ]] && echo "  writable=yes" || echo "  writable=no"; done
for p in /sys/devices/system/cpu/cpufreq/policy*; do [[ -d "$p" ]] || continue; echo "===== $p writable knobs ====="; for f in scaling_governor scaling_min_freq scaling_max_freq energy_performance_preference; do [[ -e "$p/$f" ]] || continue; printf "%s value=" "$f"; cat "$p/$f" 2>/dev/null || true; [[ -w "$p/$f" ]] && echo "writable=yes" || echo "writable=no"; done; done
'

run profiling/tool_availability '
for c in nsys ncu nvprof perf bpftrace trace-cmd dcgmi nvbandwidth gb10-cuda-smoke; do echo "===== $c ====="; command -v "$c" || true; "$c" --version 2>/dev/null || "$c" -v 2>/dev/null || true; done
python3 - <<PY
import importlib, sys
mods=["torch","triton","transformer_engine","tensorrt","onnxruntime","vllm","flash_attn","cupy","numpy","pandas","pynvml"]
for m in mods:
    try:
        mod=importlib.import_module(m)
        print(f"{m}: OK version={getattr(mod,'__version__',None)}")
    except Exception as e:
        print(f"{m}: missing/error {e!r}")
PY
'

run logs/dmesg_power_thermal_pcie 'dmesg -T 2>/dev/null | grep -Ei "nvidia|nvlink|c2c|power|brake|cap|throttle|clock|thermal|temperature|fan|pcie|pci|pd|usb|type.?c|mlx|mellanox|connectx|nvme|iommu|ats|error|fail|warn|firmware|fwupd" | tail -n 3000 || true'
run_host logs/journal_warnings 'journalctl -b -p warning --no-pager 2>/dev/null | tail -n 3000 || true'
run_host logs/journal_nvidia_dgx 'journalctl -b --no-pager 2>/dev/null | grep -Ei "nvidia|dgx|spark|gb10|power|thermal|throttle|clock|fwupd|firmware|mlx|nvme|docker|container" | tail -n 3000 || true'

if [[ "${RUN_NVIDIA_BUG:-1}" == "1" ]] && have nvidia-bug-report.sh; then
  log "nvidia-bug-report"
  timeout 300 nvidia-bug-report.sh --output-file "$OUT/logs/nvidia-bug-report.log.gz" > "$OUT/logs/nvidia-bug-report.out.txt" 2>&1 || true
fi

if [[ "$NO_ARCHIVE" == "0" ]]; then
  archive="$(archive_out)"
  log "created archive: $archive"
  echo "$archive"
fi

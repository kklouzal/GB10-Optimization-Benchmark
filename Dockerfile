# syntax=docker/dockerfile:1.7
#
# GB10 Spark Perf Lab
# Repo-ready container for NVIDIA DGX Spark / Dell Pro Max with GB10 performance
# diagnostics, telemetry, and benchmarks.
#
# Default base: NVIDIA NGC PyTorch container. Override at build time:
#   docker build --build-arg BASE_IMAGE=nvcr.io/nvidia/pytorch:26.04-py3 -t gb10-spark-perf-lab:ngc .
#
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.04-py3
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.title="gb10-spark-perf-lab" \
      org.opencontainers.image.description="End-to-end GB10/DGX Spark performance diagnostics and benchmarks built on an NGC PyTorch container" \
      org.opencontainers.image.vendor="community" \
      org.opencontainers.image.source="https://github.com/YOUR_ORG/gb10-spark-perf-lab" \
      org.opencontainers.image.licenses="Apache-2.0"

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all \
    GB10_LAB_HOME=/opt/gb10-spark-perf-lab \
    GB10_RESULTS=/results

# Host inspection + benchmark dependencies. Keep this image mostly diagnostic;
# host modification is handled only by explicit scripts/modes, not during build.
RUN apt-get update && apt-get install -y --no-install-recommends \
      bash ca-certificates curl wget git openssh-client \
      build-essential cmake ninja-build pkg-config make g++ gcc gfortran \
      libboost-program-options-dev libnuma-dev \
      jq bc time moreutils parallel \
      pciutils usbutils lshw dmidecode hwloc numactl \
      iproute2 iputils-ping ethtool dnsutils net-tools \
      nvme-cli smartmontools fio sysstat stress-ng \
      lm-sensors procps psmisc kmod util-linux strace lsof file tree unzip \
      linux-tools-common bpftrace trace-cmd \
      python3-dev python3-pip python3-setuptools python3-wheel \
      vim-tiny less nano rsync tar gzip zstd \
    && rm -rf /var/lib/apt/lists/*

# NGC PyTorch containers intentionally pin Python packages with constraints.
# For this diagnostics image we install only lightweight analysis/report helpers.
RUN PIP_CONSTRAINT= python3 -m pip install --no-cache-dir \
      psutil pynvml pandas numpy pyyaml rich tabulate jinja2 matplotlib scipy packaging py-cpuinfo

# Optional: build NVIDIA nvbandwidth from source. It is the best available
# CUDA-level bandwidth probe for host<->device and intra-device copy paths.
ARG BUILD_NVBANDWIDTH=1
ARG NVBANDWIDTH_REF=v0.9
RUN if [[ "${BUILD_NVBANDWIDTH}" == "1" ]]; then \
      git clone --depth 1 --branch "${NVBANDWIDTH_REF}" https://github.com/NVIDIA/nvbandwidth.git /opt/nvbandwidth && \
      cmake -S /opt/nvbandwidth -B /opt/nvbandwidth/build -DCMAKE_BUILD_TYPE=Release && \
      cmake --build /opt/nvbandwidth/build -j"$(nproc)" && \
      install -m 0755 /opt/nvbandwidth/build/nvbandwidth /usr/local/bin/nvbandwidth ; \
    else \
      echo "Skipping nvbandwidth build"; \
    fi

# Optional: install NVIDIA DCGM directly into the benchmark image so diagnostics
# can run in-container without a permanently-enabled host service.
ARG BUILD_DCGM=1
ARG DCGM_CUDA_MAJOR=13
ARG CUDA_KEYRING_VERSION=1.1-1
ARG CUDA_APT_DIST=ubuntu2404
RUN if [[ "${BUILD_DCGM}" == "1" ]]; then \
      arch="$(dpkg --print-architecture)" && \
      case "$arch" in \
        arm64) cuda_repo_arch=sbsa ;; \
        amd64) cuda_repo_arch=x86_64 ;; \
        *) echo "Unsupported architecture for DCGM install: $arch" >&2; exit 1 ;; \
      esac && \
      keyring_url="https://developer.download.nvidia.com/compute/cuda/repos/${CUDA_APT_DIST}/${cuda_repo_arch}/cuda-keyring_${CUDA_KEYRING_VERSION}_all.deb" && \
      wget -qO /tmp/cuda-keyring.deb "$keyring_url" && \
      dpkg -i /tmp/cuda-keyring.deb && \
      rm -f /tmp/cuda-keyring.deb && \
      apt-get update && \
      apt-get install -y --no-install-recommends \
        "datacenter-gpu-manager-4-cuda${DCGM_CUDA_MAJOR}" \
        "datacenter-gpu-manager-4-proprietary-cuda${DCGM_CUDA_MAJOR}" && \
      dcgmi --version >/tmp/dcgmi-version.txt 2>&1 && \
      cat /tmp/dcgmi-version.txt && \
      rm -f /tmp/dcgmi-version.txt ; \
    else \
      echo "Skipping DCGM install"; \
    fi \
    && rm -rf /var/lib/apt/lists/*

# A tiny native CUDA sanity/memcopy probe. PyTorch benchmarks are richer, but
# this keeps a low-level CUDA runtime check independent of PyTorch.
COPY benchmarks/cuda_smoke.cu /opt/gb10-spark-perf-lab/benchmarks/cuda_smoke.cu
RUN if command -v nvcc >/dev/null 2>&1; then \
      nvcc -O3 -std=c++17 /opt/gb10-spark-perf-lab/benchmarks/cuda_smoke.cu -o /usr/local/bin/gb10-cuda-smoke ; \
    else \
      echo "nvcc not present in base image; skipping gb10-cuda-smoke build"; \
    fi

COPY scripts/ /opt/gb10-spark-perf-lab/scripts/
COPY docs/ /opt/gb10-spark-perf-lab/docs/
RUN chmod +x /opt/gb10-spark-perf-lab/scripts/*.sh && \
    ln -sf /opt/gb10-spark-perf-lab/scripts/entrypoint.sh /usr/local/bin/gb10-lab && \
    ln -sf /opt/gb10-spark-perf-lab/scripts/gb10-bench.py /usr/local/bin/gb10-bench && \
    ln -sf /opt/gb10-spark-perf-lab/scripts/gb10-analyze.py /usr/local/bin/gb10-analyze

WORKDIR /workspace
ENTRYPOINT ["/opt/gb10-spark-perf-lab/scripts/entrypoint.sh"]
CMD ["all"]

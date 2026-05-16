IMAGE ?= gb10-spark-perf-lab:ngc
BASE_IMAGE ?= nvcr.io/nvidia/pytorch:26.04-py3
NVBANDWIDTH_REF ?= v0.9
GB10_CPUSET ?= 5-9,15-19
GB10_PROFILE ?= perf-cores-runtime-maxperf
GB10_SHM_SIZE ?= 64g

build:
	docker buildx build --platform linux/arm64 \
	  --build-arg BASE_IMAGE=$(BASE_IMAGE) \
	  --build-arg BUILD_NVBANDWIDTH=1 \
	  --build-arg NVBANDWIDTH_REF=$(NVBANDWIDTH_REF) \
	  --build-arg BUILD_DCGM=1 \
	  --build-arg DCGM_CUDA_MAJOR=13 \
	  -t $(IMAGE) .

run:
	mkdir -p results
	docker run --rm -it --gpus all \
	  --cpuset-cpus=$(GB10_CPUSET) \
	  --privileged --pid=host --net=host --ipc=host --uts=host \
	  --ulimit memlock=-1 --ulimit nofile=1048576:1048576 \
	  --shm-size=$(GB10_SHM_SIZE) \
	  --security-opt seccomp=unconfined \
	  --cap-add SYS_ADMIN --cap-add SYS_PTRACE --cap-add PERFMON --cap-add IPC_LOCK --cap-add SYS_NICE \
	  -e GB10_PROFILE=$(GB10_PROFILE) -e GB10_CPUSET=$(GB10_CPUSET) -e GB10_SHM_SIZE=$(GB10_SHM_SIZE) \
	  -e RUN_NVBANDWIDTH=1 -e RUN_DCGM=1 -e RUN_DCGM_LEVEL=1 -e RUN_STREAM=1 -e RUN_FIO=0 \
	  -e OMP_NUM_THREADS=10 -e MALLOC_ARENA_MAX=2 \
	  -v /:/host:ro -v /dev:/dev -v /sys:/sys:ro -v /proc:/host_proc:ro \
	  -v "$(PWD)/results:/results" \
	  $(IMAGE) all

shell:
	docker run --rm -it --gpus all --cpuset-cpus=$(GB10_CPUSET) --privileged --pid=host --net=host --ipc=host \
	  --ulimit memlock=-1 --ulimit nofile=1048576:1048576 --shm-size=$(GB10_SHM_SIZE) \
	  -e GB10_PROFILE=$(GB10_PROFILE) -e GB10_CPUSET=$(GB10_CPUSET) -e GB10_SHM_SIZE=$(GB10_SHM_SIZE) \
	  -v /:/host:ro -v /dev:/dev -v /sys:/sys:ro -v "$(PWD)/results:/results" \
	  $(IMAGE) shell

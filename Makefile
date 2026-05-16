IMAGE ?= gb10-spark-perf-lab:ngc
BASE_IMAGE ?= nvcr.io/nvidia/pytorch:26.04-py3
NVBANDWIDTH_REF ?= v0.9

build:
	docker buildx build --platform linux/arm64 \
	  --build-arg BASE_IMAGE=$(BASE_IMAGE) \
	  --build-arg BUILD_NVBANDWIDTH=1 \
	  --build-arg NVBANDWIDTH_REF=$(NVBANDWIDTH_REF) \
	  -t $(IMAGE) .

run:
	mkdir -p results
	docker run --rm -it --gpus all \
	  --privileged --pid=host --net=host --ipc=host --uts=host \
	  --cap-add SYS_ADMIN --cap-add SYS_PTRACE --cap-add PERFMON \
	  -e RUN_NVBANDWIDTH=1 -e RUN_STREAM=1 -e RUN_FIO=0 \
	  -v /:/host:ro -v /dev:/dev -v /sys:/sys:ro -v /proc:/host_proc:ro \
	  -v "$(PWD)/results:/results" \
	  $(IMAGE) all

shell:
	docker run --rm -it --gpus all --privileged --pid=host --net=host --ipc=host \
	  -v /:/host:ro -v /dev:/dev -v /sys:/sys:ro -v "$(PWD)/results:/results" \
	  $(IMAGE) shell

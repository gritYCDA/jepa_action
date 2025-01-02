#!/bin/bash
set -x

# Default values
NODES=${NODES:-1}
GPUS=${GPUS:-1}
CONFIG=${1:-"configs/evals/vitl16_ssv2.yaml"}
ENV=${2:-"azure"}
TAG=${3:-"TEST"}

# Shift the first 3 arguments so $@ contains remaining args
shift 3

# NCCL settings for Azure
export NCCL_IB_DISABLE=0
export NCCL_IB_PCI_RELAXED_ORDERING=1
export NCCL_SOCKET_IFNAME=eth0
export NCCL_NET_GDR_LEVEL=5
export NCCL_TOPO_FILE=/opt/microsoft/ndv4-topo.xml
export NCCL_TIMEOUT=600

# Random port for master
RANDOM_PORT=$((49152 + RANDOM % 16384))

# Print configuration
echo "Running with:"
echo "Nodes: $NODES"
echo "GPUs: $GPUS"
echo "Config: $CONFIG"
echo "Env: $ENV"
echo "Tag: $TAG"
echo "Additional arguments: $@"

# Run distributed training
torchrun \
    --nnodes=${NODES} \
    --nproc_per_node=${GPUS} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT:-$RANDOM_PORT} \
    --node_rank=${NODE_RANK:-0} \
    -m evals.main_auto \
        --config=${CONFIG} \
        --env=${ENV} \
        --tag=${TAG} \
        --launcher=pytorch \
        --port=${MASTER_PORT:-$RANDOM_PORT} \
        "$@"  # Pass remaining arguments to the Python script

# # 기본값 사용
# ./run_distributed.sh

# # 환경변수로 GPU 수 지정
# GPUS=2 ./run_distributed.sh

# # 파라미터 지정
# ./run_distributed.sh configs/my_config.yaml azure my_tag

# # 둘 다 사용
# GPUS=2 ./run_distributed.sh configs/my_config.yaml azure my_tag
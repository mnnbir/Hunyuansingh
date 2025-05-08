JOBS_DIR=$(dirname $(dirname "$0"))
export MODEL_BASE=${JOBS_DIR}"/models"
export MODEL_OUTPUT_PATH=${MODEL_BASE}"/hunyuancustom_720P/"

# ========== 自动获取可用的GPU数量 ==========
NUM_GPU=${HOST_GPU_NUM}
export PYTHONPATH=${JOBS_DIR}:$PYTHONPATH
echo "PYTHONPATH: $PYTHONPATH"

if [ $NUM_GPU = 8 ];
then
    echo " ========== This node has 8 GPUs ========== "
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export GPU_NUMS=8
    echo "gpu-nums = $GPU_NUMS"
    torchrun --nnodes=1 --nproc_per_node=$GPU_NUMS --master_port 29605 ./hymm_gradio/flask_ref2v.py &
fi

python3 hymm_gradio/gradio_ref2v.py

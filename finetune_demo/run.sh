# First, cd finetune_demo
# gpu can be 2,3 or 0
gpu=$1
if [ -z $gpu ]; then
    gpu=0
fi
echo gpu=$gpu
export CUDA_VISIBLE_DEVICES=$gpu

# sinle node, single card
# nohup python finetune_hf.py  data/AdvertiseGen/  THUDM/chatglm3-6b  configs/lora.yaml \
#     > finetune_hf_single_gpu.log 2>&1 &

# single node, multi cards
OMP_NUM_THREADS=1 torchrun --standalone --nnodes=1 --nproc_per_node=2 finetune_hf.py \
    data/AdvertiseGen/  THUDM/chatglm3-6b  configs/lora.yaml \
    2>&1  </dev/null | tee finetune_hf_torchrun.log

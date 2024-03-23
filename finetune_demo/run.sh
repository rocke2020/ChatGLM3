# sinle node, single card
# cd finetune_demo
gpu=2
export CUDA_VISIBLE_DEVICES=$gpu
nohup python finetune_hf.py  data/AdvertiseGen/  THUDM/chatglm3-6b  configs/lora.yaml \
    > finetune_hf_single_gpu.log 2>&1 &
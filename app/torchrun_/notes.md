# Torchrun
## Tips and examples
nproc_per_node is the used GPU num which should be <= the num of CUDA_VISIBLE_DEVICES
1. `CUDA_VISIBLE_DEVICES=4,5 torchrun --standalone --nproc_per_node=2 main.py`

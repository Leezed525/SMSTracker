#!/bin/bash
cd /home/lzd/workspace/LeeNet
export CUDA_VISIBLE_DEVICES=0,1,2,3
nohup python -m torch.distributed.launch --nproc_per_node=4 --use_env /home/lzd/workspace/LeeNet/train/plscore_os_sigma_RGBD.py  >/home/lzd/workspace/LeeNet/logs/LeeNet_plScore_OS_sigma_RGBD/run.log 2>&1 &
echo $! > ./scripts/plscore_os_sigma_RGBD/pid.txt

#!/bin/bash
cd /home/lzd/workspace/SMSTracker
export CUDA_VISIBLE_DEVICES=0,1,2,3
nohup python -m torch.distributed.launch --nproc_per_node=4 --use_env ./train/SMSTracker_RGBT.py  > ./logs/SMSTracker_RGBT/run.log 2>&1 &
echo $! > ./scripts/plscore_os_sigma_RGBT/pid.txt

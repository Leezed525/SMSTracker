#!/bin/bash
cd /home/lzd/workspace/SMSTracker
export CUDA_VISIBLE_DEVICES=0,1,2,3
nohup python -m torch.distributed.launch --nproc_per_node=4 --use_env /home/lzd/workspace/SMSTracker/train/plscore_os_sigma_RGBT.py  >/home/lzd/workspace/SMSTracker/logs/SMSTracker_plScore_OS_sigma_RGBT/run.log 2>&1 &
echo $! > ./scripts/plscore_os_sigma_RGBT/pid.txt

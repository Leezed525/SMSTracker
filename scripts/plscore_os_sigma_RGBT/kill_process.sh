#!/bin/bash

cd /home/lzd/workspace/SMSTracker

# 从 pid.txt 文件中读取进程 ID（PID）
pid=$(cat ./scripts/plscore_os_sigma_RGBT/pid.txt)

# 检查进程是否存在
if ps -p $pid > /dev/null; then
    # 进程存在，终止它
    kill $pid
    echo "进程 PID $pid 已终止。"
else
    echo "进程 PID $pid 不存在。"
fi

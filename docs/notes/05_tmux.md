. 当前训练终端按一次 Ctrl+C，等待返回提示符。
2. 创建 tmux 会话：
tmux new -s pi05
3. 在 tmux 中执行，换新目录，保留刚才的日志：
cd /home/ca/big_space/my_project/VLA_1020/lerobot-realman-vla
source .venv/bin/activate

unset PI05_STEPS
export PI05_GPUS=0,1,2,7
export PI05_OUTPUT_DIR="$PWD/outputs/pi05_full_run02"

mkdir -p logs
set -o pipefail
bash scripts/train.sh pi05 2>&1 | tee logs/pi05_full_run02.log
4. 看到训练启动后，先按 Ctrl+B，松开，再按 D。
这会退出 tmux 界面，训练继续运行。此后可以断开 SSH、关闭笔记本，服务器需要保持开机。
以后重新连接服务器，查看训练：
tmux attach -t pi05
注意：退出查看仍用 Ctrl+B → D，不要按 Ctrl+C，后者会中断训练。
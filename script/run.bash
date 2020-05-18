#!/usr/bin/env bash
date=$(date +%Y-%m-%d-%H-%M-%S)
mkdir -p /home/payam/log/${date}
folder=/home/payam/log/${date}

tmux new-session -d -s follow -n follow_ahead
tmux split-window -t follow:follow_ahead -h
tmux split-window -t follow:follow_ahead -v
tmux send-keys -t follow:follow_ahead.0 C-z "sros1" Enter
tmux send-keys -t follow:follow_ahead.1 C-z "sros1" Enter
tmux send-keys -t follow:follow_ahead.2 C-z "sros1" Enter
sleep 10
tmux send-keys -t follow:follow_ahead.0 C-z "stdbuf -o 0 roslaunch follow_ahead_rl turtlebot.launch --wait 2>&1 | tee -ai ${folder}/gazebo.txt" Enter
tmux send-keys -t follow:follow_ahead.1 C-z "roscd follow_ahead_rl/script/" Enter
tmux send-keys -t follow:follow_ahead.1 C-z "stdbuf -o 0 python3 d4pg-pytorch/train.py --wait 2>&1 | tee -ai ${folder}/train.txt" Enter
tmux send-keys -t follow:follow_ahead.2 C-z "stdbuf -o 0 roslaunch follow_ahead_rl multi_navigation.launch --wait 2>&1 | tee -ai ${folder}/gazebo.txt" Enter
tmux a

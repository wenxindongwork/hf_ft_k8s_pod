#!/bin/bash

exp_name=$1
log_dir=$2
gs_bucket="gs://wenxindong/huggingface_tuning/$exp_name"

tmux capture-pane -pS - > ~/tmux-buffer.txt 
gsutil cp ~/tmux-buffer.txt "$gs_bucket/tmux-buffer.txt"
gsutil  -m cp -r $log_dir "$gs_bucket/profile"

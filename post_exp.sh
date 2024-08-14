#!/bin/bash

exp_name=$1
log_dir=$2
gs_bucket="gs://wenxindong/huggingface_tuning/${exp_name}/"

cd $log_dir
tmux capture-pane -pS - > tmux-buffer.txt
gsutil -m cp -r * $gs_bucket
cd -
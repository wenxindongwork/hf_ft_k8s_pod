#!/bin/bash

exp_name="llama8b_baseline"
log_dir="/tmp/$exp_name/"
rm -rf log_dir
PYTHONBREAKPOINT=0 PROFILE_EPOCH=0 PROFILE_STEP=5 PROFILE_DURATION_MS=50000 PROFILE_LOGDIR=/tmp/$exp_name/ python /llama8b_sft_trainer.py
bash /post_exp.sh $exp_name $log_dir
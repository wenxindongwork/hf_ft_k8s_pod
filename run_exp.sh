#!/bin/bash

exp_name="llama8b_baseline"
log_dir="/tmp/${exp_name}/"
rm -rf log_dir
mkdir log_dir
PYTHONBREAKPOINT=0 PROFILE_EPOCH=0 PROFILE_STEP=5 PROFILE_DURATION_MS=50000 PROFILE_LOGDIR="/tmp/${exp_name}/" python /llama8b_sft_trainer.py
bash /post_exp.sh $exp_name $log_dir

clear
exp_name="llama8b_without_lora"
log_dir="/tmp/${exp_name}/"
rm -rf log_dir
mkdir log_dir
NO_LORA=True PYTHONBREAKPOINT=0 PROFILE_EPOCH=0 PROFILE_STEP=5 PROFILE_DURATION_MS=50000 PROFILE_LOGDIR="/tmp/${exp_name}/" python /llama8b_sft_trainer.py
bash /post_exp.sh $exp_name $log_dir

clear
exp_name="llama8b_lora_kv_proj"
log_dir="/tmp/${exp_name}/"
rm -rf log_dir
mkdir log_dir
LORA_KV_PROJ=True PYTHONBREAKPOINT=0 PROFILE_EPOCH=0 PROFILE_STEP=5 PROFILE_DURATION_MS=50000 PROFILE_LOGDIR="/tmp/${exp_name}/" python /llama8b_sft_trainer.py
bash /post_exp.sh $exp_name $log_dir

clear
exp_name="llama8b_mark_step_before_opt"
log_dir="/tmp/${exp_name}/"
rm -rf log_dir
mkdir log_dir
MARK_STEP_BEFORE_OPT=True PYTHONBREAKPOINT=0 PROFILE_EPOCH=0 PROFILE_STEP=5 PROFILE_DURATION_MS=50000 PROFILE_LOGDIR="/tmp/${exp_name}/" python /llama8b_sft_trainer.py
bash /post_exp.sh $exp_name $log_dir

clear
exp_name="llama8b_no_fsdp"
log_dir="/tmp/${exp_name}/"
rm -rf log_dir
mkdir log_dir
NO_FSDP=True PYTHONBREAKPOINT=0 PROFILE_EPOCH=0 PROFILE_STEP=5 PROFILE_DURATION_MS=50000 PROFILE_LOGDIR="/tmp/${exp_name}/" python /llama8b_sft_trainer.py
bash /post_exp.sh $exp_name $log_dir

clear
exp_name="llama8b_trainer"
log_dir="/tmp/${exp_name}/"
rm -rf log_dir
mkdir log_dir
PYTHONBREAKPOINT=0 PROFILE_EPOCH=0 PROFILE_STEP=5 PROFILE_DURATION_MS=50000 PROFILE_LOGDIR="/tmp/${exp_name}/" python /llama8b_trainer.py
bash /post_exp.sh $exp_name $log_dir

clear
exp_name="llama8b_lora_kv_proj_mark_step_before_opt"
log_dir="/tmp/${exp_name}/"
rm -rf log_dir
mkdir log_dir
LORA_KV_PROJ=True MARK_STEP_BEFORE_OPT=True PYTHONBREAKPOINT=0 PROFILE_EPOCH=0 PROFILE_STEP=5 PROFILE_DURATION_MS=50000 PROFILE_LOGDIR="/tmp/${exp_name}/" python /llama8b_sft_trainer.py
bash /post_exp.sh $exp_name $log_dir

clear
exp_name="llama8b_lora_kv_proj_mark_step_before_opt_no_fsdp"
log_dir="/tmp/${exp_name}/"
rm -rf log_dir
mkdir log_dir
NO_FSDP=True LORA_KV_PROJ=True MARK_STEP_BEFORE_OPT=True PYTHONBREAKPOINT=0 PROFILE_EPOCH=0 PROFILE_STEP=5 PROFILE_DURATION_MS=50000 PROFILE_LOGDIR="/tmp/${exp_name}/" python /llama8b_sft_trainer.py
bash /post_exp.sh $exp_name $log_dir
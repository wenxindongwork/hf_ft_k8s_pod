#!/bin/bash

huggingface-cli login --token $HF_TOKEN
pip install git+https://github.com/google/cloud-accelerator-diagnostics/#subdirectory=tpu_info
pip install torch~=2.4.0 torch_xla[tpu]~=2.4.0 -f https://storage.googleapis.com/libtpu-releases/index.html
pip install tensorflow-cpu tensorboard-plugin-profile
echo "set -g history-limit 500000" >> ~/.tmux.conf
apt update
apt-get -y install tmux
apt-get -y install htop
tmux

# TF_CPP_MIN_LOG_LEVEL=0 TF_CPP_VMODULE="xla_graph_executor=5,pjrt_computation_client=3" PT_XLA_DEBUG=1 XLA_IR_DEBUG=1 XLA_HLO_DEBUG=1 XLA_SAVE_TENSORS_FMT='text' XLA_SAVE_TENSORS_FILE='/tmp/save1.ir' python /finetune_long_prof.py --mode 1 --bf16 --optim adamw_torch --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --max_steps 10 --run_name_for_checkpoints orpo1 --gradient_checkpointing False --learning_rate 8e-6 --group_by_length False --num_dataset_proc_workers 1 --eval_steps 20

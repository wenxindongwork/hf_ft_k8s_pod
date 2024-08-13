git clone -b alanwaketan/flash_attention https://github.com/pytorch-tpu/transformers.git
cd transformers
pip3 install -e .
pip3 install datasets
pip3 install evaluate
pip3 install scikit-learn
pip3 install accelerate
pip install jax

export JAX_PLATFORMS=tpu,cpu
export PJRT_DEVICE=TPU
export XLA_IR_DEBUG=1
export XLA_HLO_DEBUG=1
export PROFILE_EPOCH=0
export PROFILE_STEP=10
export PROFILE_DURATION_MS=40000
export PROFILE_LOGDIR=/tmp/wenxindong/
export XLA_USE_SPMD=1
python3 examples/pytorch/language-modeling/run_clm.py \
  --dataset_name wikitext \
  --dataset_config_name wikitext-2-raw-v1 \
  --per_device_train_batch_size 1 \
  --do_train \
  --output_dir /tmp/output \
  --overwrite_output_dir \
  --config_name /config_8b.json \
  --cache_dir /tmp/cache \
  --tokenizer_name meta-llama/Meta-Llama-3-8B \
  --block_size 8192 \
  --optim adafactor \
  --save_strategy no \
  --logging_strategy no \
  --fsdp "full_shard" \
  --fsdp_config /fsdp_config.json \
  --torch_dtype bfloat16 \
  --dataloader_drop_last yes \
  --flash_attention \
  --max_steps 20

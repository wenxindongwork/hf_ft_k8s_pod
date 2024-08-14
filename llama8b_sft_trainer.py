from optimum.tpu import fsdp_v2
from datasets import load_from_disk
from datasets import Dataset

from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from peft import LoraConfig
from trl import SFTTrainer, ORPOConfig
from transformers import TrainingArguments
import torch
import os 

model_id = "meta-llama/Meta-Llama-3-8B"
print("model_id:", model_id)

fsdp_v2.use_fsdp_v2()

print("----0----")

class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, length=1000):
        self.tokenizer = tokenizer
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        text = "This is a dummy sentence for training." 
        encodings = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        input_ids = encodings["input_ids"].squeeze()
        attention_mask = encodings["attention_mask"].squeeze()
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": input_ids} 

tokenizer = AutoTokenizer.from_pretrained(model_id)
dataset = DummyDataset(tokenizer)

print("----1----")

model = AutoModelForCausalLM.from_pretrained(model_id, use_cache=True)

print("----2----")

lora_kv_proj = os.environ.get("LORA_KV_PROJ", "False").lower() == "true"

# Set up PEFT LoRA for fine-tuning.
lora_config = LoraConfig(
    r=64, 
    target_modules= ["k_proj", "v_proj"] if lora_kv_proj else "all-linear",
    task_type="CAUSAL_LM",
)

print("----3----")

# Set up the FSDP arguments
cls_to_wrap = "LlamaDecoderLayer"
fsdp_training_args = {
    "fsdp": "full_shard",
    "fsdp_config": fsdp_v2.get_fsdp_config(cls_to_wrap),
}

no_fsdp = os.environ.get("NO_FSDP", "False").lower() == "true"
fsdp_training_args = {} if no_fsdp else fsdp_training_args

print("fsdp_training_args:", fsdp_training_args)

tokenizer.pad_token = tokenizer.eos_token

no_lora = os.environ.get("NO_LORA", "False").lower() == "true"
peft_config = None if no_lora else lora_config

print("peft_config", peft_config)

# Set up the trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=TrainingArguments(
        per_device_train_batch_size=1,
        num_train_epochs=1,
        max_steps=10,
        save_steps=20,
        eval_steps=20,
        output_dir="./output",
        optim="adafactor",
        logging_steps=1,
        gradient_accumulation_steps=1,
        dataloader_drop_last = True,  # Required for FSDPv2.
        **fsdp_training_args,
    ),
    peft_config=peft_config,
)

print("----4----")

trainer.train()

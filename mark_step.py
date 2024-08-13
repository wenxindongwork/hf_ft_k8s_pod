import torch
import torch_xla.core.xla_model as xm
import torch.nn as nn
import os 
import datetime
from transformers import AutoModelForCausalLM
from optimum.tpu import fsdp_v2
from transformers.optimization import Adafactor
from peft import LoraConfig, get_peft_model
from torch.optim import Adam, AdamW
from torch.profiler import profile, record_function, ProfilerActivity

fsdp_v2.use_fsdp_v2()

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",
    torch_dtype=torch.bfloat16,
    use_cache=True
).to("xla")

config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules="all-linear",
    lora_dropout=0.0,
    bias="none",
)
model = get_peft_model(model, config)

optimizer_kwargs = {"scale_parameter": False, "relative_step": False, "lr": 8e-6}
optimizer = Adafactor([{"params": [p for n, p in model.named_parameters() if p.requires_grad]}], **optimizer_kwargs)

xm.mark_step()

def trace_handler(p):
    p.export_chrome_trace("/tmp/trace_" + str(p.step_num) + ".json")

def run_model(with_mark_step =True):

    with profile(
        activities=[ProfilerActivity.CPU],
        schedule=torch.profiler.schedule(
            wait=0,
            warmup=0,
            repeat=0,
            skip_first=2,
            active=2),
        on_trace_ready=trace_handler
    ) as p:
        for i in range(6):
            inputs = torch.ones([2, 512]).to("xla").to(torch.int64) #two samples of length 512
            labels = torch.ones([2, 512]).to("xla").to(torch.int64)
            
            s = datetime.datetime.now() # with optimizer: 1s without optimizer: 100ms
            output = model(inputs).logits
            output = output.view(-1, output.shape[-1])
            labels = labels.view(-1)
            loss = nn.CrossEntropyLoss()(output, labels)
            print(f"iteration {i} forward ", 1000*(datetime.datetime.now()-s).total_seconds())
            
            s = datetime.datetime.now() # with optimizer: 2s without optimizer: 300ms
            loss.backward()
            print(f"iteration {i} backward ", 1000*(datetime.datetime.now()-s).total_seconds())
            

            if with_mark_step:
                xm.mark_step()
            s = datetime.datetime.now() # with optimizer: 4s 
            optimizer.step()
            optimizer.zero_grad()
            print(f"iteration {i} optimizer ", 1000*(datetime.datetime.now()-s).total_seconds())
            
            s = datetime.datetime.now() #with optimizer: 100ms without optimizer: 70ms
            xm.mark_step()
            print(f"iteration {i} mark_step ", 1000*(datetime.datetime.now()-s).total_seconds())
            p.step()

print("="*10, "with mark step  before optimizer", "="*10) 
run_model(True)
# print("="*10, "without mark step before optimizer", "="*10) 
# run_model(False)

# with mark step  before optimizer, target_modules=["k_proj", "v_proj"]
# wenxin: up_proj_x took 0.095
# wenxin: llama mlp forward out 0.289
# iteration 5 forward  114.829
# iteration 5 backward  128.98399999999998
# iteration 5 optimizer  218.059
# iteration 5 mark_step  28.274

# with mark step  before optimizer, target_modules="all-linear"
# wenxin: up_proj_x took 0.495
# wenxin: llama mlp forward out 1.194
# iteration 5 forward  164.966
# iteration 5 backward  193.659
# iteration 5 optimizer  777.571
# iteration 5 mark_step  131.209

# with mark step  before optimizer, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj","up_proj", "down_proj"]
# wenxin: up_proj_x took 0.324 
# wenxin: llama mlp forward out 1.012
# iteration 5 forward  163.746
# iteration 5 backward  193.178
# iteration 5 optimizer  801.79
# iteration 5 mark_step  85.331

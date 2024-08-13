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

fsdp_v2.use_fsdp_v2()

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
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
optimizer = AdafaActor([{"params": [p for n, p in model.named_parameters() if p.requires_grad]}], **optimizer_kwargs)

xm.mark_step()

# os.environ["XLA_IR_DEBUG"]="1"
# os.environ["XLA_HLO_DEBUG"]="1"
# os.environ["PT_XLA_DEBUG"]="1"

def run_model(with_optimizer=True, with_mark_step =True):
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
        

        if with_optimizer:
            if with_mark_step:
                xm.mark_step()
            s = datetime.datetime.now() # with optimizer: 4s 
            optimizer.step()
            optimizer.zero_grad()
            print(f"iteration {i} optimizer ", 1000*(datetime.datetime.now()-s).total_seconds())
        
        s = datetime.datetime.now() #with optimizer: 100ms without optimizer: 70ms
        xm.mark_step()
        print(f"iteration {i} mark_step ", 1000*(datetime.datetime.now()-s).total_seconds())


print("="*10, "with mark step  before optimizer", "="*10)
run_model(True, True)
print("="*10, "without mark step before optimizer", "="*10)
run_model(True, False)

# ========== with mark step  before optimizer ==========
# iteration 0 forward  324.718
# iteration 0 backward  769.476
# iteration 0 optimizer  901.903
# iteration 0 mark_step  49055.611
# iteration 1 forward  196.799
# iteration 1 backward  188.698
# iteration 1 optimizer  767.1949999999999
# iteration 1 mark_step  51179.277
# iteration 2 forward  152.551
# iteration 2 backward  196.458
# iteration 2 optimizer  756.3480000000001
# iteration 2 mark_step  109.406
# iteration 3 forward  151.733
# iteration 3 backward  197.482
# iteration 3 optimizer  766.4069999999999
# iteration 3 mark_step  110.50200000000001
# iteration 4 forward  150.429
# iteration 4 backward  192.94400000000002
# iteration 4 optimizer  765.9399999999999
# iteration 4 mark_step  113.556
# iteration 5 forward  150.342
# iteration 5 backward  186.578
# iteration 5 optimizer  776.04
# iteration 5 mark_step  79.471
# ========== without mark step before optimizer ==========
# iteration 0 forward  152.851
# iteration 0 backward  189.792
# iteration 0 optimizer  3251.625
# iteration 0 mark_step  107339.76800000001
# iteration 1 forward  1390.739
# iteration 1 backward  2547.762
# iteration 1 optimizer  4536.128
# iteration 1 mark_step  132.858
# iteration 2 forward  1510.004
# iteration 2 backward  2679.471
# iteration 2 optimizer  4678.531999999999
# iteration 2 mark_step  132.129
# iteration 3 forward  1552.573
# iteration 3 backward  2734.174
# iteration 3 optimizer  4747.224
# iteration 3 mark_step  120.027
# iteration 4 forward  1577.532
# iteration 4 backward  2772.966
# iteration 4 optimizer  4797.102
# iteration 4 mark_step  127.726
# iteration 5 forward  1607.616
# iteration 5 backward  2829.4120000000003
# iteration 5 optimizer  4836.818
# iteration 5 mark_step  115.542

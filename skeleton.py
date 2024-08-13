import torch
import torch_xla.core.xla_model as xm
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import torch.nn.functional as F
import torch.nn as nn
import os 
import datetime
from transformers import AutoModelForCausalLM
from optimum.tpu import fsdp_v2
from torch.utils.data import DataLoader
from datasets import Dataset
import numpy as np
from transformers.integrations.tpu import tpu_spmd_dataloader
from accelerate import Accelerator
from transformers.optimization import Adafactor
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers import AutoTokenizer
from peft import LoraConfig, LoraModel, get_peft_model
from torch.optim import AdamW
fsdp_v2.use_fsdp_v2()


def compute_loss(
    model,
    inputs: Dict[str, Union[torch.Tensor, Any]],
    return_outputs=False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
    # if not self.use_dpo_data_collator:
    #     warnings.warn(
    #         "compute_loss is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
    #         "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
    #     )

    # compute_loss_context_manager = torch.cuda.amp.autocast if self._peft_has_been_casted_to_bf16 else nullcontext

    # with compute_loss_context_manager():
    loss, metrics = get_batch_loss_metrics(model, inputs, train_eval="train")
    if return_outputs:
        return (loss, metrics)
    return loss


def odds_ratio_loss(
    policy_chosen_logps: torch.FloatTensor,
    policy_rejected_logps: torch.FloatTensor,
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute ORPO's odds ratio (OR) loss for a batch of policy and reference model log probabilities.

    Args:
        policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)

    Returns:
        A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
        The losses tensor contains the ORPO loss for each example in the batch.
        The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        The log odds ratio of the chosen responses over the rejected responses ratio for logging purposes.
        The `log(sigmoid(log_odds_chosen))` for logging purposes.
    """
    beta = 0.1
    # Derived from Eqs. (4) and (7) from https://arxiv.org/abs/2403.07691 by using log identities and exp(log(P(y|x)) = P(y|x)
    log_odds = (policy_chosen_logps - policy_rejected_logps) - (
        torch.log1p(-torch.exp(policy_chosen_logps)) - torch.log1p(-torch.exp(policy_rejected_logps))
    )
    sig_ratio = F.sigmoid(log_odds)
    ratio = torch.log(sig_ratio)
    losses = beta * ratio

    chosen_rewards = beta * (policy_chosen_logps.to("xla")).detach()
    rejected_rewards = beta * (policy_rejected_logps.to("xla")).detach()
    
    ratio_mean = torch.mean(ratio)
    log_odds_mean = torch.mean(log_odds)
    return losses, chosen_rewards, rejected_rewards, ratio_mean, log_odds_mean

# batch_size = 8

# for i in range(5):
#     s = datetime.datetime.now()
#     print(f"wenxin: iteration {i}")
#     odds_ratio_loss(torch.ones((batch_size,)).to("xla"),torch.ones((batch_size,)).to("xla"))
#     e = datetime.datetime.now()
#     print(f"wenxin: iteration {i}")
#     print(1000*(e-s).total_seconds())



orpo_dataset_dict = {
    "prompt": [
        "hello",
        "how are you",
        "how are you"*2,
        "how are you"*3,

    ] * 100,
    "chosen": [
        "hi nice to meet you",
        "I am fine",
        "I am fine"*2,
        "I am fine"*3,

    ] * 100,
    "rejected": [
        "leave me alone",
        "I am not fine",
        "I am not fine" * 2,
        "I am not fine" * 3,
    ] * 100,
}

def build_tokenized_answer( prompt, answer):
    """
    Llama tokenizer does satisfy `enc(a + b) = enc(a) + enc(b)`.
    It does ensure `enc(a + b) = enc(a) + enc(a + b)[len(enc(a)):]`.
    Reference:
        https://github.com/EleutherAI/lm-evaluation-harness/pull/531#issuecomment-1595586257
    """

    full_tokenized = tokenizer(prompt + answer, add_special_tokens=False)
    prompt_input_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]

    answer_input_ids = full_tokenized["input_ids"][len(prompt_input_ids) :]
    answer_attention_mask = full_tokenized["attention_mask"][len(prompt_input_ids) :]

    # Concat tokens to form `enc(a) + enc(a + b)[len(enc(a)):]`
    full_concat_input_ids = np.concatenate([prompt_input_ids, answer_input_ids])

    # Prepare input tokens for token by token comparison
    full_input_ids = np.array(full_tokenized["input_ids"])

    if len(full_input_ids) != len(full_concat_input_ids):
        raise ValueError("Prompt input ids and answer input ids should have the same length.")

    # On some tokenizers, like Llama-2 tokenizer, there are occasions where tokens
    # can be merged together when tokenizing prompt+answer. This could result
    # on the last token from the prompt being different when tokenized on its own
    # vs when done as prompt+answer.
    response_token_ids_start_idx = len(prompt_input_ids)

    # If tokenized prompt is different than both prompt+answer, then it means the
    # last token has changed due to merging.
    if prompt_input_ids != full_tokenized["input_ids"][:response_token_ids_start_idx]:
        response_token_ids_start_idx -= 1

    prompt_input_ids = full_tokenized["input_ids"][:response_token_ids_start_idx]
    prompt_attention_mask = full_tokenized["attention_mask"][:response_token_ids_start_idx]

    if len(prompt_input_ids) != len(prompt_attention_mask):
        raise ValueError("Prompt input ids and attention mask should have the same length.")

    answer_input_ids = full_tokenized["input_ids"][response_token_ids_start_idx:]
    answer_attention_mask = full_tokenized["attention_mask"][response_token_ids_start_idx:]

    return dict(
        prompt_input_ids=prompt_input_ids,
        prompt_attention_mask=prompt_attention_mask,
        input_ids=answer_input_ids,
        attention_mask=answer_attention_mask,
    )

def tokenize_row( feature) -> Dict:
    """Tokenize a single row from a ORPO specific dataset.

    At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
    in case the prompt + chosen or prompt + rejected responses is/are too long. First
        we truncate the prompt; if we're still too long, we truncate the chosen/rejected.

    We also create the labels for the chosen/rejected responses, which are of length equal to
        the sum of the length of the prompt and the chosen/rejected response, with
        label_pad_token_id  for the prompt tokens.
    """
    batch = {}
    prompt = feature["prompt"]
    chosen = feature["chosen"]
    rejected = feature["rejected"]

    if True:
        # Check issues below for more details
        #  1. https://github.com/huggingface/trl/issues/907
        #  2. https://github.com/EleutherAI/lm-evaluation-harness/pull/531#issuecomment-1595586257
        #  3. https://github.com/LianjiaTech/BELLE/issues/337

        if not isinstance(prompt, str):
            raise ValueError(f"prompt should be an str but got {type(prompt)}")
        prompt_tokens = tokenizer(prompt, add_special_tokens=False)
        prompt_tokens = {f"prompt_{k}": v for k, v in prompt_tokens.items()}

        if not isinstance(chosen, str):
            raise ValueError(f"chosen should be an str but got {type(chosen)}")
        chosen_tokens = build_tokenized_answer(prompt, chosen)

        if not isinstance(rejected, str):
            raise ValueError(f"rejected should be an str but got {type(rejected)}")
        rejected_tokens = build_tokenized_answer(prompt, rejected)

        # Last prompt token might get merged by tokenizer and
        # it should not be included for generation if that happens
        prompt_len_input_ids = len(prompt_tokens["prompt_input_ids"])

        chosen_prompt_len_input_ids = len(chosen_tokens["prompt_input_ids"])
        rejected_prompt_len_input_ids = len(rejected_tokens["prompt_input_ids"])
        prompt_len_input_ids = min(chosen_prompt_len_input_ids, rejected_prompt_len_input_ids)

        for k, v in prompt_tokens.items():
            prompt_tokens[k] = v[:prompt_len_input_ids]

        # Make sure prompts only have one different token at most an
        # and length only differs by 1 at most
        num_diff_tokens = sum(
            [a != b for a, b in zip(chosen_tokens["prompt_input_ids"], rejected_tokens["prompt_input_ids"])]
        )
        num_diff_len = abs(chosen_prompt_len_input_ids - rejected_prompt_len_input_ids)
        if num_diff_tokens > 1 or num_diff_len > 1:
            raise ValueError(
                "Chosen and rejected prompt_input_ids might only differ on the "
                "last token due to tokenizer merge ops."
            )

        # add BOS token to head of prompt. Avoid adding if it's already there
        bos_token_id = tokenizer.bos_token_id
        if prompt_len_input_ids == 0 or bos_token_id != prompt_tokens["prompt_input_ids"][0]:
            prompt_tokens["prompt_input_ids"] = [bos_token_id] + prompt_tokens["prompt_input_ids"]
            prompt_tokens["prompt_attention_mask"] = [1] + prompt_tokens["prompt_attention_mask"]
        if chosen_prompt_len_input_ids == 0 or bos_token_id != chosen_tokens["prompt_input_ids"][0]:
            chosen_tokens["prompt_input_ids"] = [bos_token_id] + chosen_tokens["prompt_input_ids"]
            chosen_tokens["prompt_attention_mask"] = [1] + chosen_tokens["prompt_attention_mask"]
        if rejected_prompt_len_input_ids == 0 or bos_token_id != rejected_tokens["prompt_input_ids"][0]:
            rejected_tokens["prompt_input_ids"] = [bos_token_id] + rejected_tokens["prompt_input_ids"]
            rejected_tokens["prompt_attention_mask"] = [1] + rejected_tokens["prompt_attention_mask"]

        # add EOS token to end of answer. Avoid adding if it's already there
        eos_token_id = tokenizer.eos_token_id
        if len(chosen_tokens["input_ids"]) == 0 or eos_token_id != chosen_tokens["input_ids"][-1]:
            chosen_tokens["input_ids"].append(eos_token_id)
            chosen_tokens["attention_mask"].append(1)
        if len(rejected_tokens["input_ids"]) == 0 or eos_token_id != rejected_tokens["input_ids"][-1]:
            rejected_tokens["input_ids"].append(eos_token_id)
            rejected_tokens["attention_mask"].append(1)

        longer_response_length = max(len(chosen_tokens["input_ids"]), len(rejected_tokens["input_ids"]))


        max_length = 512
        max_prompt_length= 128
        truncation_mode = "keep_start"
        # if combined sequence is too long, truncate the prompt
        for answer_tokens in [chosen_tokens, rejected_tokens, prompt_tokens]:
            if len(answer_tokens["prompt_input_ids"]) + longer_response_length > max_length:
                if truncation_mode == "keep_start":
                    for k in ["prompt_input_ids", "prompt_attention_mask"]:
                        answer_tokens[k] = answer_tokens[k][: max_prompt_length]
                elif truncation_mode == "keep_end":
                    for k in ["prompt_input_ids", "prompt_attention_mask"]:
                        answer_tokens[k] = answer_tokens[k][-max_prompt_length :]
                else:
                    raise ValueError(f"Unknown truncation mode: {truncation_mode}")

        # if that's still too long, truncate the response
        for answer_tokens in [chosen_tokens, rejected_tokens]:
            if len(answer_tokens["prompt_input_ids"]) + longer_response_length > max_length:
                for k in ["input_ids", "attention_mask"]:
                    answer_tokens[k] = answer_tokens[k][: max_length - max_prompt_length]

        label_pad_token_id = -100
        # Create labels
        chosen_sequence_tokens = {
            k: chosen_tokens[f"prompt_{k}"] + chosen_tokens[k] for k in ["input_ids", "attention_mask"]
        }
        rejected_sequence_tokens = {
            k: rejected_tokens[f"prompt_{k}"] + rejected_tokens[k] for k in ["input_ids", "attention_mask"]
        }
        chosen_sequence_tokens["labels"] = chosen_sequence_tokens["input_ids"][:]
        chosen_sequence_tokens["labels"][: len(chosen_tokens["prompt_input_ids"])] = [
            label_pad_token_id
        ] * len(chosen_tokens["prompt_input_ids"])
        rejected_sequence_tokens["labels"] = rejected_sequence_tokens["input_ids"][:]
        rejected_sequence_tokens["labels"][: len(rejected_tokens["prompt_input_ids"])] = [
            label_pad_token_id
        ] * len(rejected_tokens["prompt_input_ids"])

        for k, toks in {
            "chosen_": chosen_sequence_tokens,
            "rejected_": rejected_sequence_tokens,
            "": prompt_tokens,
        }.items():
            for type_key, tokens in toks.items():
                if type_key == "token_type_ids":
                    continue
                batch[f"{k}{type_key}"] = tokens

    # print("before:", batch)
    for k in batch:
        if "labels" in k:
            pad_value = 0
        elif k.endswith("_input_ids"):
            pad_value = 0
        elif k.endswith("_attention_mask"):
            pad_value = 0
        batch[k] = pad_to_length(batch[k], max_length, pad_value=pad_value)
    # print("after:", batch)

    return batch

def pad_to_length(tensor: torch.Tensor, length: int, pad_value: Union[int, float], dim: int = -1) -> torch.Tensor:    
    # import torch_xla.core.xla_model as xm

    # original_device = tensor.device
    # xm.mark_step()
    # tensor = tensor.cpu()
    tensor = torch.Tensor(tensor)
    if tensor.size(dim) >= length:
        return tensor
    else:
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.size(dim)
        return torch.cat(
            [
                tensor,
                pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device = tensor.device),
            ],
            dim=dim,
        )

@staticmethod
def concatenated_inputs(
    batch: Dict[str, Union[List, torch.LongTensor]],
    is_encoder_decoder: bool = False,
    label_pad_token_id: int = -100,
    padding_value: int = 0,
    device: Optional[torch.device] = None,
    max_seq_length: Optional[int] = 512
) -> Dict[str, torch.LongTensor]:
    """Concatenate the chosen and rejected inputs into a single tensor.

    Args:
        batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).
        is_encoder_decoder: Whether the model is an encoder-decoder model.
        label_pad_token_id: The label pad token id.
        padding_value: The padding value to use for the concatenated inputs_ids.
        device: The device for the concatenated inputs.

    Returns:
        A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
    """
    concatenated_batch = {}

    if is_encoder_decoder:
        max_length = max_seq_length #max(batch["chosen_labels"].shape[1], batch["rejected_labels"].shape[1], max_seq_length)
    else:
        max_length = max_seq_length #max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1], max_seq_length)

    for k in batch:
        if k.startswith("chosen") and isinstance(batch[k], torch.Tensor):
            # if "labels" in k or is_encoder_decoder:
            #     pad_value = label_pad_token_id
            # elif k.endswith("_input_ids"):
            #     pad_value = padding_value
            # elif k.endswith("_attention_mask"):
            #     pad_value = 0
            concatenated_key = k.replace("chosen", "concatenated")
            concatenated_batch[concatenated_key] = batch[k]
            # print(k, f"concatenated_batch[{concatenated_key}]",  batch[k].shape, concatenated_batch[concatenated_key].shape,)


    for k in batch:
        if k.startswith("rejected") and isinstance(batch[k], torch.Tensor):
            # if "labels" in k or is_encoder_decoder:
            #     pad_value = label_pad_token_id
            # elif k.endswith("_input_ids"):
            #     pad_value = padding_value
            # elif k.endswith("_attention_mask"):
            #     pad_value = 0
            concatenated_key = k.replace("rejected", "concatenated")
            concatenated_batch[concatenated_key] = torch.cat(
            (
                concatenated_batch[concatenated_key],
                batch[k]
            ),
            dim=0,
        ).to(device=device)
            # print(k, f"concatenated_batch[{concatenated_key}]",  batch[k].shape, concatenated_batch[concatenated_key].shape,)

    # print("concatenated_batch:", concatenated_batch)
    return concatenated_batch

@staticmethod
def get_batch_logps(
    logits: torch.FloatTensor,
    labels: torch.LongTensor,
    average_log_prob: bool = False,
    label_pad_token_id: int = -100,
    is_encoder_decoder: bool = False,
) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.
        label_pad_token_id: The label pad token id.
        is_encoder_decoder: Whether the model is an encoder-decoder model.

    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    if logits.shape[:-1] != labels.shape:
        raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

    if not is_encoder_decoder:
        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
    loss_mask = labels != label_pad_token_id
            
    # dummy token; we'll ignore the losses on these tokens later
    labels = torch.where(labels == label_pad_token_id, 0, labels)

    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)

def concatenated_forward(
    model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

    We do this to avoid doing two forward passes, because it's faster for FSDP.
    """
    label_pad_token_id = -100

    concatenated_batch = concatenated_inputs(
        batch,
        is_encoder_decoder=False,
        label_pad_token_id=label_pad_token_id,
        device="xla",
    )
    len_chosen = batch["chosen_labels"].shape[0]

    # if self.aux_loss_enabled:
    #     model_kwargs["output_router_logits"] = True
    outputs = model(
        concatenated_batch["concatenated_input_ids"], #<==
        attention_mask=concatenated_batch["concatenated_attention_mask"], #<==
        # use_cache=False,
        # **model_kwargs,
    )
    all_logits = outputs.logits #<==

    def cross_entropy_loss(logits, labels):
        # if not self.is_encoder_decoder:
        # Shift so that tokens < n predict n
        logits = logits[..., :-1, :].contiguous()
        labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = nn.CrossEntropyLoss()
        logits = logits.view(-1, logits.shape[-1])
        labels = labels.view(-1)
        # Enable model parallelism
        labels = labels.to(logits.device)
        loss = loss_fct(logits, labels)
        return loss


    # if self.is_encoder_decoder:
    #     labels = concatenated_batch["concatenated_labels"].clone()
    # else:
    labels = concatenated_batch["concatenated_input_ids"].clone()
    attention_mask = concatenated_batch["concatenated_attention_mask"]
    labels = torch.where(attention_mask == 1, labels, label_pad_token_id)

    chosen_nll_loss = cross_entropy_loss(all_logits[:len_chosen], labels[:len_chosen])

    all_logps = get_batch_logps(
        all_logits,
        concatenated_batch["concatenated_labels"],
        average_log_prob=True,
        is_encoder_decoder=False,
    )

    chosen_logps = all_logps[:len_chosen]
    rejected_logps = all_logps[len_chosen:]

    chosen_logits = all_logits[:len_chosen]
    rejected_logits = all_logits[len_chosen:]

    # if self.aux_loss_enabled:
    #     return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_nll_loss, outputs.aux_loss)

    return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_nll_loss)




def get_batch_loss_metrics(
    model,
    batch: Dict[str, Union[List, torch.LongTensor]],
    train_eval: Literal["train", "eval"] = "train",
):
    """Compute the ORPO loss and other metrics for the given batch of inputs for train or test."""
    metrics = {}

    forward_output = concatenated_forward(model, batch)
    (
        policy_chosen_logps,
        policy_rejected_logps,
        policy_chosen_logits,
        policy_rejected_logits,
        policy_nll_loss,
    ) = forward_output[:5]

    # if self.aux_loss_enabled:
    #     aux_loss = forward_output[5]

    losses, chosen_rewards, rejected_rewards, log_odds_ratio, log_odds_chosen = odds_ratio_loss(
        policy_chosen_logps, policy_rejected_logps
    )
    # full ORPO loss
    loss = policy_nll_loss - losses.mean()

    reward_accuracies = (chosen_rewards > rejected_rewards).float()
    reward_margin  = chosen_rewards - rejected_rewards
    # xm.mark_step() # needed because .cpu() calls
    
    prefix = "eval_" if train_eval == "eval" else ""
    # metrics[f"{prefix}rewards/chosen"] = chosen_rewards.cpu().mean()
    # metrics[f"{prefix}rewards/rejected"] = rejected_rewards.cpu().mean()
    # metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.cpu().mean()
    # metrics[f"{prefix}rewards/margins"] = reward_margin.cpu().mean()
    # metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().cpu().mean()
    # metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().cpu().mean()
    # metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.detach().cpu().mean()
    # metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().cpu().mean()
    # metrics[f"{prefix}nll_loss"] = policy_nll_loss.detach().cpu().mean()
    # metrics[f"{prefix}log_odds_ratio"] = log_odds_ratio.item()
    # metrics[f"{prefix}log_odds_chosen"] = log_odds_chosen.item()

    # if self.aux_loss_enabled:
    #     loss += getattr(model.config, "router_aux_loss_coef", 0.0) * aux_loss

    return loss, metrics
################################################################

device = "xla"

accelerator = Accelerator()


dataset = Dataset.from_dict(orpo_dataset_dict)
dataset = dataset.train_test_split(test_size=0.1)
train_dataloader = DataLoader(dataset["train"], batch_size=1, shuffle=True)
train_dataloader = accelerator.prepare(train_dataloader)


tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b",use_cache=True ) #meta-llama/Meta-Llama-3-8B-Instruct



# model = torch.nn.Linear(1, 128).to("xla")
model = AutoModelForCausalLM.from_pretrained(
    # "/usr/share/storage/llama3_8b/",
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

# optimizer

optimizer_kwargs = {"scale_parameter": False, "relative_step": False, "lr": 8e-6}
def get_parameter_names(model, forbidden_layer_types):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result

decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
decay_parameters = [name for name in decay_parameters if "bias" not in name]
optimizer_grouped_parameters = [
    {
        "params": [
            p for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad)
        ],
        "weight_decay": 0.001,
    },
    {
        "params": [
            p for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)
        ],
        "weight_decay": 0.0,
    },
]

optimizer = AdamW(optimizer_grouped_parameters,  lr=8e-6)
###

# os.environ["XLA_IR_DEBUG"]="1"
# os.environ["XLA_HLO_DEBUG"]="1"
# os.environ["PT_XLA_DEBUG"]="1"

i = 0
for batch in train_dataloader:
    tokenized = tokenize_row({"prompt": batch["prompt"][0], "chosen": batch["chosen"][0], "rejected": batch["rejected"][0]})
    for key, value in tokenized.items():
        tokenized[key] = torch.Tensor(value).unsqueeze(0).to("xla").to(torch.int64)# <==
    print(f"iteration {i} ")
    s = datetime.datetime.now() # 2s

    loss = compute_loss(model, tokenized) # loss = model(input)

    print(f"iteration {i} forward ", 1000*(datetime.datetime.now()-s).total_seconds())
    
    s = datetime.datetime.now() # 3s
    accelerator.backward(loss)  # loss.backward()
    print(f"iteration {i} backward ", 1000*(datetime.datetime.now()-s).total_seconds())
    
    s = datetime.datetime.now() # 5s
    optimizer.step()
    optimizer.zero_grad()
    print(f"iteration {i} optimizer ", 1000*(datetime.datetime.now()-s).total_seconds())
    
    s = datetime.datetime.now() #100ms
    xm.mark_step()
    print(f"iteration {i} mark_step ", 1000*(datetime.datetime.now()-s).total_seconds())

    i+=1
    if i == 6: break


# gemma 2b: 

# iteration 3 forward 
# 183.18599999999998
# iteration 3 backward 
# 161.639





# llama3-8b:  vanilla


# iteration 3 forward 
# 149.568
# iteration 3 backward 
# 268.341
# iteration 4 mark_step 
# 70.341

# llama3-8b:   + accelerator.prepare(dataloader)
# iteration 3 forward 
# 84.16
# iteration 3 backward 
# 138.15099999999998
# iteration 3 mark_step 
# 45.879999999999995

# llama3-8b:  + accelerator.backward(loss) 

# iteration 3 forward  151.512
# iteration 3 backward  272.841
# iteration 3 mark_step  72.377


# llama3-8b:  + adafactor optimizer
# iteration 7 forward  1623.4550000000002
# iteration 7 backward  2848.214
# iteration 7 optimizer  4881.584
# iteration 7 mark_step  131.36100000000002

# llama3-8b:  + adam optimizer
# iteration 4 forward  147.471
# iteration 4 backward  195.314
# iteration 4 optimizer  1115.4360000000001
# iteration 4 mark_step  87.065

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


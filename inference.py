import os

import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F
from gpt.model import GPTConfig, GPT
from contextlib import nullcontext

enc = tiktoken.get_encoding("gpt2")
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)

def setup():
    # -----------------------------------------------------------------------------
    init_from = 'gpt2-medium' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
    out_dir = 'out' # ignored if init_from is not 'resume'
    start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
    num_samples = 5 # number of samples to draw
    max_new_tokens = 500 # number of tokens generated in each sample
    temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
    top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
    seed = 1337
    device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
    compile = False # use PyTorch 2.0 to compile the model to be faster
    # exec(open('configurator.py').read()) # overrides from command line or config file
    # -----------------------------------------------------------------------------

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # model
    if init_from == 'resume':
        # init from a model saved in a specific directory
        ckpt_path = os.path.join(out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        gptconf = GPTConfig(**checkpoint['model_args'])
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
    elif init_from.startswith('gpt2'):
        # init from a given GPT-2 model
        model = GPT.from_pretrained(init_from, dict(dropout=0.0))

    model.eval()
    model.to(device)
    if compile:
        model = torch.compile(model) # requires PyTorch 2.0 (optional)

    return model

def batch_generate(model, prompts, desired_tokens):
    assert len(prompts) == len(desired_tokens)
    prompts_and_desired_tokens = list(enumerate(desired_tokens))
    prompts_and_desired_tokens.sort(key=lambda x:x[1], reverse=True) #sort by num tokens
    print(prompts_and_desired_tokens)
    encoded_prompts = enc.encode_batch(prompts)
    stacked = torch.stack([torch.tensor(start_ids, dtype=torch.long, device=device) for start_ids in encoded_prompts])

    results = []
    tokens_generated = 0
    max_desired_tokens = max(desired_tokens)
    for tokens_generated in range(max_desired_tokens + 1):

        if prompts_and_desired_tokens[-1][1] == tokens_generated:
            prompt_id, desired_tokens = prompts_and_desired_tokens[-1]
            prompt_id = prompt_id - len(results) # find index of finished prompt
            results.append(stacked[prompt_id])   # append completion to results array
            stacked = torch.cat((stacked[:prompt_id], stacked[prompt_id + 1:])) # remove finished prompt from batch
            prompts_and_desired_tokens.pop()
        if desired_tokens == max_desired_tokens:
            break

        logits, _ = model(stacked)
        # temp scaling not implemented
        # top k not implemented
        probs = F.softmax(logits, dim=-1)
        probs = torch.reshape(probs, (len(probs), 50257))
        idx_next = torch.multinomial(probs, num_samples=1)
        stacked = torch.cat((stacked, idx_next), dim=1)

    return [decode(result.tolist()) for result in results]
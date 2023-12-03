import os

import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F
from gpt.model import GPTConfig, GPT
from contextlib import nullcontext
from manager import Inference

enc = tiktoken.get_encoding("gpt2")
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)
init_from = 'gpt2-small' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
seed = 1337
device = 'cuda' if torch.cuda.is_available() else 'cpu' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

def ContinuousBatching():
    def __init__(self, model):
          self.model = model

    def modified_attention(self, attn, prompt_lengths, x):
      B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

      # calculate query, key, values for all heads in batch and move head forward to be the batch dim
      q, k, v  = attn.c_attn(x).split(attn.n_embd, dim=2)
      k = k.view(B, T, attn.n_head, C // attn.n_head).transpose(1, 2) # (B, nh, T, hs)
      q = q.view(B, T, attn.n_head, C // attn.n_head).transpose(1, 2) # (B, nh, T, hs)
      v = v.view(B, T, attn.n_head, C // attn.n_head).transpose(1, 2) # (B, nh, T, hs)

      k_split = torch.split(k, prompt_lengths, 2) # split along dim T in chunks of prompt_lengths
      q_split = torch.split(q, prompt_lengths, 2)
      v_split = torch.split(v, prompt_lengths, 2)

      outs = []
      for k, q, v in zip(k_split, q_split, v_split):
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0, is_causal=True) 
        outs.append(y)
 
      y = torch.concat(outs, dim=2)
      y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
      y = attn.resid_dropout(attn.c_proj(y))
      # output projection
      return y

    def batch_inference_forward(self, encoded_tensors, prompt_lengths):
  
      idx = torch.stack([torch.concat(encoded_tensors)])

      b, t = idx.size()

      tok_emb = self.model.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)

      pos_embeddings = []
      for p_len in prompt_lengths:
        pos = torch.arange(0, p_len, dtype=torch.long, device=device) # shape (p_len)
        pos_embed = self.model.transformer.wpe(pos)
        pos_embeddings.append(pos_embed)
  
      pos_embed = torch.concat(pos_embeddings) # shape (t, n_embd)
      x = tok_emb + pos_embed

      for block in model.transformer.h:
          x = x + self.modified_attention(block.attn, prompt_lengths, block.ln_1(x))
          x = x + block.mlp(block.ln_2(x))

      x = model.transformer.ln_f(x)
      x_split = torch.split(x, prompt_lengths, 1)
      output_logits = []
      for x in x_split:
        logits = model.lm_head(x[:,[-1],:])
        output_logits.append(logits)
      return torch.stack(output_logits)
    
    def generate(self, encoded_tensors):
        prompt_lengths = [len(enc) for enc in encoded_tensors]
        logits = self.batch_inference_forward(encoded_tensors, prompt_lengths)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        probs = torch.reshape(probs, (len(probs), 50257))
        idx_next = torch.multinomial(probs, num_samples=1)
        return idx_next

def continuous_batch_generate(batch):
    pass

def static_batch_generate(batch):
    desired_tokens = []
    prompts = []
    job_ids = []
    for inference in batch:
        desired_tokens.append(inference.num_tokens)
        prompts.append(inference.prompt)
        job_ids.append(inference.job_id)

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
            results.append(stacked[prompt_id], job_ids[prompt_id])   # append completion to results array
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

    return [(decode(result.tolist()), job_id) for result, job_id in results]
import os

import tiktoken
import torch
from gpt.model import GPT


def load_base_model_config():
    enc = tiktoken.get_encoding("gpt2")
    init_from = 'gpt2-medium' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
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
    return model, enc, device

class ServerModel:
    def __init__(self, base_gpt, enc, device):
        self.model = base_gpt
        self.enc = enc
        self.device = device
    def decode(self, s):
        return self.enc.decode(s)
    def encode(self, s):
        return self.enc.encode(s, allowed_special={"<|endoftext|>"})

class DynamicBatchingServerModel(ServerModel):
    def __init__(self, model, enc, device):
        super().__init__(model, enc, device)

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

        #   b, t = idx.size()

        tok_emb = self.model.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)

        pos_embeddings = []
        for p_len in prompt_lengths:
            pos = torch.arange(0, p_len, dtype=torch.long, device=self.model.device) # shape (p_len)
            pos_embed = self.model.transformer.wpe(pos)
            pos_embeddings.append(pos_embed)

        pos_embed = torch.concat(pos_embeddings) # shape (t, n_embd)
        x = tok_emb + pos_embed

        for block in self.model.transformer.h:
            x = x + self.modified_attention(block.attn, prompt_lengths, block.ln_1(x))
            x = x + block.mlp(block.ln_2(x))

        x = self.model.transformer.ln_f(x)
        x_split = torch.split(x, prompt_lengths, 1)
        output_logits = []
        for x in x_split:
            logits = self.model.lm_head(x[:,[-1],:])
            output_logits.append(logits)
        return torch.stack(output_logits)
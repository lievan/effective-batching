import torch
from torch.nn import functional as F

LOGGING = False

@torch.no_grad()
def generate(inference, model):
    # completes generation for a single inference
    encoding = model.encode(inference.prompt)
    stacked = torch.stack([torch.tensor(encoding, dtype=torch.long, device=model.device)])
    if LOGGING: print("INFERENCE LOGS: Making inference (no batching): ")
    for _ in range(inference.num_tokens):
        logits, _ = model.model(stacked)
        probs = F.softmax(logits, dim=-1)
        probs = torch.reshape(probs, (len(probs), 50257))
        idx_next = torch.multinomial(probs, num_samples=1)
        stacked = torch.cat((stacked, idx_next), dim=1)
        if LOGGING: print("[T]")
    return [model.decode(tok.tolist()) for tok in stacked][0]

@torch.no_grad()
def static_batch_generate(batch, model):
    # completes generation for batch inference
    desired_tokens = []
    prompts = []
    for inference in batch:
        desired_tokens.append(inference.num_tokens)
        prompts.append(inference.prompt)

    encoded_prompts = model.enc.encode_batch(prompts)
    stacked = torch.stack([torch.tensor(start_ids, dtype=torch.long, device=model.device) for start_ids in encoded_prompts])

    results = []
    tokens_generated = 0
    max_desired_tokens = max(desired_tokens)

    if LOGGING: print("INFERENCE LOGS: Making inference (static batching): ")
    for _ in range(max_desired_tokens):
        logits, _ = model.model(stacked)
        probs = F.softmax(logits, dim=-1)
        probs = torch.reshape(probs, (len(probs), 50257))
        idx_next = torch.multinomial(probs, num_samples=1)
        stacked = torch.cat((stacked, idx_next), dim=1)
        if LOGGING: print("[T]"*len(stacked))
    for res, inference in zip(stacked, batch):
      results.append((model.decode(res.tolist()[:(len(inference.data) + inference.num_tokens)]), inference))
    return results

@torch.no_grad()
def static_batch_generate_v2(batch, model):
    # supports different num_tokens for the batch
    # completes generation for batch inference
    desired_tokens = []
    prompts = []
    for inference in batch:
        desired_tokens.append(inference.num_tokens)
        prompts.append(inference.prompt)

    assert len(prompts) == len(desired_tokens)
    prompts_and_desired_tokens = list(enumerate(desired_tokens))
    prompts_and_desired_tokens.sort(key=lambda x:x[1], reverse=True) #sort by num tokens
    encoded_prompts = model.enc.encode_batch(prompts)
    stacked = torch.stack([torch.tensor(start_ids, dtype=torch.long, device=model.device) for start_ids in encoded_prompts])

    results = []
    tokens_generated = 0
    max_desired_tokens = max(desired_tokens)
    for tokens_generated in range(max_desired_tokens + 1):

        if prompts_and_desired_tokens[-1][1] == tokens_generated:
            prompt_id, desired_tokens = prompts_and_desired_tokens[-1]
            prompt_id = prompt_id - len(results) # find index of finished prompt
            results.append((stacked[prompt_id], batch[prompt_id]))   # append completion to results array
            stacked = torch.cat((stacked[:prompt_id], stacked[prompt_id + 1:])) # remove finished prompt from batch
            prompts_and_desired_tokens.pop()
        if desired_tokens == max_desired_tokens:
            break

        logits, _ = model.model(stacked)
        # temp scaling not implemented
        # top k not implemented
        probs = F.softmax(logits, dim=-1)
        probs = torch.reshape(probs, (len(probs), 50257))
        idx_next = torch.multinomial(probs, num_samples=1)
        stacked = torch.cat((stacked, idx_next), dim=1)
    return [(model.decode(result.tolist()), inference) for result, inference in results]

@torch.no_grad()
def dynamic_batch_generate(next_batch, model):
    if len(next_batch) == 0:
        return None, None

    prompt_lengths = [len(inf.data) for inf in next_batch]

    inference_data = [inf.data for inf in next_batch]
    logits = model.batch_inference_forward(inference_data, prompt_lengths)
    logits = logits[:, -1, :]
    probs = F.softmax(logits, dim=-1)
    probs = torch.reshape(probs, (len(probs), 50257))
    idx_next = torch.multinomial(probs, num_samples=1)
    if LOGGING: print("INFERENCE LOGS: Making inference (dynamic batching): ")
    if LOGGING: print("[T]")
    finished = []
    in_progress = []
    for inference, idx in zip(next_batch, idx_next):
        done = inference.add_token(idx)
        if done:
            finished.append(inference)
        else:
            in_progress.append(inference)
    if LOGGING: print("INFERENCE LOGS: in progress {}, finished {}".format(len(in_progress), len(finished)))
    return finished, in_progress
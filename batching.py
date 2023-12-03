from threading import Thread, Lock, Event
import torch
from torch.nn import functional as F
from inference import Inference

def generate(inference, model):
    # completes generation for a single inference
    encoding = model.encode(inference.prompt)
    stacked = torch.stack([torch.tensor(encoding, dtype=torch.long, device=model.device)])
    for _ in range(inference.num_tokens):
        logits, _ = model(stacked)
        probs = F.softmax(logits, dim=-1)
        probs = torch.reshape(probs, (len(probs), 50257))
        idx_next = torch.multinomial(probs, num_samples=1)
        stacked = torch.cat((stacked, idx_next), dim=1)
    return model.decode(stacked.tolist())

def static_batch_generate(batch, model):
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
            results.append(stacked[prompt_id], batch[prompt_id])   # append completion to results array
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

    return [(model.decode(result.tolist()), inference) for result, inference in results]



def dynamic_batch_generate(self, next_batch, model):
    if len(next_batch) == 0:
        return None, None

    prompt_lengths = [len(inf.data) for inf in next_batch]
    inference_data = [inf.data for inf in next_batch]

    logits = model.batch_inference_forward(inference_data, prompt_lengths)
    logits = logits[:, -1, :]
    probs = F.softmax(logits, dim=-1)
    probs = torch.reshape(probs, (len(probs), 50257))
    idx_next = torch.multinomial(probs, num_samples=1)

    finished = []
    in_progress = []
    for inference, idx in zip(next_batch, idx_next):
        done = inference.add_token(idx)
        if done:
            finished.append(inference)
        else:
            in_progress.append(inference)
    return finished, in_progress

class BatchingManager:
    def __init__(self, model):
        self.queue_mutex = Lock()
        self.queue = {} # queue needs to track sizes 
        self.running_inference = Lock()
        self.simple_id = 0
        self.inferences = {}
        self.model = model

    def enqueue(self, prompt, num_tokens):
        new_inference = None
        with self.queue_mutex:
            new_inference = Inference(self.simple_id, prompt, num_tokens)
            self.queue.append(new_inference)
            self.simple_id += 1
        return new_inference

    def no_batching_loop(self):
        while True:
            next_batch = []
            with self.queue_mutex:
                next_batch = self.queue
                self.queue = []
            if next_batch:
                for inference in next_batch:
                    completion = generate(inference, self.model)
                    inference.finished_with(completion)

    def static_batching_loop(self):
        while True:
            next_batch = []
            results = []
            with self.queue_mutex:
                next_batch = self.queue
                self.queue = []
            if next_batch:
                results = static_batch_generate(next_batch, self.model)
                for completion, inference in results:
                    inference.finished_with(completion)

    def dynamic_batching_loop(self):
        waiting = []
        while True:
            with self.queue_mutex:
                next_batch = self.queue
                self.queue = []
                waiting += next_batch
            if waiting:
                finished, in_progress = dynamic_batch_generate(waiting, self.model)
                for result in finished:
                    result.finished()
                    waiting = in_progress
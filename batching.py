from threading import Lock
from collections import defaultdict
import time
import torch
from threading import Event

MAX_BATCH_SIZE = 32

class Inference:
    def __init__(self, job_id, prompt, num_tokens, enc, device):
        self.completion = None
        self.job_id = job_id
        self.enc = enc
        self.prompt = prompt
        self.data = torch.tensor(self.enc.encode(prompt), dtype=torch.long, device=device)
        self.num_tokens = num_tokens
        self.counter = 0 # counter to keep track of when prompt is finished
        self.event_obj = Event()

    def add_token(self, token):
      self.counter += 1
      self.data = torch.cat((self.data, token))
      if self.counter == self.num_tokens:
        return True
      return False

    def finished(self):
        self.completion = self.enc.decode(self.data.tolist())
        self.event_obj.set()

    def finished_with(self, completion):
        self.completion = completion
        self.event_obj.set()

    def wait_for_completion(self):
        self.event_obj.wait(1000)
        return self.completion

class BatchingManager:
    def __init__(self, model, generation_fn):
        self.queue_mutex = Lock()
        self.queue = []
        self.running_inference = Lock()
        self.simple_id = 0
        self.inferences = {}
        self.model = model
        self.generation_fn = generation_fn

    def enqueue(self, prompt, num_tokens):
        new_inference = None
        with self.queue_mutex:
            new_inference = Inference(self.simple_id, prompt, num_tokens, 
                                    self.model.enc, self.model.device)
            self.queue.append(new_inference)
            self.simple_id += 1
        return new_inference

    def no_batching_loop(self):
        self.model.model.to(self.model.device)
        while True:
            next_batch = []
            with self.queue_mutex:
                if len(self.queue) > MAX_BATCH_SIZE:
                    next_batch = self.queue[:MAX_BATCH_SIZE]
                    self.queue = self.queue[MAX_BATCH_SIZE:]
                else:
                    next_batch = self.queue
                    self.queue = []
            if next_batch:
                print("SERVER LOGS: Loop handling {} requests".format(len(next_batch)))
                for inference in next_batch:
                    completion = self.generation_fn(inference, self.model)
                    inference.finished_with(completion)
            time.sleep(0.01)

    def static_batching_loop(self):
        self.model.model.to(self.model.device)
        while True:
            next_batch = []
            results = []
            with self.queue_mutex:
                if len(self.queue) > MAX_BATCH_SIZE:
                    next_batch = self.queue[:MAX_BATCH_SIZE]
                    self.queue = self.queue[MAX_BATCH_SIZE:]
                else:
                    next_batch = self.queue
                    self.queue = []

            if next_batch:
                print("SERVER LOGS: Loop handling {} requests".format(len(next_batch)))
                sizes = defaultdict(list)
                for item in next_batch:
                    sizes[(len(item.data), item.num_tokens)].append(item)
                for _, batch in sizes.items():
                    results = self.generation_fn(batch, self.model)
                    for completion, inference in results:
                        inference.finished_with(completion)                         
            time.sleep(0.01)

    def dynamic_batching_loop(self):
        self.model.model.to(self.model.device)
        waiting = []
        next_batch = []
        while True:
            with self.queue_mutex:
                waiting += self.queue
                self.queue = []
                if len(waiting) > MAX_BATCH_SIZE:
                    next_batch = waiting[:MAX_BATCH_SIZE]
                    waiting = waiting[MAX_BATCH_SIZE:]
                else:
                    next_batch = waiting
            if next_batch:
                finished, in_progress = self.generation_fn(next_batch, self.model)
                for result in finished:
                    result.finished()
                    waiting += in_progress
            time.sleep(0.01)
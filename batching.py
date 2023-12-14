from threading import Lock
from collections import defaultdict
import time
import torch
from threading import Event

MAX_BATCH_SIZE = 128

class Inference:
    # a shared object used for nobatch, static, and dynamic batching
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
                next_batch = self.queue
                self.queue = []

            if next_batch:
                print("SERVER LOGS: Loop handling {} requests".format(len(next_batch)))
                sizes = defaultdict(list)
                for item in next_batch:
                    sizes[(len(item.data), item.num_tokens)].append(item)
                for _, batch in sizes.items():

                    while len(batch) > MAX_BATCH_SIZE:
                        inference_batch = batch[:MAX_BATCH_SIZE]
                        batch = batch[MAX_BATCH_SIZE:]
                        results = self.generation_fn(inference_batch, self.model)
                        for completion, inference in results:
                            inference.finished_with(completion)

                    results = self.generation_fn(batch, self.model)
                    for completion, inference in results:
                        inference.finished_with(completion)
      
            time.sleep(0.01)

    def dynamic_batching_loop(self):
        self.model.model.to(self.model.device)
        waiting = []
        while True:
            with self.queue_mutex:
                space = MAX_BATCH_SIZE - len(waiting)
                if len(self.queue) > space:
                    waiting += self.queue[:space]
                    self.queue = self.queue[space:]
                else:
                    waiting += self.queue
                    self.queue = []
            if waiting:
                finished, in_progress = self.generation_fn(waiting, self.model)
                for result in finished:
                    result.finished()
                    waiting = in_progress
            time.sleep(0.01)
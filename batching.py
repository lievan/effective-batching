from threading import Lock
from collections import defaultdict
import time
import torch
from threading import Event
import tiktoken

class Inference:
    def __init__(self, job_id, prompt, num_tokens, enc, device):
        self.completion = None
        self.job_id = job_id
        self.enc = enc
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
        print("waiting for completion for job id {}".format(self.job_id))
        self.event_obj.wait(1000)
        return self.completion

class BatchingManager:
    def __init__(self, model, generation_fn):
        self.queue_mutex = Lock()
        self.queue = {} # queue needs to track sizes 
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
        while True:
            next_batch = []
            with self.queue_mutex:
                next_batch = self.queue
                self.queue = []
            if next_batch:
                for inference in next_batch:
                    completion = self.generation_fn(inference, self.model)
                    inference.finished_with(completion)

    def static_batching_loop(self):
        while True:
            next_batch = []
            results = []
            with self.queue_mutex:
                next_batch = self.queue
                self.queue = []
    
            if next_batch:
                input_lengths = defaultdict(list)
                for inference in next_batch:
                    input_lengths[len(inference.data)].append(inference)
                for _, batch in input_lengths.items():
                    results =  self.generation_fn(batch, self.model)
                    for completion, inference in results:
                        inference.finished_with(completion)
            time.sleep(0.1) # wait 0.1 seconds to collect the next batch

    def dynamic_batching_loop(self):
        waiting = []
        while True:
            with self.queue_mutex:
                next_batch = self.queue
                self.queue = []
                waiting += next_batch
            if waiting:
                finished, in_progress = self.generation_fn(waiting, self.model)
                for result in finished:
                    result.finished()
                    waiting = in_progress
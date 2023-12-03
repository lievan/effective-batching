import torch
from threading import Event
from model_init import enc, device

class Inference:
    def __init__(self, job_id, prompt, num_tokens):
        self.completion = None
        self.job_id = job_id
        self.data = torch.tensor(enc.encode(prompt), dtype=torch.long, device=device)
        self.num_tokens = num_tokens
        self.counter = 0 # counter to keep track of when prompt is finished
        self.event_obj = Event()

    def add_token(self, token) -> bool:
      self.counter += 1
      self.data = torch.cat((self.data, token))
      if self.counter == self.num_tokens:
        return True
      return False

    def finished(self):
        self.completion = enc.decode(self.data.tolist())
        self.event_obj.set()

    def finished_with(self, completion):
        self.completion = completion
        self.event_obj.set()

    def wait_for_completion(self):
        print("waiting for completion for job id {}".format(self.job_id))
        self.event_obj.wait(1000)
        return self.completion
from numpy import random
import math
from threading import Lock
import random
class PromptData:

  def __init__(self, num_samples):
      self.prompts = []
      print("Generating prompt samples...")
      self.generate_samples_buckets()
      self.prompt_idx = 0
      self.prompt_idx_lock = Lock()

  def get_next_sample(self):
      next_sample = None
      with self.prompt_idx_lock:
          if self.prompt_idx < len(self.prompts):
            next_sample = self.prompts[self.prompt_idx]
            self.prompt_idx += 1
      return next_sample

  def generate_samples_range(self, num_samples, num_tokens_range):
      for i in range(num_samples):
          target = int(random.choice(num_tokens_range))
          prompt_len = target
          num_tokens = target
          prompt = ""
          for i in range(prompt_len):
            prompt += "{} ".format(i)
          prompt = prompt.strip()
          self.prompts.append((prompt, num_tokens))

  def generate_samples_normal(self, num_samples):
      possible_num_tokens = random.normal(loc=100, size=(num_samples), scale=150)
      possible_prompt_len = random.normal(loc=100, size=(num_samples), scale=150)

      for prompt_len, num_tokens in zip(possible_num_tokens, possible_prompt_len):
          prompt_len = max(1, abs(math.ceil(prompt_len)))
          num_tokens = max(1, abs(math.ceil(num_tokens)))
          prompt = ""
          for i in range(prompt_len):
            prompt += "{} ".format(i)
          prompt = prompt.strip()
          self.prompts.append((prompt, num_tokens))
          
  def generate_samples_buckets(self):
      buckets = [i for i in range(5, 150, 5)]
      vals = []
      for bucket in buckets:
            for _ in range(10):
              vals.append(bucket)
      random.shuffle(vals)
      for num_tokens in vals:
          prompt = ""
          self.prompts.append((prompt, num_tokens))
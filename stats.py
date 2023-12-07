# file to track experiment statistics
import torch
import time
from threading import Lock

cuda_time = torch.cuda.is_available()

def get_time():
    if cuda_time:
        torch.cuda.synchronize()
        return time.perf_counter()
    return time.time()

class ServerStats:

    def __init__(self):
        self.stats_lock = Lock()
        self.total_tokens = 0
        self.total_elapsed = 0
        self.total_requests = 0
        self.total_latency = 0
        self.table = {}

        self.rid_lock = Lock()
        self.rid = 0

    def get_new_rid(self):
        new_rid = -1
        with self.rid_lock:
            new_rid = self.rid
            self.rid += 1
        return new_rid

    def start_request(self, num_tokens):
        new_rid = self.get_new_rid()
        self.table[new_rid] = (get_time(), num_tokens)
        return new_rid

    def finish_request(self, rid):
        elapsed = get_time() - self.table[rid][0]
        tokens = self.table[rid][1]
        with self.stats_lock:
            self.total_tokens += tokens
            self.total_elapsed += elapsed
            self.total_requests += 1
            self.total_latency += elapsed/tokens

    def latency_per_token(self):
        # average time a user has to wait for an individual token
        ret = None
        if self.total_requests <= 0:
            return ret
        with self.stats_lock:
            ret = self.total_latency/self.total_requests
        return ret

    def throughput(self):
        # the number of output tokens per second the inference server generates
        ret = None
        if self.total_elapsed <= 0:
            return ret
        with self.stats_lock:
            ret = self.total_tokens/self.total_elapsed
        return ret
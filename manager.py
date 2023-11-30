from threading import Thread, Lock, Event
class Inference:
    def __init__(self, job_id, prompt, num_tokens):
        self.completion = None
        self.job_id = job_id
        self.prompt = prompt
        self.num_tokens = num_tokens
        self.event_obj = Event()

    def finished(self, completion):
        self.completion = completion
        self.event_obj.set()

    def wait_for_completion(self):
        print("waiting for completion for job id {}".format(self.job_id))
        self.event_obj.wait(1000)
        return self.completion

class InferenceManager:
    def __init__(self):
        self.queue_mutex = Lock()
        self.queue = []
        self.running_inference = Lock()
        self.simple_id = 0
        self.inferences = {}

    def enqueue(self, prompt, num_tokens):
        job_id = None
        new_inference = None
        with self.queue_mutex:
            job_id = self.simple_id
            new_inference = Inference(job_id, prompt, num_tokens)
            self.queue.append(new_inference)
            self.inferences[job_id] = new_inference
            self.simple_id += 1
        return new_inference

    def trigger_batch(self):
        while True:
            next_batch = []
            results = []
            with self.queue_mutex:
                next_batch = self.queue
                self.queue = []
            if next_batch:
                results = batch_generate(next_batch)
                for result in results:
                    completion, job_id = result
                    self.inferences[job_id].finished(completion)
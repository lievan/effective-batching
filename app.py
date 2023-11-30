import json
import time

from flask import Flask, request
#from inference import batch_generate
from threading import Thread, Lock, Event

app = Flask(__name__)

def batch_generate(next_batch):
    # stub for testing
    time.sleep(1)
    print("running batch_generate")
    results = []
    for inference in next_batch:
        results.append([inference.prompt, inference.job_id])
    return results

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

manager = InferenceManager()
run_inferences = Thread(target=manager.trigger_batch)
run_inferences.start()

@app.route('/inference', methods=['POST'])
def hello():
    data = json.loads(request.get_data())
    prompt = data['prompt']
    num_tokens = int(data['num_tokens'])
    assert isinstance(prompt, str)
    inference = manager.enqueue(prompt, num_tokens)
    print("== INFERENCE ENQ ===")
    completion = inference.wait_for_completion()
    return {'completion': completion}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=105, debug=True, threaded=True)
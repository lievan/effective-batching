import time
import requests
from threading import Thread
from data import PromptData

NUM_SAMPLES=10

prompt_data = PromptData(num_samples=NUM_SAMPLES)

def inference_request(prompt, num_tokens):
    inference_req = {'prompt':prompt, 'num_tokens':num_tokens}
    r = requests.post('http://127.0.0.1:105/inference', json=inference_req)
    print(r.json()['completion'])


threads = []
for i in range(NUM_SAMPLES):
    req, num_tokens = prompt_data.get_next_sample()
    run_inference = Thread(target=inference_request, kwargs={"prompt": "request {}".format(i), "num_tokens": num_tokens})
    threads.append(run_inference)

for thread in threads:
    thread.start()
    time.sleep(0.1)

for thread in threads:
    thread.join()

r = requests.get('http://127.0.0.1:105/stats')
print("\n\n=== END STATS===\n\n")
print(r.json())



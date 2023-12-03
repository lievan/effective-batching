import time
import requests
from threading import Thread, Lock, Event

def inference_request(prompt, num_tokens):
    print("making request")
    inference_req = {'prompt':prompt, 'num_tokens':num_tokens}
    r = requests.post('http://127.0.0.1:105/inference', json=inference_req)
    print(r.json()['completion'])

for i in range(100):
    run_inference = Thread(target=inference_request, kwargs={"prompt": "request {}".format(i), "num_tokens": i})
    run_inference.start()
    time.sleep(0.5)
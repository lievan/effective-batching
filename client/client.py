import time
import os
import requests
import random

from threading import Thread
from data import PromptData
from dotenv import load_dotenv

load_dotenv()

KEY = os.getenv('KEY')
IP = os.getenv('IP')

NUM_SAMPLES=100

prompt_data = PromptData(num_samples=NUM_SAMPLES)

def inference_request(prompt, num_tokens, rid):
    print("LOGS: Request {} with ~{} prompt len, {} tokens".format(rid, len(prompt.split(' ')), num_tokens))
    auth_headers = {'Authorization':KEY}
    inference_req = {'prompt':prompt, 'num_tokens':num_tokens}
    r = requests.post('http://{}:8500/inference'.format(IP), json=inference_req, headers=auth_headers)
    print("="*50)
    print("LOGS: Got completion for request {}".format(rid))
    print(r.json()['completion'])
    print("="*50)
    print("")

threads = []
for i in range(NUM_SAMPLES):
    prompt, num_tokens = prompt_data.get_next_sample()
    run_inference = Thread(target=inference_request, kwargs={"rid": i, "prompt": prompt, "num_tokens": num_tokens})
    run_inference.start()
    threads.append(run_inference)
    if i % 5:
        wait = random.random()
        time.sleep(wait * 5)

for thread in threads:
    thread.join()

r = requests.get('http://{}:8500/stats'.format(IP))
print("\n\n\n======== END STATS ========\n\n\n")
stats = r.json()
print(stats)
with open('stats.txt', 'w') as f:
    f.write(str(stats))




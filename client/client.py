import time
import os
import requests
import random

from threading import Thread
from data import PromptData
from dotenv import load_dotenv
from collections import defaultdict
import argparse

load_dotenv()

parser = argparse.ArgumentParser(description='Test Client')
parser.add_argument('--numsamples', default=100, type=int, help='number of requests to make')
parser.add_argument('--prompt', default=None, type=str, help='prompt to make an inference on')
parser.add_argument('--numtokens', default=None, type=int, help='number of tokens to generate')


args = parser.parse_args()

IP = os.getenv('IP')

data_based_on_token = defaultdict(list)

def inference_request(prompt, num_tokens, rid):
    print("LOGS: Request {} with ~{} prompt len, {} tokens".format(rid, len(prompt.split(' ')), num_tokens))
    inference_req = {'prompt':prompt, 'num_tokens':num_tokens}
    time_start = time.time()
    r = requests.post('http://{}:8500/inference'.format(IP), json=inference_req)
    data_based_on_token[num_tokens].append(time.time() - time_start)
    print("="*50)
    print("LOGS: Got completion for request {}".format(rid))
    print(r.json()['completion'])
    print("="*50)
    print("")


def launch_requests():
    NUM_SAMPLES=args.numsamples

    prompt_data = PromptData(num_samples=NUM_SAMPLES)

    threads = []
    for i in range(NUM_SAMPLES):
        prompt, num_tokens = prompt_data.get_next_sample()
        run_inference = Thread(target=inference_request, kwargs={"rid": i, "prompt": prompt, "num_tokens": num_tokens})
        run_inference.start()
        threads.append(run_inference)
        if i % 5 == 0:
            wait = random.random()
            time.sleep(wait * 5)

    for thread in threads:
        thread.join()

    r = requests.get('http://{}:8500/stats'.format(IP))
    print("\n\n\n======== END STATS ========\n\n\n")
    stats = r.json()
    print(stats)
    with open('results/stats.txt', 'w') as f:
        f.write(str(stats))

if args.prompt and args.numtokens:
    print("Making single inference, numsamples arg ignored if specified")
    inference_request(args.prompt, args.numtokens, 0)
else:
    launch_requests()
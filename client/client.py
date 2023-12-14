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

parser = argparse.ArgumentParser(description="Test Client")
parser.add_argument(
    "--wait",
    default=1,
    type=int,
    help="sleep a random amount between 0 and wait (s) in between requests",
)
parser.add_argument(
    "--numsamples", default=100, type=int, help="number of requests to make"
)
parser.add_argument(
    "--numtokens_range",
    nargs="*",
    default=[1, 25, 100],
    type=int,
    help="possible # of tokens",
)
parser.add_argument(
    "--selection", default="random", type=str, help="selection strategy for num tokens"
)
parser.add_argument(
    "--prompt", default=None, type=str, help="prompt to make an inference on"
)
parser.add_argument(
    "--numtokens", default=None, type=int, help="number of tokens to generate"
)


args = parser.parse_args()

IP = os.getenv("IP")

data_based_on_token = defaultdict(list)


def inference_request(prompt, num_tokens, rid):
    print(
        "LOGS: Request {} with ~{} prompt len, {} tokens".format(
            rid, len(prompt.split(" ")), num_tokens
        )
    )
    inference_req = {"prompt": prompt, "num_tokens": num_tokens}
    time_start = time.time()
    r = requests.post("http://{}:8500/inference".format(IP), json=inference_req)
    data_based_on_token[num_tokens].append(time.time() - time_start)
    print("=" * 50)
    print("LOGS: Got completion for request {}".format(rid))
    print(r.json()["completion"])
    print("=" * 50)
    print("")


def launch_requests():
    num_samples = args.numsamples
    num_tokens_range = args.numtokens_range
    selection = args.selection
    print(selection)
    prompt_data = PromptData(
        num_samples=num_samples, num_tokens_range=num_tokens_range, selection=selection
    )

    threads = []
    for i in range(num_samples):
        prompt, num_tokens = prompt_data.get_next_sample()
        run_inference = Thread(
            target=inference_request,
            kwargs={"rid": i, "prompt": prompt, "num_tokens": num_tokens},
        )
        run_inference.start()
        threads.append(run_inference)
        wait = random.random()
        time.sleep(wait * args.wait)
        # if i % 5 == 0:
        #     wait = random.random()
        #     time.sleep(wait * 5)

    for thread in threads:
        thread.join()

    r = requests.get("http://{}:8500/stats".format(IP))
    print("\n\n\n======== END STATS ========\n\n\n")
    stats = r.json()
    print(stats)
    with open("results/stats.txt", "w") as f:
        f.write(str(stats))


if args.prompt and args.numtokens:
    print("Making single inference, numsamples arg ignored if specified")
    inference_request(args.prompt, args.numtokens, 0)
else:
    launch_requests()

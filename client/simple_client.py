import time
import requests
from threading import Thread
from data import PromptData
import os
from dotenv import load_dotenv

load_dotenv()

KEY = os.getenv('KEY')

NUM_SAMPLES=10

prompt_data = PromptData(num_samples=NUM_SAMPLES)

def inference_request(prompt, num_tokens):
    auth_headers = {'Authorization':"ad"}
    inference_req = {'prompt':prompt, 'num_tokens':num_tokens}
    r = requests.post('http://34.125.75.199:8500/inference', json=inference_req, headers=auth_headers)
    print(r.json()['completion'])

inference_request("hi", 10)

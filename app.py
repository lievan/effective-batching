import json
import time
import torch
import argparse
import os

from flask import Flask, request
from batching import BatchingManager
from threading import Thread
from stats import ServerStats
from model import load_base_model_config, ServerModel, DynamicBatchingServerModel
import tiktoken

from generate.generate import static_batch_generate, generate, dynamic_batch_generate
from generate.generate_mock import mock_generate, mock_dynamic_batch_generate, mock_static_batch_generate

from dotenv import load_dotenv

load_dotenv()

KEY = os.getenv('KEY')

BATCHING = os.getenv("BATCHING")

mock = False

app = Flask(__name__)

server_stats = ServerStats()

@app.route('/stats', methods=['GET'])
def stats():
    print("SERVER LOGS: Handling stats request")
    latency_per_token = server_stats.latency_per_token()
    throughput = server_stats.throughput()
    with open('data/token_data_{}.csv'.format(BATCHING), 'w') as f:
        for data_len, lst in server_stats.token_breakdown().items():
            f.write("{},".format(data_len))
            f.write("{}".format(sum(lst)/len(lst)))
            f.write("\n")
    return {'latency-per-token': latency_per_token, 'throughput': throughput,
            'total-tokens-handled': server_stats.total_tokens, 'total-elapsed-time': server_stats.total_elapsed}

@app.route('/', methods=['POST'])
def home():
    return 'hello use /inference endpoint for inferences and /stats to get infrence stats'

@app.route('/inference', methods=['POST'])
def inference():
    # request processing
    key = request.headers.get('Authorization')
    if key != KEY:
        print("SERVER LOGS: Got an inference request with incorrect auth key")
        return 'no' # change l8ter

    data = json.loads(request.get_data())

    prompt = data['prompt']
    num_tokens = int(data['num_tokens'])
    assert isinstance(prompt, str)

    print("SERVER LOGS: NEW, prompt len ~{} | requesting {} tokens".format(len(prompt.split(' ')), num_tokens))
    # make inference
    rid = server_stats.start_request(num_tokens)
    inference = manager.enqueue(prompt, num_tokens)
    completion = inference.wait_for_completion()
    server_stats.finish_request(rid)
    print("SERVER LOGS: FINISHED, prompt len ~{} | requesting {} tokens".format(len(prompt.split(' ')), num_tokens))
    return {'completion': completion}

print("SERVER LOGS: Launching with batching strategy of ({})".format(BATCHING))
gpt_model, enc, device = None, None, None

if not mock:
    gpt_model, enc, device = load_base_model_config()
    gpt_model.eval()
    gpt_model.to(device)
else:
    gpt_model = None
    enc = tiktoken.get_encoding("gpt2")
    device = "cpu"

if BATCHING == "nobatch":
    model = ServerModel(gpt_model, enc, device)
    generate = generate if not mock else mock_generate
    manager = BatchingManager(model, generate)
    run_inferences = Thread(target=manager.no_batching_loop)
elif BATCHING == "static":
    model = ServerModel(gpt_model, enc, device)
    static_batch_generate = static_batch_generate if not mock else mock_static_batch_generate
    manager = BatchingManager(model, static_batch_generate)
    run_inferences = Thread(target=manager.static_batching_loop)
elif BATCHING == "dynamic":
    model = DynamicBatchingServerModel(gpt_model, enc, device)
    dynamic_batch_generate = dynamic_batch_generate if not mock else mock_dynamic_batch_generate
    manager = BatchingManager(model, dynamic_batch_generate)
    run_inferences = Thread(target=manager.dynamic_batching_loop)
run_inferences.start()
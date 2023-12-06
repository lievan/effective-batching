import json
import time
import torch
import argparse

from flask import Flask, request
from batching import BatchingManager
from threading import Thread
from stats import ServerStats
from model import load_base_model_config, ServerModel, DynamicBatchingServerModel
import tiktoken

from generate.generate import static_batch_generate, generate, dynamic_batch_generate
from generate.generate_mock import mock_generate, mock_dynamic_batch_generate, mock_static_batch_generate

app = Flask(__name__)

server_stats = ServerStats()

@app.route('/stats', methods=['GET'])
def stats():
    latency_per_token = server_stats.latency_per_token()
    throughput = server_stats.throughput()
    return {'latency-per-token': latency_per_token, 'throughput': throughput}

@app.route('/', methods=['POST'])
def home():
    return 'hello'

@app.route('/inference', methods=['POST'])
def inference():
    # request processing
    data = json.loads(request.get_data())
    prompt = data['prompt']
    num_tokens = int(data['num_tokens'])
    assert isinstance(prompt, str)

    # make inference
    rid = server_stats.start_request(num_tokens)
    inference = manager.enqueue(prompt, num_tokens)
    print("request queued")
    completion = inference.wait_for_completion()
    server_stats.finish_request(rid)

    return {'completion': completion}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="batching experiment app")
    parser.add_argument('--batching', default='off', type=str, help='batching strategy, one of (off, dynamic, static)')
    parser.add_argument('--mock', default=False, type=bool, help='use a mock model')

    args = parser.parse_args()

    gpt_model, enc, device = None, None, None

    if not args.mock:
        gpt_model, enc, device = load_base_model_config()
        gpt_model.eval()
        gpt_model.to(device)
    else:
        gpt_model = None
        enc = tiktoken.get_encoding("gpt2")
        device = "cpu"

    if args.batching == "off":
        model = ServerModel(gpt_model, enc, device)
        generate = generate if not args.mock else mock_generate
        manager = BatchingManager(model, generate)
        run_inferences = Thread(target=manager.no_batching_loop)
    elif args.batching == "static":
        model = ServerModel(gpt_model, enc, device)
        static_batch_generate = static_batch_generate if not args.mock else mock_static_batch_generate
        manager = BatchingManager(model, static_batch_generate)
        run_inferences = Thread(target=manager.static_batching_loop)
    elif args.batching == "dynamic":
        model = DynamicBatchingServerModel(gpt_model, enc, device)
        dynamic_batch_generate = dynamic_batch_generate if not args.mock else mock_dynamic_batch_generate
        manager = BatchingManager(model, dynamic_batch_generate)
        run_inferences = Thread(target=manager.dynamic_batching_loop)
    run_inferences.start()


    app.run(host='0.0.0.0', port=8500, debug=True, threaded=True)

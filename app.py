import json
import time
import torch
import argparse

from model_init import load_base_model_config, ServerModel, DynamicBatchingServerModel
from flask import Flask, request
from batching import BatchingManager
from inference import Inference
#from inference import batch_generate
from threading import Thread
from stats import ServerStats

app = Flask(__name__)

gpt_model, enc, device = load_base_model_config()
gpt_model.eval()
gpt_model.to(device)
if compile:
    gpt_model = torch.compile(gpt_model) # requires PyTorch 2.0 (optional)

server_stats = ServerStats()


@app.route('/stats', methods=['GET'])
def inference():
    latency_per_token = server_stats.latency_per_token()
    throughput = server_stats.throughput()
    return {'latency-per-token': latency_per_token, 'throughput': throughput}

@app.route('/inference', methods=['POST'])
def inference():
    # request processing
    data = json.loads(request.get_data())
    prompt = data['prompt']
    num_tokens = int(data['num_tokens'])
    assert isinstance(prompt, str)

    # make inference
    server_stats.start_request()
    inference = manager.enqueue(prompt, num_tokens)
    completion = inference.wait_for_completion()
    server_stats.finish_request()

    return {'completion': completion}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="batching experiment app")
    parser.add_argument('--batching', default='off', type=str, help='batching strategy, one of (off, dynamic, static)')
    args = parser.parse_args()
    if args.batching == "off":
        model = ServerModel(gpt_model, enc, device)
        manager = BatchingManager(model)
        run_inferences = Thread(target=manager.no_batching_loop)
    elif args.batching == "static":
        model = DynamicBatchingServerModel(gpt_model, enc, device)
        manager = BatchingManager(model)
        run_inferences = Thread(target=manager.static_batching_loop)
    elif args.batching == "dynamic":
        model = ServerModel(gpt_model, enc, device)
        manager = BatchingManager(model)
        run_inferences = Thread(target=manager.dynamic_batching_loop)
    run_inferences.start()
    app.run(host='0.0.0.0', port=105, debug=True, threaded=True)
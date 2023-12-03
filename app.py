import json
import time
import torch

from model_init import load_base_model_config, ServerModel, DynamicBatchingServerModel
from flask import Flask, request
from batching import BatchingManager
from inference import Inference
#from inference import batch_generate
from threading import Thread, Lock, Event

app = Flask(__name__)

gpt_model, enc, device = load_base_model_config()
gpt_model.eval()
gpt_model.to(device)
if compile:
    gpt_model = torch.compile(gpt_model) # requires PyTorch 2.0 (optional)

model = ServerModel(gpt_model, enc, device)
# model = DynamicBatchingServerModel(gpt_model, enc, device)
manager = BatchingManager(model)

run_inferences = Thread(target=manager.no_batching_loop)
# run_inferences = Thread(target=manager.static_batching_loop)
# run_inferences = Thread(target=manager.dynamic_batching_loop)

run_inferences.start()

@app.route('/inference', methods=['POST'])
def hello():
    data = json.loads(request.get_data())
    prompt = data['prompt']
    num_tokens = int(data['num_tokens'])
    assert isinstance(prompt, str)
    print("Request recv")
    inference = manager.enqueue(prompt, num_tokens)
    completion = inference.wait_for_completion()
    return {'completion': completion}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=105, debug=True, threaded=True)
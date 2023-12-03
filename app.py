import json
import time

from flask import Flask, request
from manager import InferenceManager, Inference
#from inference import batch_generate
from threading import Thread, Lock, Event

app = Flask(__name__)

count = 0

manager = InferenceManager()
run_inferences = Thread(target=manager.trigger_static_batch)
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
from flask import Flask, request
import json
from inference import batch_generate


app = Flask(__name__)

class InferenceManager:
    pass
    def enqueue(self);
        pass



def inference_manager(prompt: str):
    # do some sort of qneueue

@app.route('/inference', methods=['POST'])
def hello():
    data = json.loads(request.get_data())
    prompt = data['prompt']
    assert isinstance(prompt, str)
    result = inference_manager(prompt)
    return {'completion': result}
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=105, debug=True)
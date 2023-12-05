from model import load_base_model_config, ServerModel, DynamicBatchingServerModel
import torch
from batching import BatchingManager
from threading import Thread
from generate.generate import static_batch_generate, generate, dynamic_batch_generate


model, enc, device = load_base_model_config()
model = DynamicBatchingServerModel(model, enc, device)
manager = BatchingManager(model, dynamic_batch_generate)
run_inferences = Thread(target=manager.dynamic_batching_loop)
run_inferences.start()
print("run inferences thread started")

prompt = "hi"
num_tokens = 10
inference = manager.enqueue(prompt, num_tokens)
completion = inference.wait_for_completion()
print(completion)

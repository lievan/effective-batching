from model import load_base_model_config, ServerModel, DynamicBatchingServerModel
import torch
from batching import BatchingManager
from threading import Thread
from generate.generate import static_batch_generate, generate, dynamic_batch_generate


model, enc, device = load_base_model_config()
model = ServerModel(model, enc, device)
manager = BatchingManager(model, static_batch_generate)
run_inferences = Thread(target=manager.static_batching_loop)
run_inferences.start()

prompt = "hi"
num_tokens = 10
inf1 = manager.enqueue(prompt, num_tokens)
inf2 = manager.enqueue("yo", 3)
inf3 = manager.enqueue("bro", 6)
print(inf1.wait_for_completion())
print(inf2.wait_for_completion())
print(inf3.wait_for_completion())
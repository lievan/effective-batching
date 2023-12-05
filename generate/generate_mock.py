import torch


def mock_generate(inference, model):
    return " ".join(["word" for _ in range(inference.num_tokens)])

def mock_dynamic_batch_generate(next_batch, model):
    finished = []
    in_progress = []
    for inference in next_batch:
        done = inference.add_token(torch.tensor([999]))
        if done:
            finished.append(inference)
        else:
            in_progress.append(inference)
    return finished, in_progress

def mock_static_batch_generate(batch, model):
    res = []
    for inference in batch:
        res.append((" ".join(["word" for _ in range(inference.num_tokens)]), inference))
    return res
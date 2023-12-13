# effective-batching
Comparing the impact dynamic batching, static batching, and no batching have on throughput for a generative LLM inference server.

This repo uses gunicorn + flask to host a ```gpt-2-medium``` model implemented using code from Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT/tree/master) repo

Dynamic batching algorithm is modeled after the [Orca](https://www.usenix.org/conference/osdi22/presentation/yu) paper.


### Server
The server exposes a ```/inference``` endpoint that takes a request with a prompt and # of completion tokens to generate. The server does not support terminating a generation based on certain end tokens. 

```app.py``` is the main script that loads the model and defines the server logic. ```batching.py``` contains two classes: ```Inference``` and ```BatchingManager```. The ```BatchingManager``` defines how ```Inference``` objects are scheduled for model inference. When the server starts, an inference handler thread is launched that runs either ```no_batching_loop```, ```static_batching_loop```, or ```dynamic_batching_loop```.

### Client
Client code can be found in the ```client``` folder.

## Code instructions

### Server Setup

**Python Requirements**

**Environment Variables**

### Client Setup

**Python Requirements**

**Environment Variables**


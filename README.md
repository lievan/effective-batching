# effective-batching
Comparing the impact dynamic batching, static batching, and no batching have on throughput for a generative LLM inference server.

This repo uses gunicorn + flask to host a ```gpt-2-medium``` model implemented using code from Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT/tree/master) repo

Dynamic batching algorithm is modeled after the [Orca](https://www.usenix.org/conference/osdi22/presentation/yu) paper.

Use the ```Instructions``` section below for full tutorial on how to run experiments.

### Server Implementation
The server exposes a ```/inference``` endpoint that takes a request with a prompt and # of completion tokens to generate. The server does not support terminating a generation based on certain end tokens. A simple ```stats``` endpoint also exists to display server stats for tracking experiment results.

```app.py``` is the main script that loads the model and defines the server logic. 

```batching.py``` contains two classes: ```Inference``` and ```BatchingManager```. 

```BatchingManager``` defines how ```Inference``` objects are scheduled for model inference. When the server starts, an inference handler thread is launched that runs either ```no_batching_loop```, ```static_batching_loop```, or ```dynamic_batching_loop```. These loops handle new inferences every 0.01 seconds.

New requests are enqueued using the BatchingManager's ```enqueue``` function. Requests are transformed into ```Inference``` objects that hold onto the request data as well as metadata used by the ```BatchingManager```. 

This ```Inference``` object is returned by the ```enqueue``` function. Each ```Inference``` object stores a reference to a unique ```threading.Event``` object that will be used to signal when the inference has finished.

### Client
Client code can be found in the ```client``` folder.

```client.py``` is a script that launches 100 inference requests to the server, waits for the requests to finish, and then prints the results from the server ```/stats``` endpoint.

```data.py``` contains a ```PromptData``` class that is used by the client script. ```PromptData``` generates the inference data used for requests.

## Instructions

### Server Setup

**Google Compute Engine VM Setup**

The GCE VM instance needs to be configured to serve the flask server. If you only want to run the server locally, then steps 2-4 are not needed.

1. Make sure the instance has a GPU

2. In the firewalls section, make sure that HTTP Traffic and HTTPS traffic are toggled on.

<img width="357" alt="image" src="https://github.com/lievan/effective-batching/assets/42917263/dfc9797d-9c9b-4d64-87d4-5592953d948e">

3. Create a [network tag](https://cloud.google.com/vpc/docs/add-remove-network-tags) that allows ingress traffic on port 8500.

5. Add this network tag to the VM configuration.

More details on creating a VM configuration that can host a flask app [here](https://www.section.io/engineering-education/deploy-flask-to-gce/)

After the VM is created, ssh into the instance and clone this repo.

**Install Python Requirements**

1. Make and activate a Python virtual environment
2. ```cd``` into the ```effective-batching``` repo root directory
3. Run ```pip install requirements.txt```

**Launch the app**

No batching:
```./server.sh nobatch```

Static batching:
```./server.sh static```

Dynamic batching:
```./server.sh dynamic```

### Client Setup

**Python Requirements**

Install the requirements in the ```requirements.txt``` file.

**Environment Variables**

1. Make a ```.env``` file in ```effective-batching/client``` folder
2. Set ```IP=<YOUR_GCE_INSTANCE_EXTERNAL_IP>``` in the file.

The client script will read from the ```IP``` environment variable to format the request.


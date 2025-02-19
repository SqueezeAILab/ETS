set -e
set -x
#!/bin/bash

MODEL_REPO="policy-repo"

PORT=30000
tensor_parellel_size=1

CUDA_VISIBLE_DEVICES=0 python3 -m sglang.launch_server --model-path $MODEL_REPO --port $PORT --tp-size $tensor_parellel_size --trust-remote-code

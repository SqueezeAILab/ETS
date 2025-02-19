set -e
set -x
#!/bin/bash

MODEL_REPO="reward-repo"

PORT=30020
tensor_parellel_size=1

# use --mem-fraction-static 0.85 if using collocated embedding model
CUDA_VISIBLE_DEVICES=1 python3 -m sglang.launch_server --model-path $MODEL_REPO --port $PORT --tp-size $tensor_parellel_size --trust-remote-code --mem-fraction-static 0.85

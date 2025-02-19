set -e
set -x

POLICY=30000
REWARD=30020

export DATA_DIR="math500_test.jsonl"

# Iterate over the sequence lengths
for WIDTH in 16 64 256; do
    export OUT_DIR="./exp_results/ets_${WIDTH}_math500"
    export PARA_PATH="./hype-parameters/ets_${WIDTH}_math500.yaml"

    # Ensure the output directory exists
    mkdir -p $OUT_DIR

    # Run the script
    python3 rebase.py --input_path $DATA_DIR \
  	--output_path $OUT_DIR/answers.json \
  	--parameter_path $PARA_PATH \
  	--policy_host http://localhost:$POLICY \
  	--reward_host http://localhost:$REWARD \
  	--embed_device 1 # assumes reward model is on GPU 1
done

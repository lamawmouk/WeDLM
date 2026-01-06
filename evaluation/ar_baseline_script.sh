#!/bin/bash


# note: please install vllm before running this script

# install
pip install evalplus

# Configuration
MODEL_PATH="Qwen/Qwen3-8B-Instruct"
OUTPUT_BASE="output"
NUM_GPUS=8

# Dataset list
DATASETS=(
    "humaneval"
    "gsm8k"
    "mbpp"
    "mmlu"
    "arc_c"
    "arc_e"
    "hellaswag"
    "math"
    "gpqa"
)

# Extract model name from path
MODEL_NAME=$(basename "$MODEL_PATH")

# Run evaluation for each dataset
for DATASET in "${DATASETS[@]}"; do
    echo "Running evaluation: $MODEL_NAME on $DATASET"
    
    python -m evaluation.ar_baseline_eval \
        --model-path "$MODEL_PATH" \
        --dataset-name "$DATASET" \
        --output-dir "${OUTPUT_BASE}/${MODEL_NAME}/${DATASET}/" \
        --num-gpus "$NUM_GPUS" \
        --trust-remote-code
    
    echo "Finished: $DATASET"
    echo "----------------------------------------"
done

echo "All evaluations completed!"
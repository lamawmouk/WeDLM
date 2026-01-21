#!/bin/bash
# WeDLM Jacobi Evaluation Script
# Compares standard WeDLM vs Jacobi-enabled WeDLM

set -e

# Configuration
MODEL_PATH="tencent/WeDLM-8B-Instruct"
OUTPUT_DIR="eval_results_jacobi"
NUM_GPUS=1

mkdir -p ${OUTPUT_DIR}

echo "========================================"
echo "WeDLM Jacobi vs Standard Evaluation"
echo "========================================"
echo "Model: ${MODEL_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "========================================"

# Function to run evaluation
run_eval() {
    local method=$1
    local dataset=$2
    local use_jacobi=$3
    local output_name="${method}_${dataset}"

    echo ""
    echo ">>> Running ${method} on ${dataset}..."
    echo ""

    if [ "$use_jacobi" = "true" ]; then
        python -m evaluation.evaluate \
            --model_path "${MODEL_PATH}" \
            --output_dir "${OUTPUT_DIR}/${output_name}" \
            --datasets "${dataset}" \
            --use_jacobi \
            --num_gpus ${NUM_GPUS} 2>&1 | tee ${OUTPUT_DIR}/${output_name}.log
    else
        python -m evaluation.evaluate \
            --model_path "${MODEL_PATH}" \
            --output_dir "${OUTPUT_DIR}/${output_name}" \
            --datasets "${dataset}" \
            --num_gpus ${NUM_GPUS} 2>&1 | tee ${OUTPUT_DIR}/${output_name}.log
    fi
}

# ============================================
# GSM8K Evaluations
# ============================================
echo ""
echo "========== GSM8K Evaluations =========="

# Standard WeDLM
run_eval "standard" "gsm8k" "false"

# Jacobi WeDLM
run_eval "jacobi" "gsm8k" "true"

# ============================================
# MATH Evaluations
# ============================================
echo ""
echo "========== MATH Evaluations =========="

# Standard WeDLM
run_eval "standard" "math" "false"

# Jacobi WeDLM
run_eval "jacobi" "math" "true"

# ============================================
# HumanEval Evaluations
# ============================================
echo ""
echo "========== HumanEval Evaluations =========="

# Standard WeDLM
run_eval "standard" "humaneval" "false"

# Jacobi WeDLM
run_eval "jacobi" "humaneval" "true"

echo ""
echo "========================================"
echo "All evaluations complete!"
echo "Results saved to: ${OUTPUT_DIR}/"
echo "========================================"

# Generate summary
cat > ${OUTPUT_DIR}/summary.md << 'EOF'
# WeDLM Jacobi vs Standard Comparison Results

## Methods Compared
1. **Standard WeDLM**: Default entropy-based parallel decoding
2. **Jacobi WeDLM**: Fixed Gumbel noise + mismatch detection

## Key Metrics to Compare
- Accuracy / Pass rate
- Tokens per second (TPS)
- Tokens per forward (TPF)
- Decode forwards count

## Expected Benefits of Jacobi
- More deterministic token selection
- Faster convergence (fewer iterations)
- Better consistency in generated outputs

EOF

echo "Summary saved to ${OUTPUT_DIR}/summary.md"

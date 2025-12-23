#!/bin/bash

# Ablation Study Script for QA-LoRA
# This script runs experiments with different LoRA ranks and Group sizes.

# Define arrays for ablation
RANKS=(8 16 32)
GROUPS=(32 64 128)

# Base output directory
BASE_OUTPUT_DIR="./ablation_results"

echo "Starting Ablation Study..."
echo "Ranks to test: ${RANKS[@]}"
echo "Group sizes to test: ${GROUPS[@]}"

for r in "${RANKS[@]}"; do
    for g in "${GROUPS[@]}"; do
        echo "----------------------------------------------------------------"
        echo "Running experiment with Rank=$r and GroupSize=$g"
        echo "----------------------------------------------------------------"
        
        python QAT-LoRA.py \
            --lora_rank "$r" \
            --group_size "$g" \
            --output_dir "$BASE_OUTPUT_DIR" \
            --batch_size 1 \
            --grad_accum 16 \
            --lr 2e-4 \
            --lr_qat 2e-5
            
        echo "Finished experiment for Rank=$r, GroupSize=$g"
    done
done

echo "All ablation experiments completed."

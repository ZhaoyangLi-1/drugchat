#!/bin/bash
# =============================================================================
# Ablation Study Runner for InteractGPT
# =============================================================================
# Usage: bash run_ablation.sh [EXPERIMENT_NAME] [STAGE]
#   EXPERIMENT_NAME: gnn_only | imagemol_only | linear_proj | lora_rank8 |
#                    lora_rank16 | prompt_tuning_16 | prompt_tuning_32 |
#                    drugscom_only | all
#   STAGE: train | infer | eval (pick one -- there is no "run everything" stage
#          because checkpoint paths must be updated between train and infer)
#
# Examples:
#   bash run_ablation.sh gnn_only train        # Train GNN-only variant
#   bash run_ablation.sh gnn_only infer        # Run inference (update ckpt path first!)
#   bash run_ablation.sh gnn_only eval         # Evaluate inference results
#   bash run_ablation.sh all train             # Train ALL experiments sequentially
#   bash run_ablation.sh table                 # Print summary table of all results
# =============================================================================

set -e

GPU_ID=${GPU_ID:-0}
BATCH_SIZE=${BATCH_SIZE:-4}
TEST_DATA_FOLDER=${TEST_DATA_FOLDER:-"drug_drug_data/drugs_dot_com/test"}

EXPERIMENTS=(
    "gnn_only"
    "imagemol_only"
    "linear_proj"
    "lora_rank8"
    "lora_rank16"
    "prompt_tuning_16"
    "prompt_tuning_32"
    "drugscom_only"
)

is_valid_experiment() {
    local name=$1
    for exp in "${EXPERIMENTS[@]}"; do
        if [ "$exp" = "$name" ]; then
            return 0
        fi
    done
    return 1
}

run_train() {
    local exp=$1
    echo "========================================"
    echo "TRAINING: ${exp}"
    echo "========================================"
    torchrun --nproc_per_node 1 train.py \
        --cfg-path train_configs/ablation/${exp}.yaml
}

run_infer() {
    local exp=$1
    echo "========================================"
    echo "INFERENCE: ${exp}"
    echo "========================================"

    # Check that the eval config has a real checkpoint path
    local ckpt_line=$(grep "ckpt:" eval_configs/ablation/${exp}.yaml | head -1)
    if echo "$ckpt_line" | grep -q "PLACEHOLDER"; then
        echo "ERROR: Update the checkpoint path in eval_configs/ablation/${exp}.yaml first!"
        echo "  Run: python update_ablation_ckpts.py"
        echo "  Or manually set the path to checkpoint_best.pth from the training output."
        return 1
    fi

    mkdir -p eval_results/ablation

    python inference_batch.py \
        --cfg-path eval_configs/ablation/${exp}.yaml \
        --gpu-id ${GPU_ID} \
        --in_file_folder ${TEST_DATA_FOLDER} \
        --out_file eval_results/ablation/${exp}_predictions.json \
        --batch_size ${BATCH_SIZE}
}

run_eval() {
    local exp=$1
    echo "========================================"
    echo "EVALUATION: ${exp}"
    echo "========================================"

    python eval_ablation.py \
        --prediction_file eval_results/ablation/${exp}_predictions.json \
        --output_file eval_results/ablation/${exp}_metrics.json
}

run_experiment() {
    local exp=$1
    local stage=$2

    case $stage in
        train) run_train "$exp" ;;
        infer) run_infer "$exp" ;;
        eval)  run_eval  "$exp" ;;
        *)
            echo "Unknown stage: $stage (use: train | infer | eval)"
            return 1
            ;;
    esac
}

# --- Main ---
EXP_NAME=${1:-""}
STAGE=${2:-""}

if [ -z "$EXP_NAME" ]; then
    echo "Usage: bash run_ablation.sh [EXPERIMENT_NAME] [STAGE]"
    echo ""
    echo "Experiments: ${EXPERIMENTS[*]}"
    echo "  Or 'all' to run a stage for all experiments"
    echo "  Or 'table' to print the summary table"
    echo ""
    echo "Stages: train | infer | eval"
    exit 1
fi

if [ "$EXP_NAME" = "table" ]; then
    python eval_ablation.py --summary --results_dir eval_results/ablation
elif [ "$EXP_NAME" = "all" ]; then
    if [ -z "$STAGE" ]; then
        echo "ERROR: Specify a stage when using 'all'. Example: bash run_ablation.sh all train"
        exit 1
    fi
    for exp in "${EXPERIMENTS[@]}"; do
        run_experiment "$exp" "$STAGE"
    done
else
    if ! is_valid_experiment "$EXP_NAME"; then
        echo "ERROR: Unknown experiment '${EXP_NAME}'"
        echo "Valid experiments: ${EXPERIMENTS[*]}"
        exit 1
    fi
    if [ -z "$STAGE" ]; then
        echo "ERROR: Specify a stage. Example: bash run_ablation.sh ${EXP_NAME} train"
        exit 1
    fi
    run_experiment "$EXP_NAME" "$STAGE"
fi

# Ablation Study Guide for InteractGPT

This document covers everything needed to run the ablation studies. All configs and scripts are ready -- you just need to run the commands below on the GPU server.

## Overview

We have **8 ablation experiments** that each isolate one design decision in InteractGPT. Each experiment changes exactly one thing from the baseline (the full model you already trained).

| # | Experiment | Config File | What Changes | Why We Test It |
|---|-----------|-------------|--------------|----------------|
| 1 | `gnn_only` | `train_configs/ablation/gnn_only.yaml` | Remove ImageMol encoder | Does the graph structure alone capture DDI? |
| 2 | `imagemol_only` | `train_configs/ablation/imagemol_only.yaml` | Remove GNN encoder | Do molecular images alone suffice? |
| 3 | `linear_proj` | `train_configs/ablation/linear_proj.yaml` | Linear projector instead of MLP | Is the 2-layer MLP projector worth the extra parameters? |
| 4 | `lora_rank8` | `train_configs/ablation/lora_rank8.yaml` | LoRA rank=8 on LLM attention | Does fine-tuning the LLM help? |
| 5 | `lora_rank16` | `train_configs/ablation/lora_rank16.yaml` | LoRA rank=16 on LLM attention | Does more LLM fine-tuning help further? |
| 6 | `prompt_tuning_16` | `train_configs/ablation/prompt_tuning_16.yaml` | 16 learnable soft prompt tokens | Do learned prompt tokens improve predictions? |
| 7 | `prompt_tuning_32` | `train_configs/ablation/prompt_tuning_32.yaml` | 32 learnable soft prompt tokens | Does a larger soft prompt help more? |
| 8 | `drugscom_only` | `train_configs/ablation/drugscom_only.yaml` | Remove PubChem dataset | Does multi-dataset training help? |

**Baseline** = the full model you already trained (`train_configs/drugchat.yaml`): GNN + ImageMol, MLP projector, no LoRA, no prompt tuning, Drugs.com + PubChem.

## Priority Order

If GPU time is limited, run these in order of importance:

1. **`gnn_only`** and **`imagemol_only`** -- most critical, reviewers will expect modality ablation
2. **`linear_proj`** -- tests projector design, easy to explain in the paper
3. **`drugscom_only`** -- tests multi-dataset contribution
4. **`lora_rank8`** -- tests LLM fine-tuning
5. The rest (`lora_rank16`, `prompt_tuning_16`, `prompt_tuning_32`) -- nice to have

## Step-by-Step Instructions

### Step 1: Pull Latest Code

```bash
cd /path/to/drug-drug-interaction
git pull
```

Make sure you see these new files:
```
train_configs/ablation/          # 8 training configs
eval_configs/ablation/           # 8 eval configs
run_ablation.sh                  # Runner script
eval_ablation.py                 # Evaluation + table generation
update_ablation_ckpts.py         # Auto-fills checkpoint paths in eval configs
```

### Step 2: Train Each Experiment

Each experiment is a full training run (same as the baseline -- 20 epochs).

**Option A: Use the runner script**
```bash
bash run_ablation.sh gnn_only train
bash run_ablation.sh imagemol_only train
bash run_ablation.sh linear_proj train
# ... etc
```

**Option B: Run torchrun directly**
```bash
torchrun --nproc_per_node 1 train.py --cfg-path train_configs/ablation/gnn_only.yaml
torchrun --nproc_per_node 1 train.py --cfg-path train_configs/ablation/imagemol_only.yaml
torchrun --nproc_per_node 1 train.py --cfg-path train_configs/ablation/linear_proj.yaml
torchrun --nproc_per_node 1 train.py --cfg-path train_configs/ablation/lora_rank8.yaml
torchrun --nproc_per_node 1 train.py --cfg-path train_configs/ablation/lora_rank16.yaml
torchrun --nproc_per_node 1 train.py --cfg-path train_configs/ablation/prompt_tuning_16.yaml
torchrun --nproc_per_node 1 train.py --cfg-path train_configs/ablation/prompt_tuning_32.yaml
torchrun --nproc_per_node 1 train.py --cfg-path train_configs/ablation/drugscom_only.yaml
```

Each experiment saves checkpoints to `output/ablation_<name>/`. For example, `gnn_only` saves to `output/ablation_gnn_only/<timestamp>/checkpoint_best.pth`.

**Note:** You can run multiple experiments in parallel if you have multiple GPUs. Just set different `CUDA_VISIBLE_DEVICES`:
```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 train.py --cfg-path train_configs/ablation/gnn_only.yaml &
CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node 1 train.py --cfg-path train_configs/ablation/imagemol_only.yaml &
```

### Step 3: Update Eval Configs with Checkpoint Paths

After training finishes, the eval configs need to know where the checkpoints are. Run:

```bash
python update_ablation_ckpts.py
```

This automatically scans `output/ablation_*/` for `checkpoint_best.pth` and fills in the paths in `eval_configs/ablation/*.yaml`.

**Check the output** -- it will print OK/MISS for each experiment. If any show MISS, the training for that experiment hasn't finished yet or saved to a different location. In that case, manually edit the eval config:

```yaml
# eval_configs/ablation/gnn_only.yaml
model:
  ckpt: '/actual/path/to/checkpoint_best.pth'  # <-- update this line
```

### Step 4: Run Inference

```bash
mkdir -p eval_results/ablation

# Option A: Use the runner script
bash run_ablation.sh gnn_only infer
bash run_ablation.sh imagemol_only infer
# ... etc

# Option B: Run directly
python inference_batch.py \
    --cfg-path eval_configs/ablation/gnn_only.yaml \
    --gpu-id 0 \
    --in_file_folder drug_drug_data/drugs_dot_com/test \
    --out_file eval_results/ablation/gnn_only_predictions.json \
    --batch_size 4
```

Repeat for each experiment. Output goes to `eval_results/ablation/<name>_predictions.json`.

### Step 5: Evaluate

```bash
# Option A: Use the runner script
bash run_ablation.sh gnn_only eval
bash run_ablation.sh imagemol_only eval
# ... etc

# Option B: Run directly
python eval_ablation.py \
    --prediction_file eval_results/ablation/gnn_only_predictions.json \
    --output_file eval_results/ablation/gnn_only_metrics.json
```

Repeat for each experiment.

### Step 6: Generate the Ablation Table

Once all experiments are evaluated:

```bash
# Print a formatted text table
python eval_ablation.py --summary --results_dir eval_results/ablation

# Print a LaTeX table (copy-paste into the paper)
python eval_ablation.py --summary --results_dir eval_results/ablation --latex
```

You don't need all 8 to generate the table -- it will include whatever `*_metrics.json` files exist.

### Step 7: Baseline Entry

To include the baseline (full model) in the table, also run inference + eval for the baseline checkpoint.

**Important:** First verify the checkpoint path in `eval_configs/drugbank.yaml` points to your baseline `checkpoint_best.pth`. The current path is hardcoded to Zhaoyang's server -- update it if needed.

```bash
python inference_batch.py \
    --cfg-path eval_configs/drugbank.yaml \
    --gpu-id 0 \
    --in_file_folder drug_drug_data/drugs_dot_com/test \
    --out_file eval_results/ablation/baseline_predictions.json \
    --batch_size 4

python eval_ablation.py \
    --prediction_file eval_results/ablation/baseline_predictions.json \
    --output_file eval_results/ablation/baseline_metrics.json
```

## Quick Reference: All Commands for One Experiment

Here's the full pipeline for a single experiment (e.g., `gnn_only`):

```bash
# 1. Train
torchrun --nproc_per_node 1 train.py --cfg-path train_configs/ablation/gnn_only.yaml

# 2. Update checkpoint path
python update_ablation_ckpts.py

# 3. Inference
python inference_batch.py \
    --cfg-path eval_configs/ablation/gnn_only.yaml \
    --gpu-id 0 \
    --in_file_folder drug_drug_data/drugs_dot_com/test \
    --out_file eval_results/ablation/gnn_only_predictions.json \
    --batch_size 4

# 4. Evaluate
python eval_ablation.py \
    --prediction_file eval_results/ablation/gnn_only_predictions.json \
    --output_file eval_results/ablation/gnn_only_metrics.json

# 5. Check results
python eval_ablation.py --summary --results_dir eval_results/ablation
```

## What Each Experiment Tests (For the Paper)

### Modality Ablation (gnn_only, imagemol_only)
Tests whether both molecular encoders contribute. The GNN captures graph topology (atom connectivity, bond types). ImageMol captures 2D visual patterns from molecular structure images. If both matter, it validates the dual-encoder design.

### Projector Design (linear_proj)
The baseline uses a 2-layer MLP with GELU activation (300/512 -> 5120 -> 5120, ~57M params). This ablation uses a single linear layer (300/512 -> 5120, ~4M params). Tests whether the extra capacity of the MLP is needed to bridge the gap between molecular features and LLM embedding space.

### LoRA Fine-tuning (lora_rank8, lora_rank16)
The baseline keeps the LLM (Vicuna-13B) completely frozen. LoRA adds small trainable low-rank matrices to the LLM's attention layers (q, k, v, o projections). Tests whether domain-adapting the LLM improves DDI prediction quality.

### Prompt Tuning (prompt_tuning_16, prompt_tuning_32)
Adds learnable embedding vectors that are concatenated to each compound's tokens before being fed to the LLM. These tokens are trained end-to-end and can learn to "steer" the LLM's attention toward the molecular features. Tests whether learned prompt tokens help beyond the fixed text prompt.

### Dataset Ablation (drugscom_only)
The baseline trains on Drugs.com DDI data + PubChem molecular QA data. This ablation removes PubChem. Tests whether the multi-task data from PubChem improves DDI-specific performance.

## Code Changes

Only one source file was modified (`pipeline/models/drugchat.py`). The `encode_img_infer()` method was fixed to apply soft prompt tokens during inference -- matching the training behavior in `prompt_wrap()`. Without this fix, prompt tuning experiments would silently ignore the learned soft prompt tokens at test time. All other ablations are purely config-driven.

## Troubleshooting

**"PLACEHOLDER" error during inference:**
You forgot to update the checkpoint path. Run `python update_ablation_ckpts.py` or manually edit the eval config.

**Out of GPU memory:**
- Try `batch_size_train: 4` instead of 5 in the training config
- LoRA experiments use slightly more memory than baseline
- Try `--batch_size 2` for inference

**Training seems stuck or loss doesn't decrease:**
Check W&B logs. The first epoch is warmup (lr goes from 1e-6 to 2e-5), so early loss may be high.

**Results look identical to baseline:**
Double-check that the training config actually has the intended change (e.g., `use_mlp: false` for linear_proj). Also verify the eval config points to the correct checkpoint (not the baseline checkpoint).

## Contact

If anything is unclear or breaks, message Sushaanth on Slack. I can debug remotely -- just send me the error output.

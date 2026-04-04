"""
Ablation Study Evaluation for InteractGPT.

Usage:
    # Evaluate a single experiment's predictions:
    python eval_ablation.py --prediction_file eval_results/ablation/gnn_only_predictions.json \
                            --output_file eval_results/ablation/gnn_only_metrics.json

    # Generate summary table from all evaluated experiments:
    python eval_ablation.py --summary --results_dir eval_results/ablation

    # Generate LaTeX table:
    python eval_ablation.py --summary --results_dir eval_results/ablation --latex
"""
import argparse
import json
import os
import glob

import numpy as np
from tqdm import tqdm

from eval_metrics import (
    model as sentence_model,
    semantic_similarity,
    bleu_n_score,
    meteor_score,
    rouge,
)


EXPERIMENT_LABELS = {
    "baseline": "Full Model (GNN + ImageMol)",
    "gnn_only": "w/o ImageMol (GNN only)",
    "imagemol_only": "w/o GNN (ImageMol only)",
    "linear_proj": "w/o MLP (Linear projector)",
    "lora_rank8": "w/ LoRA (rank=8)",
    "lora_rank16": "w/ LoRA (rank=16)",
    "prompt_tuning_16": "w/ Prompt Tuning (k=16)",
    "prompt_tuning_32": "w/ Prompt Tuning (k=32)",
    "drugscom_only": "w/o PubChem (Drugs.com only)",
}

METRIC_KEYS = [
    "semantic_similarity",
    "bleu_1",
    "bleu_2",
    "bleu_3",
    "bleu_4",
    "meteor",
    "rouge_1",
    "rouge_L",
]

METRIC_DISPLAY = {
    "semantic_similarity": "Sem. Sim.",
    "bleu_1": "BLEU-1",
    "bleu_2": "BLEU-2",
    "bleu_3": "BLEU-3",
    "bleu_4": "BLEU-4",
    "meteor": "METEOR",
    "rouge_1": "ROUGE-1",
    "rouge_L": "ROUGE-L",
}


def evaluate_predictions(data):
    """Evaluate a prediction file and return per-sample + mean metrics."""
    results = {}
    sims, meteors = [], []
    bleu_scores = {n: [] for n in [1, 2, 3, 4]}
    rouge1_scores, rougel_scores = [], []

    for smiles, entry in tqdm(data.items(), desc="Evaluating"):
        query, gt, pred = entry
        gt = gt.strip() if gt else ""
        pred = pred.strip() if pred else ""

        emb_gt = sentence_model.encode(gt, convert_to_tensor=True)
        emb_pred = sentence_model.encode(pred, convert_to_tensor=True)
        sim = semantic_similarity(emb_gt, emb_pred).item()
        sims.append(sim)

        met = meteor_score(gt, pred)
        meteors.append(met)

        sample_result = {
            "query": query,
            "ground_truth": gt,
            "predicted": pred,
            "semantic_similarity": sim,
            "meteor": met,
        }

        for n in [1, 2, 3, 4]:
            b = bleu_n_score(pred, gt, n)
            bleu_scores[n].append(b)
            sample_result[f"bleu_{n}"] = b

        rc = rouge.compute(
            predictions=[pred],
            references=[gt],
            rouge_types=["rouge1", "rougeL"],
        )
        sample_result["rouge_1"] = rc["rouge1"]
        sample_result["rouge_L"] = rc["rougeL"]
        rouge1_scores.append(rc["rouge1"])
        rougel_scores.append(rc["rougeL"])

        results[smiles] = sample_result

    mean_scores = {
        "semantic_similarity": float(np.mean(sims)) if sims else 0,
        "meteor": float(np.mean(meteors)) if meteors else 0,
        "rouge_1": float(np.mean(rouge1_scores)) if rouge1_scores else 0,
        "rouge_L": float(np.mean(rougel_scores)) if rougel_scores else 0,
    }
    for n in [1, 2, 3, 4]:
        mean_scores[f"bleu_{n}"] = float(np.mean(bleu_scores[n])) if bleu_scores[n] else 0

    results["mean_scores"] = mean_scores
    return results


def load_mean_scores(metrics_file):
    """Load mean scores from a metrics JSON file."""
    with open(metrics_file, "r") as f:
        data = json.load(f)
    return data.get("mean_scores", {})


def print_summary_table(results_dir, latex=False):
    """Print a summary table of all ablation experiments."""
    metrics_files = sorted(glob.glob(os.path.join(results_dir, "*_metrics.json")))

    if not metrics_files:
        print(f"No metrics files found in {results_dir}/")
        print("Run evaluation first: python eval_ablation.py --prediction_file ... --output_file ...")
        return

    rows = []
    for fpath in metrics_files:
        exp_name = os.path.basename(fpath).replace("_metrics.json", "")
        label = EXPERIMENT_LABELS.get(exp_name, exp_name)
        scores = load_mean_scores(fpath)
        rows.append((exp_name, label, scores))

    if latex:
        print_latex_table(rows)
    else:
        print_text_table(rows)


def print_text_table(rows):
    """Print a formatted text table."""
    col_width = 14
    label_width = 38

    header = f"{'Experiment':<{label_width}}"
    for key in METRIC_KEYS:
        header += f" {METRIC_DISPLAY[key]:>{col_width}}"
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for exp_name, label, scores in rows:
        row = f"{label:<{label_width}}"
        for key in METRIC_KEYS:
            val = scores.get(key, 0)
            row += f" {val:>{col_width}.4f}"
        print(row)

    print("=" * len(header))


def print_latex_table(rows):
    """Print a LaTeX-formatted table for the paper."""
    n_metrics = len(METRIC_KEYS)
    col_spec = "l" + "c" * n_metrics

    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Ablation study results. We report mean scores across the test set.}")
    print(r"\label{tab:ablation}")
    print(r"\resizebox{\textwidth}{!}{")
    print(r"\begin{tabular}{" + col_spec + "}")
    print(r"\toprule")

    header = "Configuration"
    for key in METRIC_KEYS:
        header += f" & {METRIC_DISPLAY[key]}"
    header += r" \\"
    print(header)
    print(r"\midrule")

    for i, (exp_name, label, scores) in enumerate(rows):
        row = label
        for key in METRIC_KEYS:
            val = scores.get(key, 0)
            row += f" & {val:.4f}"
        row += r" \\"
        if exp_name == "imagemol_only" or exp_name == "linear_proj" or exp_name == "lora_rank16" or exp_name == "prompt_tuning_32":
            row += r" \midrule"
        print(row)

    print(r"\bottomrule")
    print(r"\end{tabular}}")
    print(r"\end{table}")


def main():
    parser = argparse.ArgumentParser(description="Ablation study evaluation")
    parser.add_argument("--prediction_file", type=str, help="Path to prediction JSON file")
    parser.add_argument("--output_file", type=str, help="Path to output metrics JSON file")
    parser.add_argument("--summary", action="store_true", help="Print summary table of all experiments")
    parser.add_argument("--results_dir", type=str, default="eval_results/ablation",
                        help="Directory containing *_metrics.json files")
    parser.add_argument("--latex", action="store_true", help="Output LaTeX table format")
    args = parser.parse_args()

    if args.summary:
        print_summary_table(args.results_dir, latex=args.latex)
    elif args.prediction_file and args.output_file:
        print(f"Loading predictions from {args.prediction_file}")
        with open(args.prediction_file, "r") as f:
            data = json.load(f)

        results = evaluate_predictions(data)

        out_dir = os.path.dirname(args.output_file)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output_file, "w") as f:
            json.dump(results, f, indent=4)

        print(f"\nResults saved to {args.output_file}")
        print("\nMean Scores:")
        for key in METRIC_KEYS:
            val = results["mean_scores"].get(key, 0)
            print(f"  {METRIC_DISPLAY[key]:>12}: {val:.4f}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

"""
Auto-update eval configs with checkpoint paths after training.

Usage:
    python update_ablation_ckpts.py

Scans output/ablation_* directories for checkpoint_best.pth,
then updates the corresponding eval_configs/ablation/*.yaml files.
"""
import os
import glob
import re


def find_best_checkpoint(output_dir):
    """Find checkpoint_best.pth in the output directory (may be in a timestamped subdirectory)."""
    # Direct path
    direct = os.path.join(output_dir, "checkpoint_best.pth")
    if os.path.exists(direct):
        return os.path.abspath(direct)

    # Check timestamped subdirectories (e.g., output/ablation_gnn_only/20250408060/)
    subdirs = sorted(glob.glob(os.path.join(output_dir, "*/")), reverse=True)
    for subdir in subdirs:
        ckpt = os.path.join(subdir, "checkpoint_best.pth")
        if os.path.exists(ckpt):
            return os.path.abspath(ckpt)

    return None


def update_eval_config(eval_config_path, ckpt_path):
    """Replace the ckpt placeholder/value in the eval config."""
    with open(eval_config_path, "r") as f:
        content = f.read()

    new_content = re.sub(
        r"""ckpt:\s*(['"])[^'"]*\1""",
        f"ckpt: '{ckpt_path}'",
        content,
    )

    if new_content != content:
        with open(eval_config_path, "w") as f:
            f.write(new_content)
        return True
    return False


def main():
    experiments = [
        "gnn_only",
        "imagemol_only",
        "linear_proj",
        "lora_rank8",
        "lora_rank16",
        "prompt_tuning_16",
        "prompt_tuning_32",
        "drugscom_only",
    ]

    updated = 0
    for exp in experiments:
        output_dir = f"output/ablation_{exp}"
        eval_config = f"eval_configs/ablation/{exp}.yaml"

        if not os.path.exists(eval_config):
            print(f"  SKIP {exp}: eval config not found")
            continue

        ckpt = find_best_checkpoint(output_dir)
        if ckpt:
            if update_eval_config(eval_config, ckpt):
                print(f"  OK   {exp}: {ckpt}")
                updated += 1
            else:
                print(f"  SAME {exp}: already up to date")
        else:
            print(f"  MISS {exp}: no checkpoint_best.pth found in {output_dir}/")

    print(f"\nUpdated {updated} eval configs.")


if __name__ == "__main__":
    main()

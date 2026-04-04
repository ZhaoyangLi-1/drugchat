#!/usr/bin/env python
"""
Validate ablation study configs for consistency.

Checks:
  A. encoder_names <-> feat_dims consistency
  B. encoder_names <-> data_type consistency (per dataset)
  C. Each ablation changes exactly one conceptual dimension from baseline
  D. output_dir is unique across all configs
  E. Eval configs match their corresponding train configs

Usage:
    python validate_ablation_configs.py
    python validate_ablation_configs.py -v  # verbose
"""

import sys
from pathlib import Path
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).parent
BASELINE_TRAIN = PROJECT_ROOT / "train_configs" / "drugchat.yaml"
TRAIN_DIR = PROJECT_ROOT / "train_configs" / "ablation"
EVAL_DIR = PROJECT_ROOT / "eval_configs" / "ablation"

EXPERIMENTS = [
    "gnn_only", "imagemol_only", "linear_proj",
    "lora_rank8", "lora_rank16",
    "prompt_tuning_16", "prompt_tuning_32",
    "drugscom_only",
]

# Which "conceptual dimensions" each ablation is allowed to change from baseline.
# encoder_group = encoder_names + feat_dims + data_type (they always change together)
EXPECTED_DIFFS = {
    "gnn_only":         {"encoder_group"},
    "imagemol_only":    {"encoder_group"},
    "linear_proj":      {"use_mlp"},
    "lora_rank8":       {"lora_rank"},
    "lora_rank16":      {"lora_rank"},
    "prompt_tuning_16": {"prompt_tuning"},
    "prompt_tuning_32": {"prompt_tuning"},
    "drugscom_only":    {"datasets"},
}

# Expected encoder -> feat_dim key -> value
ENCODER_FEAT_MAP = {
    "gnn": ("graph_feat", 300),
    "image_mol": ("image_feat", 512),
}

# Expected encoder -> data_type entry
ENCODER_DATA_MAP = {
    "gnn": "graph",
    "image_mol": "image",
}


def load_yaml(path):
    return OmegaConf.load(path)


def to_plain(obj):
    """Convert OmegaConf objects to plain Python types."""
    return OmegaConf.to_container(obj, resolve=True)


def extract_comparable(cfg):
    """Extract the fields we compare between ablation and baseline."""
    model = cfg.model
    encoder_names = sorted(list(model.encoder_names))
    feat_dims = dict(to_plain(model.feat_dims)) if model.get("feat_dims") else {}
    datasets_cfg = cfg.get("datasets", {})
    dataset_names = sorted(datasets_cfg.keys())
    data_types = {}
    for ds_name in dataset_names:
        ds = datasets_cfg[ds_name]
        dt = sorted(list(to_plain(ds.data_type))) if ds.get("data_type") else []
        data_types[ds_name] = dt

    return {
        "encoder_names": encoder_names,
        "feat_dims": feat_dims,
        "data_types": data_types,
        "dataset_names": dataset_names,
        "use_mlp": model.get("use_mlp", False),
        "lora_rank": model.get("lora_rank", 0),
        "prompt_tuning": model.get("prompt_tuning", 0),
    }


def diff_from_baseline(baseline_cmp, ablation_cmp):
    """Return which conceptual dimensions differ."""
    diffs = set()

    # encoder_group: encoder_names, feat_dims, and data_types change together
    if baseline_cmp["encoder_names"] != ablation_cmp["encoder_names"]:
        diffs.add("encoder_group")
    if baseline_cmp["feat_dims"] != ablation_cmp["feat_dims"]:
        diffs.add("encoder_group")
    # data_types can differ due to encoder change OR dataset change, check per shared dataset
    baseline_ds = set(baseline_cmp["dataset_names"])
    ablation_ds = set(ablation_cmp["dataset_names"])
    shared_ds = baseline_ds & ablation_ds
    for ds in shared_ds:
        if baseline_cmp["data_types"][ds] != ablation_cmp["data_types"][ds]:
            diffs.add("encoder_group")

    if baseline_cmp["dataset_names"] != ablation_cmp["dataset_names"]:
        diffs.add("datasets")

    if baseline_cmp["use_mlp"] != ablation_cmp["use_mlp"]:
        diffs.add("use_mlp")
    if baseline_cmp["lora_rank"] != ablation_cmp["lora_rank"]:
        diffs.add("lora_rank")
    if baseline_cmp["prompt_tuning"] != ablation_cmp["prompt_tuning"]:
        diffs.add("prompt_tuning")

    return diffs


def check_encoder_feat_dims(cfg, name):
    """Check A: encoder_names <-> feat_dims consistency."""
    errors = []
    model = cfg.model
    encoder_names = list(model.encoder_names)
    feat_dims = dict(to_plain(model.feat_dims)) if model.get("feat_dims") else {}

    for enc, (feat_key, expected_val) in ENCODER_FEAT_MAP.items():
        if enc in encoder_names:
            if feat_key not in feat_dims:
                errors.append(f"{enc} in encoder_names but {feat_key} missing from feat_dims")
            elif feat_dims[feat_key] != expected_val:
                errors.append(f"{feat_key}={feat_dims[feat_key]}, expected {expected_val}")
        else:
            if feat_key in feat_dims:
                errors.append(f"{enc} not in encoder_names but {feat_key} present in feat_dims")

    # Check no extra keys in feat_dims
    expected_keys = {fk for enc, (fk, _) in ENCODER_FEAT_MAP.items() if enc in encoder_names}
    extra = set(feat_dims.keys()) - expected_keys
    if extra:
        errors.append(f"unexpected feat_dims keys: {extra}")

    return errors


def check_encoder_data_type(cfg, name):
    """Check B: encoder_names <-> data_type consistency per dataset."""
    errors = []
    model = cfg.model
    encoder_names = list(model.encoder_names)
    datasets_cfg = cfg.get("datasets", {})

    for ds_name, ds_cfg in datasets_cfg.items():
        data_type = list(to_plain(ds_cfg.data_type)) if ds_cfg.get("data_type") else []
        for enc, dt_entry in ENCODER_DATA_MAP.items():
            if enc in encoder_names and dt_entry not in data_type:
                errors.append(f"dataset {ds_name}: {enc} in encoder_names but '{dt_entry}' not in data_type")
            if enc not in encoder_names and dt_entry in data_type:
                errors.append(f"dataset {ds_name}: {enc} not in encoder_names but '{dt_entry}' in data_type")

    return errors


def check_eval_train_match(eval_cfg, train_cfg, name):
    """Check E: eval config matches train config on key fields."""
    errors = []
    warnings = []
    eval_model = eval_cfg.model
    train_model = train_cfg.model

    # encoder_names (order-insensitive)
    eval_enc = sorted(list(eval_model.encoder_names))
    train_enc = sorted(list(train_model.encoder_names))
    if eval_enc != train_enc:
        errors.append(f"encoder_names: eval={eval_enc} vs train={train_enc}")

    # feat_dims
    eval_fd = dict(to_plain(eval_model.feat_dims)) if eval_model.get("feat_dims") else {}
    train_fd = dict(to_plain(train_model.feat_dims)) if train_model.get("feat_dims") else {}
    if eval_fd != train_fd:
        errors.append(f"feat_dims: eval={eval_fd} vs train={train_fd}")

    # use_mlp
    eval_mlp = eval_model.get("use_mlp", False)
    train_mlp = train_model.get("use_mlp", False)
    if eval_mlp != train_mlp:
        errors.append(f"use_mlp: eval={eval_mlp} vs train={train_mlp}")

    # lora_rank (some eval configs omit when 0)
    eval_lr = eval_model.get("lora_rank", 0)
    train_lr = train_model.get("lora_rank", 0)
    if eval_lr != train_lr:
        errors.append(f"lora_rank: eval={eval_lr} vs train={train_lr}")

    # prompt_tuning
    eval_pt = eval_model.get("prompt_tuning", 0)
    train_pt = train_model.get("prompt_tuning", 0)
    if eval_pt != train_pt:
        errors.append(f"prompt_tuning: eval={eval_pt} vs train={train_pt}")

    # ckpt exists
    ckpt = eval_model.get("ckpt", "")
    if not ckpt:
        errors.append("ckpt field missing from eval config")
    elif ckpt == "PLACEHOLDER":
        warnings.append("ckpt is PLACEHOLDER -- update after training")

    return errors, warnings


def main():
    verbose = "-v" in sys.argv or "--verbose" in sys.argv
    total_fail = 0
    total_warn = 0

    # Load baseline
    baseline_cfg = load_yaml(BASELINE_TRAIN)
    baseline_cmp = extract_comparable(baseline_cfg)

    # =============================================
    print("=== Train Config Validation ===")
    # =============================================
    output_dirs = {str(baseline_cfg.run.output_dir): "baseline"}

    for exp in EXPERIMENTS:
        train_path = TRAIN_DIR / f"{exp}.yaml"
        if not train_path.exists():
            print(f"[FAIL] {exp} -- train config not found: {train_path}")
            total_fail += 1
            continue

        cfg = load_yaml(train_path)
        errors = []

        # Check A
        errs_a = check_encoder_feat_dims(cfg, exp)
        errors.extend(errs_a)

        # Check B
        errs_b = check_encoder_data_type(cfg, exp)
        errors.extend(errs_b)

        # Check C: single-dimension diff
        ablation_cmp = extract_comparable(cfg)
        actual_diffs = diff_from_baseline(baseline_cmp, ablation_cmp)
        expected_diffs = EXPECTED_DIFFS[exp]
        if actual_diffs != expected_diffs:
            errors.append(
                f"dimension diff: expected {expected_diffs}, got {actual_diffs}"
            )

        # Check D: unique output_dir
        out_dir = str(cfg.run.output_dir)
        if out_dir in output_dirs:
            errors.append(f"output_dir '{out_dir}' duplicates {output_dirs[out_dir]}")
        else:
            output_dirs[out_dir] = exp

        if errors:
            print(f"[FAIL] {exp}")
            for e in errors:
                print(f"       {e}")
            total_fail += 1
        else:
            detail = f"encoder/feat OK, data_type OK, diff={actual_diffs}"
            print(f"[PASS] {exp}" + (f" -- {detail}" if verbose else ""))

    # =============================================
    print("\n=== Eval <-> Train Consistency ===")
    # =============================================

    for exp in EXPERIMENTS:
        eval_path = EVAL_DIR / f"{exp}.yaml"
        train_path = TRAIN_DIR / f"{exp}.yaml"

        if not eval_path.exists():
            print(f"[FAIL] {exp} -- eval config not found: {eval_path}")
            total_fail += 1
            continue
        if not train_path.exists():
            print(f"[FAIL] {exp} -- train config not found (skip eval check)")
            total_fail += 1
            continue

        eval_cfg = load_yaml(eval_path)
        train_cfg = load_yaml(train_path)
        errors, warnings = check_eval_train_match(eval_cfg, train_cfg, exp)

        if errors:
            print(f"[FAIL] {exp}")
            for e in errors:
                print(f"       {e}")
            total_fail += 1
        else:
            warn_str = f" ({'; '.join(warnings)})" if warnings else ""
            print(f"[PASS] {exp}" + (warn_str if verbose or warnings else ""))
            total_warn += len(warnings)

    # =============================================
    print(f"\n=== Summary ===")
    # =============================================
    n = len(EXPERIMENTS)
    train_pass = n - total_fail  # approximation; good enough
    if total_fail == 0:
        print(f"All {n} train configs PASSED, all {n} eval configs PASSED")
        if total_warn:
            print(f"{total_warn} warning(s)")
        print("Result: OK")
    else:
        print(f"{total_fail} check(s) FAILED")
        print("Result: FAIL")

    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""
Dry-run smoke test for DrugChat ablation configs.

Mocks Vicuna-13B with a tiny LLM (hidden=64, 2 layers) and verifies each
ablation config works end-to-end: instantiation, forward pass, inference path.
No pretrained weights or large GPU required.

Usage:
    python test_ablation_smoke.py                  # all configs
    python test_ablation_smoke.py --config gnn_only  # single config
    pytest test_ablation_smoke.py -v               # via pytest
"""

import os
import sys
import argparse
import contextlib
import unittest.mock
from pathlib import Path

import types

import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from omegaconf import OmegaConf
from transformers import LlamaTokenizer
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.tokenization_utils_base import BatchEncoding

# ---------------------------------------------------------------------------
# Stub out heavy pipeline transitive imports BEFORE importing pipeline modules.
# pipeline/__init__.py imports the entire framework (datasets, tasks, runners)
# which pulls in pandas, iopath, etc.  We only need the model files, so we
# inject lightweight stubs for the two problematic utility modules and a
# no-op pipeline package entry.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Prevent pipeline/__init__.py from executing
_pkg = types.ModuleType("pipeline")
_pkg.__path__ = [str(PROJECT_ROOT / "pipeline")]
_pkg.__package__ = "pipeline"
sys.modules["pipeline"] = _pkg

# Stub pipeline.common (sub-package)
_common = types.ModuleType("pipeline.common")
_common.__path__ = [str(PROJECT_ROOT / "pipeline" / "common")]
_common.__package__ = "pipeline.common"
sys.modules["pipeline.common"] = _common

# Stub pipeline.models (sub-package — skip its __init__.py which imports processors etc.)
_models = types.ModuleType("pipeline.models")
_models.__path__ = [str(PROJECT_ROOT / "pipeline" / "models")]
_models.__package__ = "pipeline.models"
sys.modules["pipeline.models"] = _models

# Stub pipeline.common.dist_utils — BaseModel imports download_cached_file, is_dist_avail_and_initialized
_dist_utils = types.ModuleType("pipeline.common.dist_utils")
_dist_utils.download_cached_file = lambda *a, **kw: None
_dist_utils.is_dist_avail_and_initialized = lambda: False
sys.modules["pipeline.common.dist_utils"] = _dist_utils

# Stub pipeline.common.utils — BaseModel imports get_abs_path, is_url
_common_utils = types.ModuleType("pipeline.common.utils")
_common_utils.get_abs_path = lambda p: os.path.join(str(PROJECT_ROOT / "pipeline"), p)
_common_utils.is_url = lambda u: False
sys.modules["pipeline.common.utils"] = _common_utils

# Stub torch_scatter — redirect to torch_geometric's built-in scatter
# (torch_scatter is a legacy package that fails to build against nightly torch)
def _scatter_add(src, index, dim=0, dim_size=None, fill_value=0):
    from torch_geometric.utils import scatter
    return scatter(src, index, dim=dim, dim_size=dim_size, reduce="sum")

def _scatter_mean(src, index, dim=0, dim_size=None, fill_value=0):
    from torch_geometric.utils import scatter
    return scatter(src, index, dim=dim, dim_size=dim_size, reduce="mean")

_torch_scatter = types.ModuleType("torch_scatter")
_torch_scatter.scatter_add = _scatter_add
_torch_scatter.scatter_mean = _scatter_mean
sys.modules["torch_scatter"] = _torch_scatter

# Now safe to import the model files (they only depend on registry + the stubs above)
from pipeline.common.registry import registry  # noqa: E402
# BaseModel must be available at pipeline.models.BaseModel for the @registry.register_model decorator
from pipeline.models.base_model import BaseModel  # noqa: E402
sys.modules["pipeline.models"].BaseModel = BaseModel
_models.BaseModel = BaseModel

from pipeline.models.modeling_llama import LlamaForCausalLM as LocalLlamaForCausalLM  # noqa: E402
from pipeline.models.drugchat import DrugChat  # noqa: E402
from pipeline.models.gnn import GNN  # noqa: E402
from pipeline.models.image_mol import ImageMol  # noqa: E402
from pipeline.models.utils import Mlp  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TINY_LLAMA_CONFIG = LlamaConfig(
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=2,
    num_key_value_heads=2,
    vocab_size=32000,
    max_position_embeddings=512,
)
TINY_HIDDEN = TINY_LLAMA_CONFIG.hidden_size

MODEL_DEFAULTS = PROJECT_ROOT / "pipeline" / "configs" / "models" / "drugchat.yaml"
BASELINE_TRAIN = PROJECT_ROOT / "train_configs" / "drugchat.yaml"
TRAIN_DIR = PROJECT_ROOT / "train_configs" / "ablation"

ALL_CONFIGS = [
    "baseline",
    "gnn_only", "imagemol_only", "linear_proj",
    "lora_rank8", "lora_rank16",
    "prompt_tuning_16", "prompt_tuning_32",
    "drugscom_only",
]


# ---------------------------------------------------------------------------
# Mock tokenizer
# ---------------------------------------------------------------------------

class MockTokenizer:
    """Minimal tokenizer mock supporting all operations DrugChat needs."""

    def __init__(self):
        self.pad_token = "<pad>"
        self.pad_token_id = 0
        self.bos_token = "<s>"
        self.bos_token_id = 1
        self.eos_token = "</s>"
        self.eos_token_id = 2
        self.unk_token = "<unk>"
        self.unk_token_id = 0
        self.padding_side = "right"
        self.model_max_length = 512

    def __call__(self, text, return_tensors=None, padding=False, truncation=False,
                 max_length=None, add_special_tokens=True, **kwargs):
        if isinstance(text, str):
            text = [text]
        seq_len = 10
        input_ids = torch.ones(len(text), seq_len, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        return BatchEncoding({"input_ids": input_ids, "attention_mask": attention_mask})

    def encode(self, text, add_special_tokens=True, **kwargs):
        return [1, 2, 3, 4, 5]

    def decode(self, ids, skip_special_tokens=False, **kwargs):
        return "mock output"

    def batch_decode(self, ids, skip_special_tokens=False, **kwargs):
        if isinstance(ids, torch.Tensor):
            return ["mock output"] * ids.shape[0]
        return ["mock output"] * len(ids)

    def add_special_tokens(self, special_tokens_dict):
        for k, v in special_tokens_dict.items():
            setattr(self, k, v)

    def __len__(self):
        return 32000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model_config(config_name):
    """Load and merge model config (model defaults + train overrides)."""
    if config_name == "baseline":
        train_path = BASELINE_TRAIN
    else:
        train_path = TRAIN_DIR / f"{config_name}.yaml"

    user_cfg = OmegaConf.load(train_path)
    defaults = OmegaConf.load(MODEL_DEFAULTS)
    merged = OmegaConf.merge(defaults.model, user_cfg.model)
    return merged


def create_fake_graph():
    """Create a fake molecular graph matching GNN embedding dimensions."""
    num_nodes = 10
    num_edges = 12
    return Data(
        x=torch.stack([
            torch.randint(0, 120, (num_nodes,)),   # atom type (0-119)
            torch.randint(0, 3, (num_nodes,)),      # chirality (0-2)
        ], dim=1),
        edge_index=torch.randint(0, num_nodes, (2, num_edges)),
        edge_attr=torch.stack([
            torch.randint(0, 6, (num_edges,)),      # bond type (0-5)
            torch.randint(0, 3, (num_edges,)),       # bond direction (0-2)
        ], dim=1),
    )


def create_fake_samples(encoder_names, batch_size=2):
    """Create fake training samples dict matching forward() expectations."""
    samples = {
        "question": ["What is the interaction between these two drugs?"] * batch_size,
        "text_input": ["Mild interaction between the compounds."] * batch_size,
    }

    if "gnn" in encoder_names:
        samples["graph"] = [[create_fake_graph(), create_fake_graph()]
                            for _ in range(batch_size)]

    if "image_mol" in encoder_names:
        samples["image"] = [[torch.randn(3, 224, 224), torch.randn(3, 224, 224)]
                            for _ in range(batch_size)]

    return samples


def create_fake_infer_inputs(encoder_names):
    """Create fake inputs for encode_img / encode_img_infer (single compound)."""
    inputs = {}
    if "gnn" in encoder_names:
        inputs["graph"] = Batch.from_data_list([create_fake_graph()])
    if "image_mol" in encoder_names:
        inputs["image"] = torch.randn(1, 3, 224, 224)
    return inputs


# ---------------------------------------------------------------------------
# Mock context manager
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def mock_heavy_dependencies():
    """Patch LLM loading, tokenizer, and encoder weight loading."""
    tiny_llm = LocalLlamaForCausalLM(TINY_LLAMA_CONFIG)

    mock_tok = MockTokenizer()

    with (
        unittest.mock.patch.object(
            LocalLlamaForCausalLM, "from_pretrained",
            return_value=tiny_llm,
        ),
        unittest.mock.patch.object(
            LlamaTokenizer, "from_pretrained",
            return_value=mock_tok,
        ),
        unittest.mock.patch.object(
            GNN, "load_from_pretrained",
            return_value=None,
        ),
        unittest.mock.patch.object(
            ImageMol, "load_from_pretrained",
            return_value=None,
        ),
    ):
        yield


@contextlib.contextmanager
def cpu_device_patch():
    """Redirect torch.device('cuda:*') to CPU for forward pass.

    We can't use unittest.mock.patch because torch.device is a type used in
    isinstance() checks.  Instead we swap it with a proxy class whose metaclass
    redirects cuda->cpu in __call__ and delegates isinstance to the real type.
    """
    _real_cls = torch.device

    class _Meta(type):
        def __instancecheck__(cls, instance):
            return isinstance(instance, _real_cls)

        def __subclasscheck__(cls, subclass):
            if subclass is _real_cls:
                return True
            return type.__subclasscheck__(cls, subclass)

        def __call__(cls, arg, *args, **kwargs):
            if isinstance(arg, str) and "cuda" in arg:
                return _real_cls("cpu")
            return _real_cls(arg, *args, **kwargs)

    class _DeviceProxy(metaclass=_Meta):
        pass

    torch.device = _DeviceProxy
    try:
        yield
    finally:
        torch.device = _real_cls


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------

class TestResult:
    def __init__(self, config_name):
        self.config_name = config_name
        self.results = []  # list of (test_name, status, detail)

    def record(self, test_name, passed, detail=""):
        status = "PASS" if passed else "FAIL"
        self.results.append((test_name, status, detail))

    def skip(self, test_name, detail=""):
        self.results.append((test_name, "SKIP", detail))

    @property
    def all_passed(self):
        return all(s != "FAIL" for _, s, _ in self.results)

    def print(self):
        print(f"\n=== Smoke Test: {self.config_name} ===")
        for name, status, detail in self.results:
            suffix = f" -- {detail}" if detail else ""
            print(f"  [{status}] {name}{suffix}")


def run_tests_for_config(config_name):
    """Run all smoke tests for a single config. Returns TestResult."""
    result = TestResult(config_name)

    # Ensure CWD is project root (for prompts/alignment.txt)
    os.chdir(PROJECT_ROOT)

    model_cfg = load_model_config(config_name)
    encoder_names = list(model_cfg.encoder_names)
    use_mlp = model_cfg.get("use_mlp", False)
    lora_rank = model_cfg.get("lora_rank", 0)
    prompt_tuning = model_cfg.get("prompt_tuning", 0)

    # --- Test 1: Instantiation ---
    try:
        with mock_heavy_dependencies():
            model = DrugChat.from_config(model_cfg)
        model = model.float().cpu()
        result.record("Instantiation", True)
    except Exception as e:
        result.record("Instantiation", False, str(e))
        return result  # can't continue without model

    # --- Test 2: Projector type ---
    try:
        if hasattr(model.llama_proj, "items"):
            # ModuleDict
            for key, proj in model.llama_proj.items():
                if use_mlp:
                    assert isinstance(proj, Mlp), f"expected Mlp for {key}, got {type(proj).__name__}"
                else:
                    assert isinstance(proj, nn.Linear), f"expected Linear for {key}, got {type(proj).__name__}"
        proj_type = "Mlp" if use_mlp else "Linear"
        result.record("Projector type", True, proj_type)
    except Exception as e:
        result.record("Projector type", False, str(e))

    # --- Test 3: LoRA ---
    if lora_rank > 0:
        try:
            assert hasattr(model.llama_model, "peft_config") or hasattr(model.llama_model, "base_model"), \
                "LoRA wrapper not found"
            result.record("LoRA", True, f"rank={lora_rank}")
        except Exception as e:
            result.record("LoRA", False, str(e))
    else:
        result.skip("LoRA", "rank=0")

    # --- Test 4: Prompt tuning ---
    if prompt_tuning > 0:
        try:
            assert model.soft_prompt is not None, "soft_prompt is None"
            expected_shape = (1, prompt_tuning, TINY_HIDDEN)
            assert model.soft_prompt.shape == expected_shape, \
                f"shape {tuple(model.soft_prompt.shape)}, expected {expected_shape}"
            result.record("Prompt tuning", True, f"n={prompt_tuning}, shape={expected_shape}")
        except Exception as e:
            result.record("Prompt tuning", False, str(e))
    else:
        result.skip("Prompt tuning", "n=0")

    # --- Test 5: encode_img ---
    try:
        device = torch.device("cpu")
        inputs = create_fake_infer_inputs(encoder_names)
        with torch.no_grad():
            img_embeds, atts = model.encode_img(inputs, device)

        expected_tokens = 0
        if "gnn" in encoder_names:
            expected_tokens += 1  # use_graph_agg=True -> 1 token
        if "image_mol" in encoder_names:
            expected_tokens += 1  # unsqueeze(1) -> 1 token

        assert img_embeds.ndim == 3, f"ndim={img_embeds.ndim}"
        assert img_embeds.shape[0] == 1, f"batch={img_embeds.shape[0]}"
        assert img_embeds.shape[1] == expected_tokens, \
            f"tokens={img_embeds.shape[1]}, expected {expected_tokens}"
        assert img_embeds.shape[2] == TINY_HIDDEN, \
            f"hidden={img_embeds.shape[2]}, expected {TINY_HIDDEN}"

        result.record("encode_img", True, f"shape={tuple(img_embeds.shape)}")
    except Exception as e:
        result.record("encode_img", False, str(e))

    # --- Test 6: encode_img_infer ---
    try:
        device = torch.device("cpu")
        inputs = create_fake_infer_inputs(encoder_names)
        with torch.no_grad():
            img_embeds, atts = model.encode_img_infer(
                inputs, device, autocast=False, autocast_proj=False
            )

        expected_tokens = 0
        if "gnn" in encoder_names:
            expected_tokens += 1
        if "image_mol" in encoder_names:
            expected_tokens += 1
        expected_tokens += prompt_tuning  # soft prompt tokens appended

        assert img_embeds.shape == (1, expected_tokens, TINY_HIDDEN), \
            f"shape={tuple(img_embeds.shape)}, expected (1, {expected_tokens}, {TINY_HIDDEN})"
        assert atts.shape == (1, expected_tokens), \
            f"atts shape={tuple(atts.shape)}, expected (1, {expected_tokens})"

        result.record("encode_img_infer", True, f"shape={tuple(img_embeds.shape)}")
    except Exception as e:
        result.record("encode_img_infer", False, str(e))

    # --- Test 7: Forward pass ---
    try:
        samples = create_fake_samples(encoder_names, batch_size=2)
        with torch.no_grad(), cpu_device_patch():
            output = model.forward(samples)

        assert "loss" in output, "no 'loss' key in output"
        loss = output["loss"]
        assert isinstance(loss, torch.Tensor), f"loss type: {type(loss)}"
        assert loss.ndim == 0, f"loss ndim={loss.ndim}, expected scalar"
        assert not torch.isnan(loss), "loss is NaN"
        assert not torch.isinf(loss), "loss is Inf"

        result.record("Forward pass", True, f"loss={loss.item():.4f}")
    except Exception as e:
        result.record("Forward pass", False, str(e))

    # --- Test 8: LoRA merge_and_unload (separate, destructive) ---
    if lora_rank > 0:
        try:
            # Re-instantiate to get a fresh model for merge test
            with mock_heavy_dependencies():
                model2 = DrugChat.from_config(model_cfg)
            model2 = model2.float().cpu()
            merged = model2.llama_model.merge_and_unload()
            assert merged is not None
            result.record("LoRA merge_and_unload", True)
            del model2, merged
        except Exception as e:
            result.record("LoRA merge_and_unload", False, str(e))

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Smoke test for ablation configs")
    parser.add_argument("--config", type=str, default=None,
                        help="Run a single config (e.g., gnn_only)")
    args = parser.parse_args()

    configs = [args.config] if args.config else ALL_CONFIGS
    results = []

    for config_name in configs:
        r = run_tests_for_config(config_name)
        r.print()
        results.append(r)

    # Summary
    total_pass = sum(1 for r in results if r.all_passed)
    total = len(results)
    n_tests = sum(len(r.results) for r in results)
    n_passed = sum(1 for r in results for _, s, _ in r.results if s == "PASS")
    n_failed = sum(1 for r in results for _, s, _ in r.results if s == "FAIL")
    n_skipped = sum(1 for r in results for _, s, _ in r.results if s == "SKIP")

    print(f"\n=== Summary ===")
    print(f"{total_pass}/{total} configs: all tests passed")
    print(f"Total: {n_passed} passed, {n_failed} failed, {n_skipped} skipped")

    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    AutoModelForCausalLM, GenerationConfig
)
import os
import torch
import json
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = '/data2/zhaoyang/transformers_cache'
BATCH_SIZE = 32  # You can increase to 32+ if GPU memory allows
DEVICE = "cuda:1"


def load_model(model_name):
    """
    Load tokenizer, model, and generation config for a given model.
    """
    static_prompt_prefix = (
        "You are a helpful medical assistant.\n"
        "Answer the user's question based on the provided context.\n\n"
        "### Context:\n"
    )
    static_prompt_suffix = (
        "\n\n### Question:\nBased on the SMILES of these two drugs, describe their interaction mechanism in one sentence, assuming typical pharmacological behavior. Do not include any reasoning or uncertainty and must give an answer.\n\n"
        "### Answer:"
    )
    # "\n\n### Question:\nBased on the SMILES of these two drugs, describe their interaction mechanism in one sentence, assuming typical pharmacological behavior. Do not include any reasoning or uncertainty and must give an answer.\n\n"

    if model_name == "Medical-mT5-XL":
        model_path = "HiTZ/Medical-mT5-XL"
        tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=CACHE_DIR)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path, cache_dir=CACHE_DIR)

    elif model_name == "MMed-Llama-3-8B":
        model_path = "Henrychur/MMed-Llama-3-8B"
        tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=CACHE_DIR)
        tokenizer.padding_side = 'left'
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, cache_dir=CACHE_DIR)

    elif model_name == "Apollo-MoE-7B":
        model_path = os.path.join('/data2/zhaoyang/model', 'Apollo-MoE-7B')
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokenizer.padding_side = 'left'
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    generation_config = GenerationConfig(
        pad_token_id=tokenizer.pad_token_id,
        num_return_sequences=1,
        max_new_tokens=100,
        min_new_tokens=2,
        do_sample=False,
        num_beams=1
    )

    model = model.to(DEVICE)
    if torch.__version__.startswith("2"):
        model = torch.compile(model)

    return tokenizer, model, generation_config, static_prompt_prefix, static_prompt_suffix



def generate_answer_batch(model, tokenizer, generation_config, prefix, suffix, context_list):
    """
    Generate answers for a batch of contexts.
    """
    prompts = [prefix + context + suffix for context in context_list]
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024
    ).to(DEVICE)

    with torch.inference_mode():
        outputs = model.generate(**inputs, generation_config=generation_config)

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    answers = [s.split("### Answer:")[-1].strip() for s in decoded]
    # print(f"Decoded answers: {answers}")
    # breakpoint()
    return answers


def evaluate_new_dataset(model, tokenizer, generation_config, prefix, suffix, model_name):
    """
    Evaluate the model on the old dataset.
    """
    json_file_path = os.path.join(CURRENT_DIR, "eval_results/old/gpt_results.json")

    with open(json_file_path, "r") as f:
        data = json.load(f)

    results = {}
    smiles_keys = list(data.keys())
    total = len(smiles_keys)

    for i in tqdm(range(0, total, BATCH_SIZE), desc=f"Evaluating {model_name}", ncols=100):
        batch_keys = smiles_keys[i:i + BATCH_SIZE]
        batch_contexts = [
            data[k]["query"].replace(
                "\nBased on the SMILES of these two drugs, describe their interaction mechanism in one sentence, assuming typical pharmacological behavior. Do not include any reasoning or uncertainty.",
                ""
            ) for k in batch_keys
        ]
        batch_answers = generate_answer_batch(
            model, tokenizer, generation_config, prefix, suffix, batch_contexts
        )

        for k, ans in zip(batch_keys, batch_answers):
            results[k] = [
                data[k]["query"],
                data[k]["ground_truth"],
                ans
            ]

    output_file_path = os.path.join(CURRENT_DIR, f"eval_results/old/{model_name}_results.json")
    with open(output_file_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_file_path}")


def evaluate_old_dataset(model, tokenizer, generation_config, prefix, suffix, model_name):
    """
    Evaluate the model on the new dataset.
    """
    json_file_path = os.path.join(CURRENT_DIR, "eval_results/old/gpt_results.json")

    with open(json_file_path, "r") as f:
        data = json.load(f)

    results = {}
    smiles_keys = list(data.keys())
    total = len(smiles_keys)

    for i in tqdm(range(0, total, BATCH_SIZE), desc=f"Evaluating {model_name}", ncols=100):
        # batch_keys = smiles_keys[i:i + BATCH_SIZE]
        # batch_contexts = [
        #     data[k]["query"].replace(
        #         "\nBased on the SMILES of these two drugs, describe their interaction mechanism in one sentence, assuming typical pharmacological behavior. Do not include any reasoning or uncertainty.",
        #         ""
        #     ) for k in batch_keys
        # ]
        # batch_answers = generate_answer_batch(
        #     model, tokenizer, generation_config, prefix, suffix, batch_contexts
        # )
        batch_keys = smiles_keys[i:i + BATCH_SIZE]
        inside_batch_keys = [k for k in batch_keys if k in data]
        batch_contexts = [
            data[k]["query"].replace(
                "\nBased on the SMILES of these two drugs, describe their interaction mechanism in one sentence, assuming typical pharmacological behavior. Do not include any reasoning or uncertainty.",
                ""
            ) for k in inside_batch_keys
        ]

        for k, ans in zip(batch_keys, batch_answers):
            results[k] = [
                data[k]["query"],
                data[k]["ground_truth"],
                ans
            ]

    output_file_path = os.path.join(CURRENT_DIR, f"eval_results/new/{model_name}_results.json")
    with open(output_file_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_file_path}")



def main():
    # models = ["MMed-Llama-3-8B", "Apollo-MoE-7B"]
    # models = ["MMed-Llama-3-8B"]
    models = ["Apollo-MoE-7B"]
    json_file_path = os.path.join(CURRENT_DIR, "eval_results/new/gpt_results.json")

    with open(json_file_path, "r") as f:
        data = json.load(f)
    removed = data.pop("mean_scores", None)
    print(f"Removed mean scores: {removed}")
    print(f"Evaluating {len(data)} samples from the new dataset...")
    for model_name in models:
        tokenizer, model, generation_config, static_prompt_prefix, static_prompt_suffix = load_model(model_name)
        evaluate_new_dataset(model, tokenizer, generation_config, static_prompt_prefix, static_prompt_suffix, model_name)
        print(f"Evaluation for {model_name} completed.")
    # Evaluate the new dataset
    print("Evaluating the new dataset...")
    for model_name in models:
        tokenizer, model, generation_config, static_prompt_prefix, static_prompt_suffix = load_model(model_name)
        evaluate_old_dataset(model, tokenizer, generation_config, static_prompt_prefix, static_prompt_suffix, model_name)
        print(f"Evaluation for {model_name} completed.")
    print("Evaluation completed.")




if __name__ == "__main__":
    print("Starting evaluation...")
    main()

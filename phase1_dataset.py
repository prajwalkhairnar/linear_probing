"""
Phase 1: Activation collection and dataset verification.

Usage:
    python phase1_dataset.py

Expects data/prompts.json to already exist (75 candidate prompt pairs).
Outputs:
    data/dataset.json          — verified prompts, responses, labels
    activations/activations.npy — shape [n_verified, n_layers, d_model]
    activations/labels.npy      — shape [n_verified]
"""

import json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
    ACTIVATIONS_DIR,
    DATA_DIR,
    MAX_NEW_TOKENS,
    MODEL_NAME,
    N_TARGET,
)
from judge import judge_response

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

print(f"Loading tokenizer and model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="cuda",
    torch_dtype=torch.float16,
)
model.eval()

n_layers = model.config.num_hidden_layers
d_model = model.config.hidden_size
print(f"Model loaded — {n_layers} layers, d_model={d_model}")


# ---------------------------------------------------------------------------
# Core collection function
# ---------------------------------------------------------------------------


def collect_activation_and_response(prompt: str) -> tuple[np.ndarray, str]:
    """
    Single forward pass: collect residual stream activations at all layers
    at the final prompt token position, plus the generated response.

    Returns:
        activations: np.ndarray shape [n_layers, d_model]
        response: str
    """

    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    prompt_length = inputs["input_ids"].shape[1]

    layer_activations: dict[int, np.ndarray] = {}
    hooks = []

    def make_hook(layer_idx: int):
        def hook(module, input, output):
            # output[0] shape: [batch, seq_len, d_model]
            hidden = output[0][0, prompt_length - 1, :]
            layer_activations[layer_idx] = hidden.detach().float().cpu().numpy()

        return hook

    for i, layer in enumerate(model.model.layers):
        h = layer.register_forward_hook(make_hook(i))
        hooks.append(h)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    for h in hooks:
        h.remove()

    response_ids = output_ids[0][prompt_length:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True)

    activations = np.stack([layer_activations[i] for i in range(n_layers)])

    return activations, response


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------


def build_dataset(candidates: list) -> tuple[list, np.ndarray, np.ndarray]:
    """
    Run all candidates through the model, judge responses, keep verified examples.
    Target: N_TARGET examples (N_TARGET/2 per class).

    Returns:
        verified_examples: list of dicts
        all_activations: np.ndarray [n_verified, n_layers, d_model]
        all_labels: np.ndarray [n_verified]  0=honest, 1=sycophantic
    """

    verified_examples: list = []
    all_activations: list = []
    all_labels: list = []

    counts = {"sycophantic": 0, "honest": 0}
    target_per_class = N_TARGET // 2

    for pair in candidates:
        for label_name, prompt_key in [
            ("sycophantic", "sycophantic"),
            ("honest", "honest"),
        ]:
            if counts[label_name] >= target_per_class:
                continue

            prompt = pair[prompt_key]
            activations, response = collect_activation_and_response(prompt)
            is_match = judge_response(prompt, response, label_name)

            if is_match:
                label = 1 if label_name == "sycophantic" else 0
                verified_examples.append(
                    {
                        "prompt": prompt,
                        "response": response,
                        "label": label,
                        "label_name": label_name,
                        "topic": pair["topic"],
                    }
                )
                all_activations.append(activations)
                all_labels.append(label)
                counts[label_name] += 1
                print(
                    f"Kept {label_name} example "
                    f"({counts[label_name]}/{target_per_class})"
                )
            else:
                print(f"Discarded: {label_name} response did not match label")

        if all(v >= target_per_class for v in counts.values()):
            break

    return (
        verified_examples,
        np.array(all_activations),   # [n, n_layers, d_model]
        np.array(all_labels),         # [n]
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    prompts_path = f"{DATA_DIR}prompts.json"
    print(f"Loading candidates from {prompts_path}")
    with open(prompts_path) as f:
        candidates = json.load(f)
    print(f"Loaded {len(candidates)} candidate pairs")

    verified_examples, all_activations, all_labels = build_dataset(candidates)

    n_kept = len(verified_examples)
    print(f"\nKept {n_kept} verified examples")
    print(
        f"  sycophantic: {sum(1 for e in verified_examples if e['label'] == 1)}"
    )
    print(
        f"  honest:      {sum(1 for e in verified_examples if e['label'] == 0)}"
    )

    dataset_path = f"{DATA_DIR}dataset.json"
    with open(dataset_path, "w") as f:
        json.dump(verified_examples, f, indent=2)
    print(f"Saved: {dataset_path}")

    activations_path = f"{ACTIVATIONS_DIR}activations.npy"
    labels_path = f"{ACTIVATIONS_DIR}labels.npy"
    np.save(activations_path, all_activations)
    np.save(labels_path, all_labels)
    print(f"Saved: {activations_path}  shape={all_activations.shape}")
    print(f"Saved: {labels_path}  shape={all_labels.shape}")


if __name__ == "__main__":
    main()

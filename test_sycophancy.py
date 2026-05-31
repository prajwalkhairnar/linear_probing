"""
Smoke test: run 2 fake-citation prompts through Gemma-2 and print responses
for manual inspection (no API judge).

Usage:
    python test_sycophancy.py
"""

import json
import textwrap

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import DATA_DIR, DEVICE, DTYPE, MAX_NEW_TOKENS, MODEL_NAME

TEST_IDS = ["hallucination_01", "hallucination_02"]
WIDTH = 72


def _rule(char: str = "=") -> None:
    print(char * WIDTH)


def _section(title: str) -> None:
    print()
    _rule()
    print(title)
    _rule("-")


def _wrap(label: str, text: str) -> None:
    print(f"\n{label}")
    print("-" * len(label))
    for line in textwrap.wrap(text, width=WIDTH):
        print(line)


def generate_response(tokenizer, model, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(DEVICE)
    prompt_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    response_ids = output_ids[0][prompt_length:]
    return tokenizer.decode(response_ids, skip_special_tokens=True)


def main() -> None:
    with open(f"{DATA_DIR}prompts.json") as f:
        by_id = {p["id"]: p for p in json.load(f)}

    print(f"Model:  {MODEL_NAME}")
    print(f"Device: {DEVICE}")
    print(f"Decode: greedy, max_new_tokens={MAX_NEW_TOKENS}")
    print(
        f"Testing {len(TEST_IDS)} hallucination-eliciting prompts — "
        "check manually whether the model treats the fake study as real."
    )

    print("\nLoading model (first run may download weights)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=DTYPE,
        low_cpu_mem_usage=True,
    ).to(DEVICE)
    model.eval()
    print("Model ready.\n")

    for i, pid in enumerate(TEST_IDS, start=1):
        pair = by_id[pid]
        prompt = pair["hallucination"]

        _section(f"[{i}/{len(TEST_IDS)}] {pid}  |  topic: {pair['topic']}")
        print("Expected behaviour: HALLUCINATION SLIP (treat fake study as real)")
        _wrap("Ground truth (what a non-hallucinating answer should say)", pair["correct_answer"])
        _wrap("Hallucination-eliciting prompt (fake citation)", prompt)

        print("\nGenerating...")
        response = generate_response(tokenizer, model, prompt)
        _wrap("Model response", response)

        print()
        print("?" * WIDTH)
        print("  Manual check: did the model treat the fake study as real?")
        print("  (confirm figures, fabricate citation details, cite fake authors)")
        print("?" * WIDTH)

    _rule()
    print("Done. If neither response slips, revisit prompts or model choice")
    print("before running phase1_dataset.py.")
    _rule()


if __name__ == "__main__":
    main()

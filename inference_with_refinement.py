"""
Inference with Multi-Pass Prompt Recovery

Inference pipeline that:

1. Runs multiple prediction passes with different system prompts using 
   instruction tuned Gemma 7B. Each GPU independently predicts for all 
   test rows, producing different results due to stochastic sampling. 
   Results are gathered across GPUs to create diverse candidates per row.
   
2. In a third pass, data is split across GPUs. For each row, generates
   additional candidates at temperatures 0.6 and 1.0, plus comma-truncated
   variants, and selects the best prompt via cross-entropy loss from all
   candidates including hardcoded fallbacks.

Usage:
    accelerate launch inference_with_refinement.py \
        --model_path google/gemma-7b-it \
        --test_csv test.csv
"""

import argparse
import math
import re

import pandas as pd
import torch
from accelerate import Accelerator
from accelerate.utils import gather_object
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Gemma chat templates
USER_CHAT_TEMPLATE = "<start_of_turn>user\n{prompt}<end_of_turn>\n"

# Response prefix used for all prompt predictions
RESPONSE_PREFIX = (
    "The instruction that most likely would have been used to rewrite the text is: "
)

# Default fallback prompt
DEFAULT_PROMPT = (
    "Please improve this text using the writing style with maintaining the "
    "original meaning but altering the tone."
)

# System prompts — different phrasings and examples to get diverse predictions
SYSTEM_PROMPT_V1 = (
    "Given here are an original text and a rewritten text which was rewritten "
    "according to an instruction given to you. What instruction would have been "
    "most likely to be used to transform the given original text into the given "
    "rewritten text? For example, some instructions could be: "
    "**Improve the tone of this text**, **Improve the structure of this text**, "
    "**Rewrite this in a more persuasive manner** etc. "
    "Try to start the instruction with the word 'Improve' or 'Rewrite' or "
    "use the word 'improve' in the instruction if it fits and answer with "
    "only the instruction."
)

SYSTEM_PROMPT_V2 = (
    "Given here are an original text and a rewritten text which was rewritten "
    "according to an instruction given to you. What instruction would have been "
    "most likely to be used to transform the given original text into the given "
    "rewritten text? For example, some instructions could be: "
    "Make this a poem, Rewrite this in the style of a sea shanty, "
    "Rewrite this using the style of Shakespeare etc. "
    "Try to start the instruction with the word 'Improve' or 'Rewrite' or "
    "use the word 'improve' in the instruction if it fits and answer with "
    "only the instruction."
)

SYSTEM_PROMPT_V3 = (
    "Given here are an original text and a rewritten text which was rewritten "
    "according to an instruction given to you. What instruction would have been "
    "most likely to be used to transform the given original text into the given "
    "rewritten text? For example, some instructions could be: "
    "**Make this text rhyme**, **Rewrite this essay but in the style of Dr. Seuss.**, "
    "**Turn this into a sea shanty** etc. "
    "Try to start the instruction with the word 'Improve' or 'Rewrite' or "
    "use the word 'improve' in the instruction if it fits and answer with "
    "only the instruction."
)

# Hardcoded fallback prompts for common rewriting patterns
EXTRA_PROMPTS = [
    "Alter the tone of this text",
    "Add headings and subheadings to this text.",
    "Rewrite this text in a more formal manner.",
    "Improve conciseness by removing unnecessary words.",
    "Improve clarity by simplifying sentences.",
    "Improve the overall effectiveness and persuasiveness of the text.",
]


def clean_text(text: str) -> str:
    """Remove non-alphanumeric characters except spaces, periods, commas."""
    return re.sub(r"[^A-Za-z0-9 .,]+", "", text)


def generate_raw_prediction(
    model, tokenizer, original: str, rewritten: str,
    system_prompt: str, device, temperature: float = 1.0,
) -> str:
    """
    Generate a raw prompt prediction (before cleaning).

    Returns the raw decoded text split at the response prefix and first period,
    preserving commas for later truncation variants.
    """
    prompt = (
        USER_CHAT_TEMPLATE.format(
            prompt=system_prompt
            + "\nOriginal text:" + original[:10000]
            + "\nRewritten Text: " + rewritten[:10000]
        )
        + "<start_of_turn>model\n" + RESPONSE_PREFIX
    )

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    output = model.generate(
        input_ids, max_new_tokens=30,
        do_sample=True, temperature=temperature,
    )

    decoded = tokenizer.decode(output[0])
    result = decoded.split(RESPONSE_PREFIX)[1].split("<eos>")[0].split(".")[0]
    return result


def run_prediction_pass(model, tokenizer, test, system_prompt, device, temperature=1.0):
    """
    Run a prediction pass over ALL test rows on this GPU.

    Each GPU runs independently over the full dataset. Due to stochastic
    sampling, different GPUs produce different predictions for the same rows.
    """
    predictions = []
    for _, row in test.iterrows():
        raw = generate_raw_prediction(
            model, tokenizer,
            str(row["original_text"]),
            str(row["rewritten_text"]),
            system_prompt, device, temperature,
        )
        predictions.append(clean_text(raw))
    return predictions


def format_rewrite_prompt(instruction: str, original: str) -> str:
    """Format a prompt to test how well an instruction reproduces a rewrite."""
    return (
        USER_CHAT_TEMPLATE.format(prompt=instruction + ": " + original)
        + "<start_of_turn>model\n"
        + "Sure, here is the rewritten text: "
    )


def compute_loss(
    model, tokenizer, instruction: str, original: str, target_rewrite: str,
) -> float:
    """
    Compute cross-entropy loss for a candidate prompt.

    Measures how well the instruction, when applied to the original text,
    would reproduce the target rewritten text. Lower loss = better match.
    """
    prompt_full = format_rewrite_prompt(instruction, original) + target_rewrite

    with torch.no_grad():
        inputs = tokenizer.encode(prompt_full, return_tensors="pt")
        labels = tokenizer.encode(" " + target_rewrite, return_tensors="pt")

        full_labels = torch.cat([
            torch.tensor([-100] * (len(inputs[0]) - len(labels[0]) + 1)),
            labels[0][1:],
        ])
        full_labels = full_labels[None, :]

        loss = model(input_ids=inputs, labels=full_labels).loss
        return loss.item()


def run():
    """Main inference pipeline, designed for accelerate launch."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="google/gemma-7b-it")
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--output_csv", type=str, default="submission.csv")
    args, _ = parser.parse_known_args()

    state = Accelerator()
    device = state.device

    # Load test data
    test = pd.read_csv(
        args.test_csv, index_col="id", dtype=str, keep_default_na=False,
    )

    # Load model with 4-bit quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        quantization_config=quantization_config,
        device_map={"": device},
        low_cpu_mem_usage=True,
    )

    # =========================================================================
    # PASS 1: Each GPU predicts ALL rows with system prompt v1. 
    # → anss1 (GPU 0), anss2 (GPU 1) 
    # =========================================================================

    print(f"[GPU {state.process_index}] Pass 1: System prompt v1...")
    pass1_preds = run_prediction_pass(
        model, tokenizer, test, SYSTEM_PROMPT_V1, device, temperature=1.0,
    )

    state.wait_for_everyone()
    gathered_pass1 = gather_object([pass1_preds])
    anss1 = gathered_pass1[0]
    anss2 = gathered_pass1[1] if len(gathered_pass1) > 1 else gathered_pass1[0]


	# =========================================================================
    # PASS 2: Each GPU predicts ALL rows with system prompt v2. 
    # → anss3 (GPU 0), anss4 (GPU 1) 
    # =========================================================================

    print(f"[GPU {state.process_index}] Pass 2: System prompt v2...")
    pass2_preds = run_prediction_pass(
        model, tokenizer, test, SYSTEM_PROMPT_V2, device, temperature=1.0,
    )

    state.wait_for_everyone()
    gathered_pass2 = gather_object([pass2_preds])
    anss3 = gathered_pass2[0]
    anss4 = gathered_pass2[1] if len(gathered_pass2) > 1 else gathered_pass2[0]


    # =========================================================================
    # PASS 3: Split data across GPUs. For each row, generate two more
    # predictions with system prompt v3 at different temperatures and create
    # comma-truncated variants, then select the best via cross-entropy loss.
    # =========================================================================

    print(f"[GPU {state.process_index}] Pass 3: Final predictions + loss-based selection...")

    all_data = {
        "orig": list(test["original_text"]),
        "rewrite": list(test["rewritten_text"]),
        "id": list(test.index),
        "anss1": anss1,
        "anss2": anss2,
        "anss3": anss3,
        "anss4": anss4,
    }

    # Split rows across GPUs
    n = len(all_data["orig"])
    half = math.ceil(n / 2)
    ind = state.process_index
    my_data = {k: v[ind * half:(ind + 1) * half] for k, v in all_data.items()}

    final_answers = []
    for orig, rewrite, test_id, ans1, ans2, ans3, ans4 in zip(
        my_data["orig"], my_data["rewrite"], my_data["id"],
        my_data["anss1"], my_data["anss2"], my_data["anss3"], my_data["anss4"],
    ):
        # Generate at temperature 0.6 → ans5 (full) and ans6 (before first comma)
        raw5 = generate_raw_prediction(
            model, tokenizer, orig, rewrite, SYSTEM_PROMPT_V3, device, temperature=0.6,
        )
        ans5 = clean_text(raw5)
        ans6 = clean_text(raw5.split(",")[0])

        # Generate at temperature 1.0 → ans (full) and ans7 (before first comma)
        raw_t1 = generate_raw_prediction(
            model, tokenizer, orig, rewrite, SYSTEM_PROMPT_V3, device, temperature=1.0,
        )
        ans_full = clean_text(raw_t1)
        ans7 = clean_text(raw_t1.split(",")[0])

        # Select best candidate via cross-entropy loss from all candidates
        candidates = set(
            [ans_full, ans1, ans2, ans3, ans4, ans5, DEFAULT_PROMPT, ans6, ans7]
            + EXTRA_PROMPTS
        )
        best = min(
            candidates,
            key=lambda x: compute_loss(model, tokenizer, x, orig, rewrite),
        )
        best = clean_text(best)

        final_answers.append([test_id, best])
        print(f"  Selected: {best[:80]}")

    # Gather final answers from all GPUs and save
    state.wait_for_everyone()
    gathered = gather_object(final_answers)

    if state.is_main_process:
        try:
            output_df = pd.DataFrame(gathered, columns=["id", "rewrite_prompt"])
            output_df.to_csv(args.output_csv, index=False)
            print(f"\nSaved {len(output_df)} predictions to {args.output_csv}")
        except Exception:
            pass


if __name__ == "__main__":
    run()

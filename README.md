# LLM Prompt Recovery — Bronze Medal Winning Solution 🥉

Solution for the LLM Prompt Recovery competition hosted by Google. The challenge: given an original text and its rewritten version (produced by Google's Gemma), predict the instruction prompt that was used.

The original code from 2024 was cleaned up by AI.


## Approach

The solution uses instruction tuned Gemma 7B (4-bit quantized) with a multi-pass prediction strategy and cross-entropy loss-based selection. Inference is performed on multiple gpus parallelly.

```
┌──────────────────────────────────────────────────────────────────┐
│                    INFERENCE PIPELINE                            │
│                                                                  │
│  1. Multiple prediction passes with diverse system prompts       │
│     using instruction tuned Gemma 7B (4-bit quantized)           │
│  2. Each pass uses different example instructions to guide       │
│     the model toward varied prediction styles                    │
│  3. Additional candidates at temperatures 0.6 and 1.0            │
│  4. Hardcoded fallback prompts for common rewriting patterns     │
│  5. Select best from ~10+ candidates via cross-entropy loss      │
│                                                                  │
│  Cross-entropy loss measures how well each candidate prompt,     │
│  when applied to the original text, reproduces the target        │
│  rewritten text. The lowest-loss candidate wins.                 │
└──────────────────────────────────────────────────────────────────┘
```

### Key Ideas

- **Multi-pass diverse prompting**: Three prediction passes with different system prompts, each containing varied example instructions (tone changes, poetry, style transfer) to produce a diverse candidate set
- **Multi-temperature sampling**: Candidates generated at temperature 0.6 (focused) and 1.0 (diverse) for broader coverage
- **Loss-based selection**: Cross-entropy loss evaluates how well each candidate reproduces the target rewrite — the best candidate is selected automatically
- **Hardcoded fallbacks**: Common rewriting patterns ("Alter the tone", "Add headings", "Improve conciseness") serve as fallback candidates for frequent patterns
- **Multi-GPU**: HuggingFace Accelerate distributes predictions across GPUs

## Setup

```bash
pip install -r requirements.txt
```

**Hardware**: GPU with ≥16GB VRAM (uses 4-bit quantization). Multi-GPU supported via Accelerate.

## Usage

```bash
accelerate launch inference_with_refinement.py \
    --model_path google/gemma-7b-it \
    --test_csv test.csv \
    --output_csv submission.csv
```

## Technical Details

| Component | Details |
|-----------|---------|
| Model | Gemma 7B-IT |
| Quantization | 4-bit (NF4) via bitsandbytes |
| Prediction passes | 3 (different system prompts) |
| Temperatures | 0.6 and 1.0 |
| Candidate pool | ~10+ per sample (predictions + fallbacks) |
| Selection | Cross-entropy loss minimization |
| Multi-GPU | HuggingFace Accelerate |


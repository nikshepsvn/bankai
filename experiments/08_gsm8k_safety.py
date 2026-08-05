"""
Bankai Experiment 7b: GSM8K Safety Check
========================================
Run 50 GSM8K problems with and without patch.
Not testing for improvement — testing no degradation.
"""

import re
import sys
from datasets import load_dataset
from bankai.backends import get_backend
from bankai.patch import Patch, apply_patch, remove_patch

MODEL_PATH = "models/bonsai-8b-mlx"
PATCH_PATH = "patches/calculus_generalized_v1.json"
N = 50

PROMPT = """Solve this math problem. Show your work and end with "The answer is [number]".

{question}"""


def extract_answer(text):
    match = re.search(r'[Tt]he answer is[:\s]*\$?([\-\d,]+)', text)
    if match:
        return match.group(1).replace(',', '')
    match = re.search(r'####\s*([\-\d,]+)', text)
    if match:
        return match.group(1).replace(',', '')
    return None


def extract_gold(answer_text):
    match = re.search(r'####\s*([\-\d,]+)', answer_text)
    return match.group(1).replace(',', '') if match else None


def run_eval(backend, problems, label):
    print(f"\n--- {label} ---")
    correct = 0
    total = 0
    for i, prob in enumerate(problems):
        gold = extract_gold(prob['answer'])
        if not gold:
            continue
        prompt = PROMPT.format(question=prob['question'])
        response = backend.generate(prompt, max_tokens=400)
        predicted = extract_answer(response)
        match = predicted == gold if predicted else False
        if match:
            correct += 1
        total += 1
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{N}] {correct}/{total} correct", flush=True)
    print(f"  Final: {correct}/{total} ({correct/total*100:.1f}%)")
    return correct, total


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"

    backend = get_backend("mlx")
    backend.load(MODEL_PATH)
    ds = load_dataset("openai/gsm8k", "main", split="test")
    problems = list(ds)[:N]

    if mode in ("base", "both"):
        bc, bt = run_eval(backend, problems, "Without patch")

    if mode in ("patch", "both"):
        patch = Patch.load(PATCH_PATH)
        apply_patch(backend, patch)
        pc, pt = run_eval(backend, problems, "With patch")

    if mode == "both":
        print(f"\n  Base:    {bc}/{bt} ({bc/bt*100:.1f}%)")
        print(f"  Patched: {pc}/{pt} ({pc/pt*100:.1f}%)")
        print(f"  Delta:   {(pc/pt - bc/bt)*100:+.1f}%")


if __name__ == "__main__":
    main()

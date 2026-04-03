#!/usr/bin/env python3
"""
DSPy-based privacy classifier optimization.

Uses DSPy to automatically optimize the prompt and few-shot examples
for the privacy classifier. Runs BootstrapFewShot and MIPROv2 optimizers
against the 2260-case dataset with holdout validation.

Usage:
  python3 dspy_optimize.py [--model llama3.2:3b] [--iterations 6]
"""

import dspy
import re
import json
import random
import time
import argparse
import os
import sys
from dataclasses import dataclass

# ============================================================
# DATASET
# ============================================================

sys.path.insert(0, "/tmp")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from large_test_dataset_v2 import TEST_CASES as ALL_CASES
    DATASET_NAME = "large_v2"
except ImportError:
    try:
        from large_test_dataset import TEST_CASES
        from german_test_cases import GERMAN_CASES
        ALL_CASES = list(TEST_CASES) + GERMAN_CASES
        DATASET_NAME = "combined"
    except ImportError:
        from privacy_test_dataset import TEST_CASES
        ALL_CASES = list(TEST_CASES)
        DATASET_NAME = "hand-crafted"

# Deterministic split: 60% train, 20% dev, 20% test
random.seed(42)
indices = list(range(len(ALL_CASES)))
random.shuffle(indices)

n = len(ALL_CASES)
train_end = int(n * 0.6)
dev_end = int(n * 0.8)

TRAIN_SET = [ALL_CASES[i] for i in indices[:train_end]]
DEV_SET = [ALL_CASES[i] for i in indices[train_end:dev_end]]
TEST_SET = [ALL_CASES[i] for i in indices[dev_end:]]

# Convert to DSPy examples
def to_dspy_examples(cases):
    examples = []
    for text, label in cases:
        ex = dspy.Example(text=text, label=label).with_inputs("text")
        examples.append(ex)
    return examples

train_examples = to_dspy_examples(TRAIN_SET)
dev_examples = to_dspy_examples(DEV_SET)
test_examples = to_dspy_examples(TEST_SET)


# ============================================================
# REGEX (fast path — same as v4+)
# ============================================================

CODE_MARKERS = re.compile(
    r"(?:regex|pattern|validate|test fixture|faker|mock|schema|CSV|"
    r"write a|implement|library|generate|def |class |import |r'|r\")",
    re.IGNORECASE
)

STRUCTURAL_PII = {
    "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b',
    "credit_card": r'\b(?:4\d{3}|5[1-5]\d{2}|3[47]\d{2})[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
    "iban": r'\b[A-Z]{2}\d{2}[\s]?\d{4}[\s]?\d{4}[\s]?\d{4}[\s]?\d{4}[\s]?\d{0,2}\b',
    "ssn": r'\b\d{3}[-]\d{2}[-]\d{4}\b',
    "phone_intl": r'\+\d{1,3}\s?\d{2,4}\s?\d{4,8}',
    "phone_de": r'\b0\d{3,4}[\s]?\d{6,8}\b',
    "phone_us": r'\b\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b',
}

def regex_classify(text):
    """Returns SENSITIVE if structural PII found (not in code context), else None."""
    if CODE_MARKERS.search(text):
        return None
    for pattern in STRUCTURAL_PII.values():
        if re.search(pattern, text):
            return "SENSITIVE"
    return None


# ============================================================
# DSPY SIGNATURE & MODULE
# ============================================================

class PrivacyClassify(dspy.Signature):
    """Classify whether a message reveals a SPECIFIC PERSON's private information.

    SENSITIVE: The message contains someone's personal health details, financial data,
    legal situation, relationship issue, confession, contact info, or ID number.
    Messages from institutions about a named person's private data are also SENSITIVE.

    PUBLIC: Even if the TOPIC is sensitive (divorce, medical conditions, salaries),
    if no specific person's private data is revealed, it is PUBLIC.
    General questions, coding, tutorials, and abstract discussions are PUBLIC."""

    text: str = dspy.InputField(desc="The message to classify")
    label: str = dspy.OutputField(desc="SENSITIVE or PUBLIC")


class HybridClassifier(dspy.Module):
    """Hybrid regex + DSPy classifier."""

    def __init__(self):
        super().__init__()
        self.classify = dspy.Predict(PrivacyClassify)

    def forward(self, text):
        # Fast path: regex for structural PII
        regex_result = regex_classify(text)
        if regex_result:
            return dspy.Prediction(label=regex_result)

        # LLM path
        result = self.classify(text=text)
        label = result.label.strip().upper()

        # Normalize
        if "SENSITIVE" in label:
            return dspy.Prediction(label="SENSITIVE")
        elif "PUBLIC" in label:
            return dspy.Prediction(label="PUBLIC")
        else:
            # Failsafe
            return dspy.Prediction(label="SENSITIVE")


# ============================================================
# METRICS
# ============================================================

def classification_metric(example, prediction, trace=None):
    """Basic accuracy metric."""
    pred = prediction.label.strip().upper()
    gold = example.label.strip().upper()
    return pred == gold


def weighted_metric(example, prediction, trace=None):
    """Weighted metric: penalize false negatives (data leaks) 3x more than false positives."""
    pred = prediction.label.strip().upper()
    gold = example.label.strip().upper()
    if pred == gold:
        return 1.0
    # False negative (SENSITIVE classified as PUBLIC) = data leak = very bad
    if gold == "SENSITIVE" and pred == "PUBLIC":
        return -2.0  # heavy penalty
    # False positive (PUBLIC classified as SENSITIVE) = wasteful but safe
    return 0.0


# ============================================================
# EVALUATION
# ============================================================

def evaluate_detailed(classifier, test_set, name=""):
    """Run detailed evaluation with FNR/FPR breakdown."""
    n_s = sum(1 for ex in test_set if ex.label == "SENSITIVE")
    n_p = sum(1 for ex in test_set if ex.label == "PUBLIC")

    correct = 0
    false_negatives = 0
    false_positives = 0
    leaks = []
    fps = []

    for ex in test_set:
        pred = classifier(text=ex.text)
        pred_label = pred.label.strip().upper()
        gold = ex.label.strip().upper()

        if pred_label == gold:
            correct += 1
        elif gold == "SENSITIVE" and pred_label == "PUBLIC":
            false_negatives += 1
            leaks.append(ex.text[:100])
        else:
            false_positives += 1
            fps.append(ex.text[:100])

    total = len(test_set)
    acc = round(correct / total * 100, 1) if total else 0
    fnr = round(false_negatives / n_s * 100, 1) if n_s else 0
    fpr = round(false_positives / n_p * 100, 1) if n_p else 0

    results = {
        "name": name,
        "total": total,
        "correct": correct,
        "accuracy": acc,
        "fnr": fnr,
        "fpr": fpr,
        "false_negatives": false_negatives,
        "false_positives": false_positives,
        "leaks": leaks[:10],
        "fps": fps[:10],
    }

    print(f"\n  {'='*60}")
    print(f"  {name}")
    print(f"  Acc: {acc}%  FNR: {fnr}% ({false_negatives} leaks)  FPR: {fpr}% ({false_positives} waste)")
    print(f"  Total: {total} ({n_s}S, {n_p}P)")
    if leaks:
        print(f"  Top leaks:")
        for l in leaks[:5]:
            print(f"    - {l}")
    print()

    return results


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama3.2:3b")
    parser.add_argument("--api-base", default="http://localhost:11434")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--max-demos", type=int, default=8)
    args = parser.parse_args()

    print("=" * 70)
    print("  DSPY PRIVACY CLASSIFIER OPTIMIZATION")
    print(f"  Model: {args.model}")
    print(f"  Dataset: {DATASET_NAME} ({len(ALL_CASES)} total)")
    print(f"  Train: {len(TRAIN_SET)}  Dev: {len(DEV_SET)}  Test: {len(TEST_SET)}")
    print(f"  Optimizers: BootstrapFewShot (max_demos={args.max_demos})")
    print("=" * 70)

    # Configure DSPy with Ollama
    model_prefix = "ollama_chat" if "qwen" in args.model else "ollama_chat"
    lm = dspy.LM(
        f"{model_prefix}/{args.model}",
        api_base=args.api_base,
        temperature=0,
        max_tokens=200,
    )
    dspy.configure(lm=lm)

    all_results = []

    # --- Baseline: unoptimized ---
    print("\n>>> BASELINE (unoptimized)")
    baseline = HybridClassifier()
    r_baseline = evaluate_detailed(baseline, dev_examples, "Baseline (dev)")
    all_results.append(r_baseline)

    # --- BootstrapFewShot optimization ---
    print(f"\n>>> BOOTSTRAPFEWSHOT (max_demos={args.max_demos})")
    optimizer = dspy.BootstrapFewShot(
        metric=weighted_metric,
        max_bootstrapped_demos=args.max_demos,
        max_labeled_demos=args.max_demos,
        max_rounds=2,
    )

    optimized = optimizer.compile(
        HybridClassifier(),
        trainset=train_examples[:200],  # Use subset for speed
    )

    # Evaluate on dev
    r_opt_dev = evaluate_detailed(optimized, dev_examples, "BootstrapFewShot (dev)")
    all_results.append(r_opt_dev)

    # Evaluate on holdout test
    r_opt_test = evaluate_detailed(optimized, test_examples, "BootstrapFewShot (test/holdout)")
    all_results.append(r_opt_test)

    # --- Try MIPROv2 if available ---
    try:
        print(f"\n>>> MIPROv2 optimization")
        mipro_optimizer = dspy.MIPROv2(
            metric=weighted_metric,
            auto="light",
            num_threads=1,
        )

        mipro_optimized = mipro_optimizer.compile(
            HybridClassifier(),
            trainset=train_examples[:200],
            max_bootstrapped_demos=args.max_demos,
            max_labeled_demos=args.max_demos,
        )

        r_mipro_dev = evaluate_detailed(mipro_optimized, dev_examples, "MIPROv2 (dev)")
        all_results.append(r_mipro_dev)

        r_mipro_test = evaluate_detailed(mipro_optimized, test_examples, "MIPROv2 (test/holdout)")
        all_results.append(r_mipro_test)
    except Exception as e:
        print(f"  MIPROv2 failed: {e}")

    # --- Final comparison ---
    print("\n" + "=" * 70)
    print("  FINAL COMPARISON")
    print("=" * 70)
    print(f"  {'Name':<40s} {'Acc':>5s} {'FNR':>5s} {'FPR':>5s}")
    print(f"  {'-'*60}")
    for r in sorted(all_results, key=lambda x: x["fnr"] + x["fpr"]):
        print(f"  {r['name']:<40s} {r['accuracy']:>4.1f}% {r['fnr']:>4.1f}% {r['fpr']:>4.1f}%")

    # Save optimized program
    try:
        optimized.save("/tmp/dspy_optimized_classifier.json")
        print(f"\n  Optimized program saved to /tmp/dspy_optimized_classifier.json")
    except Exception as e:
        print(f"\n  Save failed: {e}")

    # Save results
    with open("/tmp/dspy_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Results saved to /tmp/dspy_results.json")


if __name__ == "__main__":
    main()

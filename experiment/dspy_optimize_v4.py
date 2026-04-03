#!/usr/bin/env python3
"""
DSPy privacy classifier v4 — seeded with Prompt G, balanced demos.

Fixes from previous runs:
1. Seed DSPy with our proven Prompt G few-shot examples
2. Force balanced demo selection (equal SENSITIVE/PUBLIC)
3. Use full training set, not subset
4. More bootstrap rounds
"""

import dspy
import re
import json
import random
import time
import argparse
import os
import sys

# ============================================================
# DATASET
# ============================================================

sys.path.insert(0, "/tmp")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from large_test_dataset_v2 import TEST_CASES as ALL_CASES
    DATASET_NAME = "large_v2"
except ImportError:
    from privacy_test_dataset import TEST_CASES
    from german_test_cases import GERMAN_CASES
    ALL_CASES = list(TEST_CASES) + GERMAN_CASES
    DATASET_NAME = "combined"

# Deterministic split: 60/20/20
random.seed(42)
indices = list(range(len(ALL_CASES)))
random.shuffle(indices)
n = len(ALL_CASES)
train_end = int(n * 0.6)
dev_end = int(n * 0.8)

TRAIN_SET = [ALL_CASES[i] for i in indices[:train_end]]
DEV_SET = [ALL_CASES[i] for i in indices[train_end:dev_end]]
TEST_SET = [ALL_CASES[i] for i in indices[dev_end:]]

def to_dspy_examples(cases):
    return [dspy.Example(text=t, label=l).with_inputs("text") for t, l in cases]

train_examples = to_dspy_examples(TRAIN_SET)
dev_examples = to_dspy_examples(DEV_SET)
test_examples = to_dspy_examples(TEST_SET)


# ============================================================
# REGEX (same as v4+)
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
    if CODE_MARKERS.search(text):
        return None
    for pattern in STRUCTURAL_PII.values():
        if re.search(pattern, text):
            return "SENSITIVE"
    return None


# ============================================================
# DSPY MODULE — seeded with Prompt G
# ============================================================

class PrivacyClassify(dspy.Signature):
    """Classify whether a message reveals a SPECIFIC PERSON's private information.

    The key question: Does this message reveal a SPECIFIC PERSON's private information?

    SENSITIVE: Yes — it contains someone's personal health details, financial data,
    legal situation, relationship issue, confession, contact info, or ID number.
    Messages from institutions about a named person's private data are also SENSITIVE.

    PUBLIC: No — even if the TOPIC is sensitive (divorce, medical conditions, salaries,
    arrests, bankruptcy), if no specific person's private data is revealed, it is PUBLIC.
    General questions, coding, tutorials, legal/medical information requests, and
    discussions about topics in the abstract are all PUBLIC."""

    text: str = dspy.InputField(desc="The message to classify")
    label: str = dspy.OutputField(desc="SENSITIVE or PUBLIC")


class HybridClassifier(dspy.Module):
    def __init__(self):
        super().__init__()
        self.classify = dspy.Predict(PrivacyClassify)

    def forward(self, text):
        regex_result = regex_classify(text)
        if regex_result:
            return dspy.Prediction(label=regex_result)

        result = self.classify(text=text)
        label = result.label.strip().upper()

        if "SENSITIVE" in label:
            return dspy.Prediction(label="SENSITIVE")
        elif "PUBLIC" in label:
            return dspy.Prediction(label="PUBLIC")
        return dspy.Prediction(label="SENSITIVE")  # failsafe


# Prompt G seed demos — balanced 5 SENSITIVE / 5 PUBLIC
SEED_DEMOS = [
    dspy.Example(text="My credit score dropped to 580 after missing payments", label="SENSITIVE").with_inputs("text"),
    dspy.Example(text="Customer Joe_Schuster53, verify your account 48308813", label="SENSITIVE").with_inputs("text"),
    dspy.Example(text="Mein Gehalt beträgt 65.000 Euro im Jahr", label="SENSITIVE").with_inputs("text"),
    dspy.Example(text="I was diagnosed with diabetes last month", label="SENSITIVE").with_inputs("text"),
    dspy.Example(text="I was arrested for DUI and need a lawyer", label="SENSITIVE").with_inputs("text"),
    dspy.Example(text="What are the penalties for a first-time DUI?", label="PUBLIC").with_inputs("text"),
    dspy.Example(text="What are the grounds for divorce in California?", label="PUBLIC").with_inputs("text"),
    dspy.Example(text="Wie funktioniert das Scheidungsrecht in Deutschland?", label="PUBLIC").with_inputs("text"),
    dspy.Example(text="What medications are commonly prescribed for ADHD?", label="PUBLIC").with_inputs("text"),
    dspy.Example(text="Design a database schema for encrypted credit card data", label="PUBLIC").with_inputs("text"),
]


# ============================================================
# METRICS
# ============================================================

def balanced_metric(example, prediction, trace=None):
    """Balanced metric: correct=1, FN=-2 (leak penalty), FP=0."""
    pred = prediction.label.strip().upper()
    gold = example.label.strip().upper()
    if "SENSITIVE" in pred:
        pred = "SENSITIVE"
    elif "PUBLIC" in pred:
        pred = "PUBLIC"
    else:
        pred = "SENSITIVE"  # failsafe

    if pred == gold:
        return 1.0
    if gold == "SENSITIVE" and pred == "PUBLIC":
        return -2.0  # data leak
    return 0.0  # false positive (safe but wasteful)


# ============================================================
# EVALUATION
# ============================================================

def evaluate_detailed(classifier, test_set, name=""):
    n_s = sum(1 for ex in test_set if ex.label == "SENSITIVE")
    n_p = sum(1 for ex in test_set if ex.label == "PUBLIC")
    correct = fn = fp = 0
    leaks, fps_list = [], []

    for i, ex in enumerate(test_set):
        pred = classifier(text=ex.text)
        pred_label = pred.label.strip().upper()
        if "SENSITIVE" in pred_label:
            pred_label = "SENSITIVE"
        elif "PUBLIC" in pred_label:
            pred_label = "PUBLIC"
        else:
            pred_label = "SENSITIVE"

        gold = ex.label.strip().upper()
        if pred_label == gold:
            correct += 1
        elif gold == "SENSITIVE" and pred_label == "PUBLIC":
            fn += 1
            leaks.append(ex.text[:100])
        else:
            fp += 1
            fps_list.append(ex.text[:100])

        if (i + 1) % 100 == 0:
            print(f"    [{name}] {i+1}/{len(test_set)} done...")

    total = len(test_set)
    acc = round(correct / total * 100, 1)
    fnr = round(fn / n_s * 100, 1) if n_s else 0
    fpr = round(fp / n_p * 100, 1) if n_p else 0

    r = {"name": name, "total": total, "accuracy": acc, "fnr": fnr, "fpr": fpr,
         "false_negatives": fn, "false_positives": fp, "leaks": leaks[:10], "fps": fps_list[:10]}

    print(f"\n  {'='*60}")
    print(f"  {name}")
    print(f"  Acc: {acc}%  FNR: {fnr}% ({fn} leaks)  FPR: {fpr}% ({fp} waste)")
    print(f"  Total: {total} ({n_s}S, {n_p}P)")
    if leaks:
        print(f"  Sample leaks: {leaks[:3]}")
    if fps_list:
        print(f"  Sample FPs: {fps_list[:3]}")
    return r


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama3.2:3b")
    parser.add_argument("--api-base", default="http://localhost:11434")
    parser.add_argument("--max-demos", type=int, default=8)
    args = parser.parse_args()

    print("=" * 70)
    print("  DSPY PRIVACY CLASSIFIER v4 — SEEDED WITH PROMPT G")
    print(f"  Model: {args.model}")
    print(f"  Dataset: {DATASET_NAME} ({len(ALL_CASES)} total)")
    print(f"  Train: {len(TRAIN_SET)}  Dev: {len(DEV_SET)}  Test: {len(TEST_SET)}")
    print("=" * 70)

    lm = dspy.LM(
        f"ollama_chat/{args.model}",
        api_base=args.api_base,
        temperature=0,
        max_tokens=50,
    )
    dspy.configure(lm=lm)

    all_results = []

    # --- Baseline: seeded with Prompt G demos ---
    print("\n>>> BASELINE (Prompt G seed demos)")
    baseline = HybridClassifier()
    # Manually set the seed demos
    baseline.classify.demos = list(SEED_DEMOS)
    r_base_dev = evaluate_detailed(baseline, dev_examples, "Prompt G baseline (dev)")
    all_results.append(r_base_dev)
    r_base_test = evaluate_detailed(baseline, test_examples, "Prompt G baseline (test)")
    all_results.append(r_base_test)

    # --- BootstrapFewShot from seeded baseline ---
    print(f"\n>>> BOOTSTRAPFEWSHOT (seeded, max_demos={args.max_demos})")

    # Create a balanced training subset — equal SENSITIVE/PUBLIC
    train_sensitive = [ex for ex in train_examples if ex.label == "SENSITIVE"]
    train_public = [ex for ex in train_examples if ex.label == "PUBLIC"]
    min_count = min(len(train_sensitive), len(train_public), 200)
    balanced_train = train_sensitive[:min_count] + train_public[:min_count]
    random.shuffle(balanced_train)
    print(f"  Balanced trainset: {len(balanced_train)} ({min_count}S + {min_count}P)")

    # Seed the student with our demos
    student = HybridClassifier()
    student.classify.demos = list(SEED_DEMOS)

    optimizer = dspy.BootstrapFewShot(
        metric=balanced_metric,
        max_bootstrapped_demos=args.max_demos,
        max_labeled_demos=args.max_demos,
        max_rounds=3,
    )

    optimized = optimizer.compile(
        student,
        trainset=balanced_train,
    )

    # Check what demos were selected
    demos = optimized.classify.demos
    print(f"  Optimized demos: {len(demos)} total")
    for d in demos:
        label = d.get("label", "") if isinstance(d, dict) else getattr(d, "label", "")
        text = d.get("text", "") if isinstance(d, dict) else getattr(d, "text", "")
        print(f"    [{label}] {text[:80]}")

    r_opt_dev = evaluate_detailed(optimized, dev_examples, "Bootstrap-seeded (dev)")
    all_results.append(r_opt_dev)
    r_opt_test = evaluate_detailed(optimized, test_examples, "Bootstrap-seeded (test)")
    all_results.append(r_opt_test)

    # --- MIPROv2 from seeded baseline ---
    try:
        print(f"\n>>> MIPROv2 (seeded)")
        student2 = HybridClassifier()
        student2.classify.demos = list(SEED_DEMOS)

        mipro = dspy.MIPROv2(
            metric=balanced_metric,
            auto="light",
            num_threads=1,
        )
        mipro_opt = mipro.compile(
            student2,
            trainset=balanced_train,
            max_bootstrapped_demos=args.max_demos,
            max_labeled_demos=args.max_demos,
        )

        r_mipro_dev = evaluate_detailed(mipro_opt, dev_examples, "MIPROv2-seeded (dev)")
        all_results.append(r_mipro_dev)
        r_mipro_test = evaluate_detailed(mipro_opt, test_examples, "MIPROv2-seeded (test)")
        all_results.append(r_mipro_test)
    except Exception as e:
        print(f"  MIPROv2 failed: {e}")

    # --- Final comparison ---
    print("\n" + "=" * 70)
    print("  FINAL COMPARISON")
    print("=" * 70)
    print(f"  Manual Prompt G holdout (v5): Acc 96.7%, FNR 4.6%, FPR 2.3% (241 cases)")
    print(f"  {'Name':<40s} {'Acc':>5s} {'FNR':>5s} {'FPR':>5s}")
    print(f"  {'-'*60}")
    for r in sorted(all_results, key=lambda x: x["fnr"] + x["fpr"]):
        print(f"  {r['name']:<40s} {r['accuracy']:>4.1f}% {r['fnr']:>4.1f}% {r['fpr']:>4.1f}%")

    # Save best
    test_results = [r for r in all_results if "test" in r["name"]]
    if test_results:
        best = min(test_results, key=lambda r: r["fnr"] + r["fpr"] * 0.5)
        with open("/tmp/dspy_best_result_v4.json", "w") as f:
            json.dump(best, f, indent=2)
        print(f"\n  Best test result saved to /tmp/dspy_best_result_v4.json")

    try:
        optimized.save("/tmp/dspy_optimized_v4.json")
        print(f"  Optimized program saved to /tmp/dspy_optimized_v4.json")
    except Exception as e:
        print(f"  Save failed: {e}")

    with open("/tmp/dspy_results_v4.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  All results saved to /tmp/dspy_results_v4.json")


if __name__ == "__main__":
    main()

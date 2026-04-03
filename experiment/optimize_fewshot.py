#!/usr/bin/env python3
"""
Custom few-shot optimizer — hill-climbing search for best examples.
Uses our raw prompt format (NOT DSPy's structured format).
Karpathy autoresearch: try → measure → keep/discard → iterate.
"""

import re
import json
import time
import random
import requests
import sys
import os
import copy

OLLAMA_URL = "http://localhost:11434"

sys.path.insert(0, "/tmp")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from large_test_dataset_v2 import TEST_CASES as ALL_CASES

# 60/20/20 split
random.seed(42)
indices = list(range(len(ALL_CASES)))
random.shuffle(indices)
n = len(ALL_CASES)
TRAIN = [ALL_CASES[i] for i in indices[:int(n*0.6)]]
DEV = [ALL_CASES[i] for i in indices[int(n*0.6):int(n*0.8)]]
TEST = [ALL_CASES[i] for i in indices[int(n*0.8):]]

# Separate train by label
TRAIN_S = [(t, l) for t, l in TRAIN if l == "SENSITIVE"]
TRAIN_P = [(t, l) for t, l in TRAIN if l == "PUBLIC"]

MODEL = os.environ.get("MODEL", "gemma3:1b")

# ============================================================
# REGEX
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
    for p in STRUCTURAL_PII.values():
        if re.search(p, text):
            return "SENSITIVE"
    return None

# ============================================================
# PROMPT TEMPLATE
# ============================================================

SYSTEM = """You are a privacy classifier. Respond with SENSITIVE or PUBLIC only.

The key question: Does this message reveal a SPECIFIC PERSON's private information?

SENSITIVE: Yes — it contains someone's personal health details, financial data, legal situation, relationship issue, confession, contact info, or ID number. Messages from institutions about a named person's private data are also SENSITIVE.

PUBLIC: No — even if the TOPIC is sensitive (divorce, medical conditions, salaries, arrests, bankruptcy), if no specific person's private data is revealed, it is PUBLIC. General questions, coding, tutorials, legal/medical information requests, and discussions about topics in the abstract are all PUBLIC."""

def build_user_prompt(examples, text):
    lines = ["Examples:"]
    for ex_text, ex_label in examples:
        lines.append(f'Message: "{ex_text[:150]}" -> {ex_label}')
    lines.append(f'\nMessage: "{text}"\nClassification:')
    return "\n".join(lines)


def classify(text, examples):
    regex = regex_classify(text)
    if regex:
        return regex, 0.0

    prompt = build_user_prompt(examples, text)
    start = time.time()
    try:
        resp = requests.post(f"{OLLAMA_URL}/api/chat", json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": 0, "num_predict": 10}
        }, timeout=60)
        raw = resp.json().get("message", {}).get("content", "").strip().upper()
        latency = (time.time() - start) * 1000
        if "SENSITIVE" in raw:
            return "SENSITIVE", latency
        elif "PUBLIC" in raw:
            return "PUBLIC", latency
        return "SENSITIVE", latency  # failsafe
    except:
        return "SENSITIVE", (time.time() - start) * 1000


def evaluate(examples, test_set, label=""):
    n_s = sum(1 for _, l in test_set if l == "SENSITIVE")
    n_p = sum(1 for _, l in test_set if l == "PUBLIC")
    correct = fn = fp = 0
    for i, (text, expected) in enumerate(test_set):
        predicted, _ = classify(text, examples)
        if predicted == expected:
            correct += 1
        elif expected == "SENSITIVE":
            fn += 1
        else:
            fp += 1
        if (i+1) % 100 == 0:
            print(f"    [{label}] {i+1}/{len(test_set)}")

    acc = round(correct / len(test_set) * 100, 1)
    fnr = round(fn / n_s * 100, 1) if n_s else 0
    fpr = round(fp / n_p * 100, 1) if n_p else 0
    score = fnr * 2 + fpr  # weighted: FNR matters 2x
    return {"acc": acc, "fnr": fnr, "fpr": fpr, "fn": fn, "fp": fp, "score": score}


# ============================================================
# SEED EXAMPLES (Prompt G)
# ============================================================

SEED = [
    ("My credit score dropped to 580 after missing payments", "SENSITIVE"),
    ("Customer Joe_Schuster53, verify your account 48308813", "SENSITIVE"),
    ("Mein Gehalt beträgt 65.000 Euro im Jahr", "SENSITIVE"),
    ("I was diagnosed with diabetes last month", "SENSITIVE"),
    ("What are the penalties for a first-time DUI?", "PUBLIC"),
    ("What are the grounds for divorce in California?", "PUBLIC"),
    ("Wie funktioniert das Scheidungsrecht in Deutschland?", "PUBLIC"),
    ("What medications are commonly prescribed for ADHD?", "PUBLIC"),
    ("How much does a software engineer earn in Munich?", "PUBLIC"),
    ("Design a database schema for encrypted credit card data", "PUBLIC"),
]


def main():
    print("=" * 65)
    print(f"  FEW-SHOT OPTIMIZER — hill climbing")
    print(f"  Model: {MODEL}")
    print(f"  Train: {len(TRAIN)} ({len(TRAIN_S)}S, {len(TRAIN_P)}P)")
    print(f"  Dev: {len(DEV)}, Test: {len(TEST)}")
    print(f"  Seed examples: {len(SEED)}")
    print("=" * 65)

    # Warmup
    classify("warmup", SEED)

    # Baseline
    print("\n>>> Baseline (seed examples)")
    best_examples = list(SEED)
    best = evaluate(best_examples, DEV, "baseline")
    print(f"  Acc: {best['acc']}%  FNR: {best['fnr']}%  FPR: {best['fpr']}%  Score: {best['score']}")

    # Hill climbing: try swapping/adding examples from training set
    MAX_ITERATIONS = 12
    MAX_EXAMPLES = 12  # max few-shot examples

    for iteration in range(MAX_ITERATIONS):
        print(f"\n>>> Iteration {iteration + 1}/{MAX_ITERATIONS}")

        # Strategy: randomly swap one example or add one
        candidates = []

        # Strategy 1: swap a random SENSITIVE example
        for _ in range(3):
            new_ex = list(best_examples)
            s_indices = [i for i, (_, l) in enumerate(new_ex) if l == "SENSITIVE"]
            if s_indices:
                idx = random.choice(s_indices)
                replacement = random.choice(TRAIN_S)
                new_ex[idx] = replacement
                candidates.append(("swap_s", new_ex))

        # Strategy 2: swap a random PUBLIC example
        for _ in range(3):
            new_ex = list(best_examples)
            p_indices = [i for i, (_, l) in enumerate(new_ex) if l == "PUBLIC"]
            if p_indices:
                idx = random.choice(p_indices)
                replacement = random.choice(TRAIN_P)
                new_ex[idx] = replacement
                candidates.append(("swap_p", new_ex))

        # Strategy 3: add an example (if under limit)
        if len(best_examples) < MAX_EXAMPLES:
            new_ex = list(best_examples) + [random.choice(TRAIN_S)]
            candidates.append(("add_s", new_ex))
            new_ex = list(best_examples) + [random.choice(TRAIN_P)]
            candidates.append(("add_p", new_ex))

        # Evaluate each candidate on a quick subset of dev (100 cases)
        dev_subset = random.sample(DEV, min(100, len(DEV)))
        best_candidate = None
        best_candidate_score = best["score"]

        for name, cand_examples in candidates:
            r = evaluate(cand_examples, dev_subset, f"trial-{name}")
            print(f"    {name}: Acc={r['acc']}% FNR={r['fnr']}% FPR={r['fpr']}% Score={r['score']:.1f}")
            if r["score"] < best_candidate_score:
                best_candidate_score = r["score"]
                best_candidate = (name, cand_examples)

        if best_candidate:
            name, new_examples = best_candidate
            # Verify on full dev set
            print(f"\n  Best candidate: {name} (quick score: {best_candidate_score:.1f})")
            print(f"  Verifying on full dev set...")
            full_result = evaluate(new_examples, DEV, "full-dev")
            print(f"  Full: Acc={full_result['acc']}% FNR={full_result['fnr']}% FPR={full_result['fpr']}% Score={full_result['score']:.1f}")

            if full_result["score"] < best["score"]:
                print(f"  IMPROVED: {best['score']:.1f} -> {full_result['score']:.1f}")
                best = full_result
                best_examples = new_examples
            else:
                print(f"  No improvement on full dev (was {best['score']:.1f})")
        else:
            print(f"  No candidate beat current best ({best['score']:.1f})")

    # Final: evaluate best on holdout test
    print(f"\n>>> FINAL: Evaluating best examples on holdout test set")
    test_result = evaluate(best_examples, TEST, "holdout")
    print(f"\n{'='*65}")
    print(f"  FINAL RESULTS")
    print(f"{'='*65}")
    print(f"  Dev:     Acc={best['acc']}% FNR={best['fnr']}% FPR={best['fpr']}%")
    print(f"  Holdout: Acc={test_result['acc']}% FNR={test_result['fnr']}% FPR={test_result['fpr']}%")
    print(f"\n  Best examples ({len(best_examples)}):")
    for text, label in best_examples:
        print(f"    [{label}] {text[:100]}")

    # Save
    with open("/tmp/optimized_fewshot.json", "w") as f:
        json.dump({
            "model": MODEL,
            "system": SYSTEM,
            "examples": best_examples,
            "dev": best,
            "holdout": test_result,
        }, f, indent=2)
    print(f"\n  Saved to /tmp/optimized_fewshot.json")


if __name__ == "__main__":
    main()

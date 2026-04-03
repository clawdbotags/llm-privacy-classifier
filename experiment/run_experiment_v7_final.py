#!/usr/bin/env python3
"""
Final validation of Prompt G on the 2260-case dataset.
Uses our raw Ollama prompt format (NOT DSPy).
60/20/20 train/dev/test split. Only evaluates on test (holdout).
"""

import re
import json
import time
import requests
import sys
import os
import random

OLLAMA_URL = "http://localhost:11434"

sys.path.insert(0, "/tmp")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from large_test_dataset_v2 import TEST_CASES as ALL_CASES
print(f"Loaded {len(ALL_CASES)} cases")

# 60/20/20 split (same seed as DSPy experiments for comparability)
random.seed(42)
indices = list(range(len(ALL_CASES)))
random.shuffle(indices)
n = len(ALL_CASES)
train_end = int(n * 0.6)
dev_end = int(n * 0.8)
TEST_SET = [ALL_CASES[i] for i in indices[dev_end:]]
DEV_SET = [ALL_CASES[i] for i in indices[train_end:dev_end]]

# Also run on full dataset for completeness
FULL_SET = ALL_CASES

MODELS = ["llama3.2:3b"]

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

def classify_regex(text):
    if CODE_MARKERS.search(text):
        return "UNCERTAIN", []
    matches = [n for n, p in STRUCTURAL_PII.items() if re.search(p, text)]
    return ("SENSITIVE", matches) if matches else ("UNCERTAIN", [])


# ============================================================
# PROMPT G (our winner)
# ============================================================

SYSTEM_PROMPT = """You are a privacy classifier. Respond with SENSITIVE or PUBLIC only.

The key question: Does this message reveal a SPECIFIC PERSON's private information?

SENSITIVE: Yes — it contains someone's personal health details, financial data, legal situation, relationship issue, confession, contact info, or ID number. Messages from institutions about a named person's private data are also SENSITIVE.

PUBLIC: No — even if the TOPIC is sensitive (divorce, medical conditions, salaries, arrests, bankruptcy), if no specific person's private data is revealed, it is PUBLIC. General questions, coding, tutorials, legal/medical information requests, and discussions about topics in the abstract are all PUBLIC."""

USER_TEMPLATE = '''Examples:
Message: "My credit score dropped to 580 after missing payments" -> SENSITIVE
Message: "Customer Joe_Schuster53, verify your account 48308813" -> SENSITIVE
Message: "Mein Gehalt beträgt 65.000 Euro im Jahr" -> SENSITIVE
Message: "I was diagnosed with diabetes last month" -> SENSITIVE
Message: "What are the penalties for a first-time DUI?" -> PUBLIC
Message: "What are the grounds for divorce in California?" -> PUBLIC
Message: "Wie funktioniert das Scheidungsrecht in Deutschland?" -> PUBLIC
Message: "What medications are commonly prescribed for ADHD?" -> PUBLIC
Message: "How much does a software engineer earn in Munich?" -> PUBLIC
Message: "Design a database schema for encrypted credit card data" -> PUBLIC

Message: "{text}"
Classification:'''


def classify_llm(text, model):
    prompt = USER_TEMPLATE.format(text=text)
    start = time.time()
    try:
        resp = requests.post(f"{OLLAMA_URL}/api/chat", json={
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": 0, "num_predict": 10}
        }, timeout=120)
        raw = resp.json().get("message", {}).get("content", "").strip().upper()
        latency = (time.time() - start) * 1000
        if "SENSITIVE" in raw:
            return "SENSITIVE", latency
        elif "PUBLIC" in raw:
            return "PUBLIC", latency
        return "UNCERTAIN", latency
    except Exception as e:
        return f"ERROR:{e}", (time.time() - start) * 1000


def classify_hybrid(text, model):
    regex_result, matches = classify_regex(text)
    if regex_result == "SENSITIVE":
        return "SENSITIVE", f"regex:{','.join(matches)}", 0.0
    llm_result, latency = classify_llm(text, model)
    if llm_result in ("UNCERTAIN",) or llm_result.startswith("ERROR"):
        return "SENSITIVE", "failsafe", latency
    return llm_result, f"llm:{llm_result.lower()}", latency


def run_eval(name, test_cases, model):
    n_s = sum(1 for _, e in test_cases if e == "SENSITIVE")
    n_p = sum(1 for _, e in test_cases if e == "PUBLIC")
    correct = fn = fp = 0
    leaks, fps_list = [], []
    total_latency = 0
    llm_calls = 0

    for i, (text, expected) in enumerate(test_cases):
        predicted, method, latency = classify_hybrid(text, model)
        total_latency += latency
        if latency > 0:
            llm_calls += 1

        if predicted == expected:
            correct += 1
        elif expected == "SENSITIVE" and predicted == "PUBLIC":
            fn += 1
            leaks.append({"text": text[:120], "method": method})
        else:
            fp += 1
            fps_list.append({"text": text[:120], "method": method})

        if (i + 1) % 100 == 0:
            print(f"    [{name}] {i+1}/{len(test_cases)} done...")

    total = len(test_cases)
    acc = round(correct / total * 100, 1)
    fnr = round(fn / n_s * 100, 1) if n_s else 0
    fpr = round(fp / n_p * 100, 1) if n_p else 0
    avg_lat = round(total_latency / llm_calls, 1) if llm_calls else 0

    print(f"\n  {'='*65}")
    print(f"  {name}")
    print(f"  {'='*65}")
    print(f"  Accuracy:            {acc}% ({correct}/{total})")
    print(f"  False Negative Rate: {fnr}% ({fn} SENSITIVE leaked as PUBLIC)")
    print(f"  False Positive Rate: {fpr}% ({fp} PUBLIC routed local)")
    print(f"  LLM calls:           {llm_calls} / {total}")
    print(f"  Avg LLM latency:     {avg_lat}ms")
    print(f"  Dataset:             {total} cases ({n_s}S, {n_p}P)")

    if leaks:
        print(f"\n  DATA LEAKS ({fn}):")
        for l in leaks[:15]:
            print(f"    [{l['method']}] {l['text']}")
    if fps_list:
        print(f"\n  FALSE POSITIVES ({fp}):")
        for f in fps_list[:15]:
            print(f"    [{f['method']}] {f['text']}")

    return {
        "name": name, "total": total, "accuracy": acc,
        "fnr": fnr, "fpr": fpr, "false_negatives": fn, "false_positives": fp,
        "avg_latency_ms": avg_lat, "llm_calls": llm_calls,
        "leaks": leaks, "fps": fps_list,
    }


def main():
    model = "llama3.2:3b"

    print("=" * 65)
    print("  FINAL VALIDATION — Prompt G on 2260-case dataset")
    print(f"  Model: {model}")
    print(f"  Full dataset: {len(ALL_CASES)} cases")
    print(f"  Dev set: {len(DEV_SET)} cases")
    print(f"  Holdout test: {len(TEST_SET)} cases")
    print("=" * 65)

    # Warm up
    print("\n  Warming up model...")
    classify_llm("warmup test", model)

    all_results = []

    # Dev set
    print(f"\n>>> DEV SET ({len(DEV_SET)} cases)")
    r_dev = run_eval("Prompt G (dev)", DEV_SET, model)
    all_results.append(r_dev)

    # Holdout test set
    print(f"\n>>> HOLDOUT TEST SET ({len(TEST_SET)} cases)")
    r_test = run_eval("Prompt G (holdout test)", TEST_SET, model)
    all_results.append(r_test)

    # Full dataset
    print(f"\n>>> FULL DATASET ({len(FULL_SET)} cases)")
    r_full = run_eval("Prompt G (full 2260)", FULL_SET, model)
    all_results.append(r_full)

    # Summary
    print("\n" + "=" * 65)
    print("  FINAL SUMMARY")
    print("=" * 65)
    print(f"  {'Dataset':<30s} {'Acc':>5s} {'FNR':>5s} {'FPR':>5s} {'N':>6s}")
    print(f"  {'-'*55}")
    for r in all_results:
        print(f"  {r['name']:<30s} {r['accuracy']:>4.1f}% {r['fnr']:>4.1f}% {r['fpr']:>4.1f}% {r['total']:>6d}")

    # Save
    save_results = [{k: v for k, v in r.items() if k not in ("leaks", "fps")} for r in all_results]
    with open("/tmp/final_validation_results.json", "w") as f:
        json.dump(save_results, f, indent=2)

    # Save full results with leaks for analysis
    with open("/tmp/final_validation_full.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n  Results saved to /tmp/final_validation_results.json")
    print(f"  Full results (with leaks) saved to /tmp/final_validation_full.json")


if __name__ == "__main__":
    main()

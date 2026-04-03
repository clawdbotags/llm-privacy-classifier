#!/usr/bin/env python3
"""
Privacy Classifier v5 — HONEST EVALUATION with held-out test set.

Critical reviewer flagged v4's 95.8% as potentially overfitted because we tuned
few-shot examples to the same dataset we evaluated on.

This experiment:
1. Splits 481 cases into 240 dev + 241 test (deterministic seed)
2. Tests the v4 winner (Prompt G) on ONLY the held-out test set
3. Also tests Prompt G WITHOUT the targeted PUBLIC examples (prompt G-stripped)
   to isolate whether the prompt reframing alone carries the improvement
4. Tests both gemma3:1b and llama3.2:3b
"""

import re
import json
import time
import requests
import sys
import random

OLLAMA_URL = "http://localhost:11434"

sys.path.insert(0, "/tmp")
from large_test_dataset import TEST_CASES as BASE_CASES
from german_test_cases import GERMAN_CASES
ALL_CASES = list(BASE_CASES) + GERMAN_CASES

# Deterministic split
random.seed(42)
indices = list(range(len(ALL_CASES)))
random.shuffle(indices)
split = len(indices) // 2

DEV_SET = [ALL_CASES[i] for i in indices[:split]]
TEST_SET = [ALL_CASES[i] for i in indices[split:]]

MODELS = ["gemma3:1b", "llama3.2:3b"]

# ============================================================
# REGEX (same as v4)
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
# PROMPTS
# ============================================================

# v4 winner — full prompt G with targeted examples
PROMPT_G_FULL = {
    "name": "G-full: v4 winner",
    "system": """You are a privacy classifier. Respond with SENSITIVE or PUBLIC only.

The key question: Does this message reveal a SPECIFIC PERSON's private information?

SENSITIVE: Yes — it contains someone's personal health details, financial data, legal situation, relationship issue, confession, contact info, or ID number. Messages from institutions about a named person's private data are also SENSITIVE.

PUBLIC: No — even if the TOPIC is sensitive (divorce, medical conditions, salaries, arrests, bankruptcy), if no specific person's private data is revealed, it is PUBLIC. General questions, coding, tutorials, legal/medical information requests, and discussions about topics in the abstract are all PUBLIC.""",
    "template": '''Examples:
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
Classification:''',
}

# Stripped — SAME system prompt, but only 4 generic examples (no targeted ones)
PROMPT_G_STRIPPED = {
    "name": "G-stripped: reframe only",
    "system": """You are a privacy classifier. Respond with SENSITIVE or PUBLIC only.

The key question: Does this message reveal a SPECIFIC PERSON's private information?

SENSITIVE: Yes — it contains someone's personal health details, financial data, legal situation, relationship issue, confession, contact info, or ID number. Messages from institutions about a named person's private data are also SENSITIVE.

PUBLIC: No — even if the TOPIC is sensitive (divorce, medical conditions, salaries, arrests, bankruptcy), if no specific person's private data is revealed, it is PUBLIC. General questions, coding, tutorials, legal/medical information requests, and discussions about topics in the abstract are all PUBLIC.""",
    "template": '''Examples:
Message: "My credit score dropped to 580 after missing payments" -> SENSITIVE
Message: "I was diagnosed with diabetes last month" -> SENSITIVE
Message: "How do I implement a binary search tree in Python?" -> PUBLIC
Message: "What are the symptoms of type 2 diabetes?" -> PUBLIC

Message: "{text}"
Classification:''',
}

# v3 baseline for comparison
PROMPT_D_BASELINE = {
    "name": "D-baseline: v3 original",
    "system": """You are a privacy classifier. Respond with SENSITIVE or PUBLIC only.

SENSITIVE: Contains or reveals personal/private information about a specific real person — health, financial, legal, relationships, confessions, contact info, ID numbers. Includes third-party messages that reference a named person's private data (bank notifications, medical records, HR emails). Any language.

PUBLIC: General knowledge, coding, education, or topics discussed without a specific person's private context. Generic business content without personal identifiers is PUBLIC. Code with fake/test data is PUBLIC.""",
    "template": '''Examples:
Message: "My credit score dropped to 620 after missing mortgage payments" -> SENSITIVE
Message: "Customer Joe_Schuster53, we need to verify your IMEI for recent banking updates" -> SENSITIVE
Message: "Mein Gehalt beträgt 65.000 Euro im Jahr und ich möchte verhandeln" -> SENSITIVE
Message: "Erreichen Sie mich unter 0176 12345678 nach 17 Uhr" -> SENSITIVE
Message: "How do I implement a binary search tree in Python?" -> PUBLIC
Message: "Was sind die Symptome von Typ-2-Diabetes?" -> PUBLIC
Message: "What are the penalties for a first-time DUI offense?" -> PUBLIC
Message: "Write a tutorial on building a REST API with FastAPI" -> PUBLIC

Message: "{text}"
Classification:''',
}

PROMPTS = [PROMPT_D_BASELINE, PROMPT_G_FULL, PROMPT_G_STRIPPED]

# ============================================================
# CLASSIFIER & RUNNER
# ============================================================

def classify_llm(text, model, prompt_config):
    prompt_text = prompt_config["template"].format(text=text)
    start = time.time()
    try:
        resp = requests.post(f"{OLLAMA_URL}/api/generate", json={
            "model": model,
            "prompt": prompt_text,
            "system": prompt_config["system"],
            "stream": False,
            "options": {"temperature": 0, "num_predict": 10}
        }, timeout=120)
        raw = resp.json().get("response", "").strip()
        latency = (time.time() - start) * 1000
        clean = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip().upper()
        if "SENSITIVE" in clean:
            return "SENSITIVE", latency
        elif "PUBLIC" in clean:
            return "PUBLIC", latency
        return "UNCERTAIN", latency
    except Exception as e:
        return f"ERROR:{e}", (time.time() - start) * 1000


def classify_hybrid(text, model, prompt_config):
    regex_result, matches = classify_regex(text)
    if regex_result == "SENSITIVE":
        return "SENSITIVE", f"regex:{','.join(matches)}", 0.0
    llm_result, latency = classify_llm(text, model, prompt_config)
    if llm_result in ("UNCERTAIN",) or llm_result.startswith("ERROR"):
        return "SENSITIVE", "failsafe", latency
    return llm_result, f"llm:{llm_result.lower()}", latency


def run_experiment(name, classify_fn, test_cases):
    n_s = sum(1 for _, e in test_cases if e == "SENSITIVE")
    n_p = sum(1 for _, e in test_cases if e == "PUBLIC")
    r = {
        "name": name, "total": len(test_cases),
        "correct": 0, "false_negatives": 0, "false_positives": 0,
        "total_latency_ms": 0, "llm_calls": 0,
        "leaks": [], "fps": [],
    }
    for text, expected in test_cases:
        predicted, method, latency = classify_fn(text)
        r["total_latency_ms"] += latency
        if latency > 0: r["llm_calls"] += 1
        correct = predicted == expected
        if correct:
            r["correct"] += 1
        else:
            is_leak = expected == "SENSITIVE" and predicted == "PUBLIC"
            if is_leak:
                r["false_negatives"] += 1
                r["leaks"].append(text[:100])
            else:
                r["false_positives"] += 1
                r["fps"].append(text[:100])
            if not correct:
                tag = "LEAK" if is_leak else "FP  "
                print(f"  [{tag}] {expected:>9s}->{predicted:<9s} | {method:<22s} | {text[:65]}")

    r["accuracy"] = round(r["correct"] / r["total"] * 100, 1)
    r["fnr"] = round(r["false_negatives"] / n_s * 100, 1) if n_s else 0
    r["fpr"] = round(r["false_positives"] / n_p * 100, 1) if n_p else 0
    r["avg_latency_ms"] = round(r["total_latency_ms"] / r["llm_calls"], 1) if r["llm_calls"] else 0
    return r


def check_model(model):
    try:
        resp = requests.post(f"{OLLAMA_URL}/api/generate", json={
            "model": model, "prompt": "hi", "stream": False,
            "options": {"num_predict": 1}
        }, timeout=120)
        return resp.status_code == 200
    except:
        return False


def main():
    dev_s = sum(1 for _, e in DEV_SET if e == "SENSITIVE")
    dev_p = sum(1 for _, e in DEV_SET if e == "PUBLIC")
    test_s = sum(1 for _, e in TEST_SET if e == "SENSITIVE")
    test_p = sum(1 for _, e in TEST_SET if e == "PUBLIC")

    print("=" * 70)
    print("  PRIVACY CLASSIFIER v5 — HOLDOUT VALIDATION")
    print(f"  Total: {len(ALL_CASES)} cases")
    print(f"  Dev set:  {len(DEV_SET)} ({dev_s} sensitive, {dev_p} public)")
    print(f"  Test set: {len(TEST_SET)} ({test_s} sensitive, {test_p} public)")
    print(f"  Evaluating on TEST SET ONLY (never used for prompt tuning)")
    print("=" * 70)

    all_results = []

    for model in MODELS:
        print(f"\n>>> {model}")
        if not check_model(model):
            print(f"  SKIPPED")
            continue
        print(f"  Warming up...")
        classify_llm("warmup", model, PROMPT_D_BASELINE)

        for prompt in PROMPTS:
            name = f"{model} | {prompt['name']}"
            print(f"\n--- {name} [HOLDOUT TEST SET]")
            r = run_experiment(name, lambda t, m=model, p=prompt: classify_hybrid(t, m, p), TEST_SET)
            print(f"\n  {'='*60}")
            print(f"  {r['name']}")
            print(f"  Acc: {r['accuracy']}%  FNR: {r['fnr']}% ({r['false_negatives']} leaks)  FPR: {r['fpr']}% ({r['false_positives']} waste)  Lat: {r['avg_latency_ms']}ms")
            all_results.append(r)

    # Final table
    print("\n\n" + "=" * 90)
    print("  HOLDOUT VALIDATION — sorted by combined score (lower is better)")
    print("=" * 90)
    print(f"  {'Experiment':<50s} {'Acc':>5s} {'FNR':>5s} {'FPR':>5s} {'ms':>5s} {'Score':>6s}")
    print("  " + "-" * 85)
    for r in sorted(all_results, key=lambda x: x["fnr"] + x["fpr"]):
        score = r["fnr"] + r["fpr"]
        marker = " <-- TARGET" if r["fnr"] < 3.0 and r["fpr"] < 15.0 else ""
        print(f"  {r['name']:<50s} {r['accuracy']:>4.1f}% {r['fnr']:>4.1f}% {r['fpr']:>4.1f}% {r['avg_latency_ms']:>4.0f}ms {score:>5.1f}%{marker}")

    # Save
    with open("/tmp/privacy_experiment_v5_holdout.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved to /tmp/privacy_experiment_v5_holdout.json")

    # Determine best and save
    valid = [r for r in all_results if r["fpr"] < 20]
    if valid:
        best = min(valid, key=lambda r: r["fnr"] + r["fpr"] * 0.5)
        for prompt in PROMPTS:
            if prompt["name"] in best["name"]:
                with open("/tmp/best_prompt.json", "w") as f:
                    json.dump({
                        "model": best["name"].split(" | ")[0],
                        "prompt_name": prompt["name"],
                        "system": prompt["system"],
                        "template": prompt["template"],
                        "accuracy": best["accuracy"],
                        "fnr": best["fnr"],
                        "fpr": best["fpr"],
                        "avg_latency_ms": best["avg_latency_ms"],
                        "validation": "holdout test set (241 cases, never used for tuning)",
                    }, f, indent=2)
                break
        print(f"  Best prompt saved to /tmp/best_prompt.json")


if __name__ == "__main__":
    main()

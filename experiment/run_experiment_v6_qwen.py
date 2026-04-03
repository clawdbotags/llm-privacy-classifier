#!/usr/bin/env python3
"""
Privacy Classifier v6 — Qwen3 properly tested.

Fix: Qwen3 models need the /api/chat endpoint (not /api/generate) with
num_predict=1000 to allow thinking tokens. The chat API separates the
thinking into a 'thinking' field so we get clean classification output.

Tests: qwen3:1.7b, qwen3:4b alongside gemma3:1b, llama3.2:3b
Using Prompt G (v4 winner) on the full 481-case dataset.
"""

import re
import json
import time
import requests
import sys
import random

OLLAMA_URL = "http://localhost:11434"

import os
sys.path.insert(0, "/tmp")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from large_test_dataset import TEST_CASES as BASE_CASES
except ImportError:
    from privacy_test_dataset import TEST_CASES as BASE_CASES
from german_test_cases import GERMAN_CASES
ALL_CASES = list(BASE_CASES) + GERMAN_CASES

# Holdout split (same seed as v5)
random.seed(42)
indices = list(range(len(ALL_CASES)))
random.shuffle(indices)
split = len(indices) // 2
DEV_SET = [ALL_CASES[i] for i in indices[:split]]
TEST_SET = [ALL_CASES[i] for i in indices[split:]]

MODELS = ["gemma3:1b", "qwen3:1.7b", "llama3.2:3b", "qwen3:4b"]
QWEN_MODELS = {"qwen3:1.7b", "qwen3:4b"}

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
# PROMPT G (v4 winner)
# ============================================================

PROMPT_G = {
    "name": "G: topic-vs-person",
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


# ============================================================
# CLASSIFIERS — different API for Qwen vs others
# ============================================================

def classify_llm_generate(text, model, prompt_config):
    """Standard /api/generate for non-thinking models."""
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
        raw = resp.json().get("response", "").strip().upper()
        latency = (time.time() - start) * 1000
        if "SENSITIVE" in raw:
            return "SENSITIVE", latency
        elif "PUBLIC" in raw:
            return "PUBLIC", latency
        return "UNCERTAIN", latency
    except Exception as e:
        return f"ERROR:{e}", (time.time() - start) * 1000


def classify_llm_chat(text, model, prompt_config):
    """Chat API for Qwen3 thinking models — separates thinking from response."""
    user_text = prompt_config["template"].format(text=text)
    start = time.time()
    try:
        resp = requests.post(f"{OLLAMA_URL}/api/chat", json={
            "model": model,
            "messages": [
                {"role": "system", "content": prompt_config["system"]},
                {"role": "user", "content": user_text},
            ],
            "stream": False,
            "options": {"temperature": 0, "num_predict": 1000}
        }, timeout=120)
        data = resp.json()
        msg = data.get("message", {})
        raw = msg.get("content", "").strip().upper()
        latency = (time.time() - start) * 1000

        if "SENSITIVE" in raw:
            return "SENSITIVE", latency
        elif "PUBLIC" in raw:
            return "PUBLIC", latency
        return "UNCERTAIN", latency
    except Exception as e:
        return f"ERROR:{e}", (time.time() - start) * 1000


def classify_llm(text, model, prompt_config):
    """Route to correct API based on model."""
    if model in QWEN_MODELS:
        return classify_llm_chat(text, model, prompt_config)
    return classify_llm_generate(text, model, prompt_config)


def classify_hybrid(text, model, prompt_config):
    regex_result, matches = classify_regex(text)
    if regex_result == "SENSITIVE":
        return "SENSITIVE", f"regex:{','.join(matches)}", 0.0
    llm_result, latency = classify_llm(text, model, prompt_config)
    if llm_result in ("UNCERTAIN",) or llm_result.startswith("ERROR"):
        return "SENSITIVE", "failsafe", latency
    return llm_result, f"llm:{llm_result.lower()}", latency


# ============================================================
# RUNNER
# ============================================================

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
            tag = "LEAK" if is_leak else "FP  "
            print(f"  [{tag}] {expected:>9s}->{predicted:<9s} | {method:<22s} | {text[:65]}")

    r["accuracy"] = round(r["correct"] / r["total"] * 100, 1)
    r["fnr"] = round(r["false_negatives"] / n_s * 100, 1) if n_s else 0
    r["fpr"] = round(r["false_positives"] / n_p * 100, 1) if n_p else 0
    r["avg_latency_ms"] = round(r["total_latency_ms"] / r["llm_calls"], 1) if r["llm_calls"] else 0
    return r


def check_model(model):
    try:
        if model in QWEN_MODELS:
            resp = requests.post(f"{OLLAMA_URL}/api/chat", json={
                "model": model,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
                "options": {"num_predict": 5}
            }, timeout=120)
        else:
            resp = requests.post(f"{OLLAMA_URL}/api/generate", json={
                "model": model, "prompt": "hi", "stream": False,
                "options": {"num_predict": 1}
            }, timeout=120)
        return resp.status_code == 200
    except:
        return False


def main():
    # Run on BOTH full dataset and holdout
    datasets = {
        "full (481)": ALL_CASES,
        "holdout (241)": TEST_SET,
    }

    for ds_name, test_cases in datasets.items():
        n_s = sum(1 for _, e in test_cases if e == "SENSITIVE")
        n_p = sum(1 for _, e in test_cases if e == "PUBLIC")
        print("\n" + "=" * 70)
        print(f"  DATASET: {ds_name} — {len(test_cases)} cases ({n_s}S, {n_p}P)")
        print("=" * 70)

        results = []

        for model in MODELS:
            print(f"\n>>> {model}" + (" [chat API, num_predict=1000]" if model in QWEN_MODELS else " [generate API]"))
            if not check_model(model):
                print(f"  SKIPPED — not available")
                continue
            print(f"  Warming up...")
            classify_llm("warmup test", model, PROMPT_G)

            name = f"{model} | G"
            print(f"\n--- {name}")
            r = run_experiment(name,
                lambda t, m=model: classify_hybrid(t, m, PROMPT_G),
                test_cases)
            print(f"\n  {'='*60}")
            print(f"  {r['name']}")
            print(f"  Acc: {r['accuracy']}%  FNR: {r['fnr']}% ({r['false_negatives']} leaks)  FPR: {r['fpr']}% ({r['false_positives']} waste)  Lat: {r['avg_latency_ms']}ms")
            results.append(r)

        # Comparison table
        print(f"\n\n{'='*85}")
        print(f"  COMPARISON — {ds_name}")
        print(f"{'='*85}")
        print(f"  {'Model':<25s} {'Acc':>5s} {'FNR':>5s} {'FPR':>5s} {'Latency':>9s} {'API':>12s}")
        print(f"  {'-'*80}")
        for r in sorted(results, key=lambda x: x["fnr"] + x["fpr"]):
            model_name = r["name"].split(" | ")[0]
            api = "chat/think" if model_name in QWEN_MODELS else "generate"
            print(f"  {model_name:<25s} {r['accuracy']:>4.1f}% {r['fnr']:>4.1f}% {r['fpr']:>4.1f}% {r['avg_latency_ms']:>7.0f}ms {api:>12s}")

    # Save
    with open("/tmp/privacy_experiment_v6_qwen.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to /tmp/privacy_experiment_v6_qwen.json")


if __name__ == "__main__":
    main()

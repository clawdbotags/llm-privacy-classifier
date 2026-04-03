#!/usr/bin/env python3
"""
Privacy Classifier Experiment v3 — Karpathy autoresearch iteration 3.

Fixes from v2:
- Prompt C extended with German examples + institutional message examples
- New Prompt D: prompt C + 8 examples covering all hard categories
- Only testing models that showed promise: gemma3:1b, llama3.2:3b
- Skip 27B (too slow for iteration loop)
"""

import re
import json
import time
import requests
import sys
import os

OLLAMA_URL = "http://localhost:11434"

sys.path.insert(0, "/tmp")
try:
    from large_test_dataset import TEST_CASES as BASE_CASES
except ImportError:
    from privacy_test_dataset import TEST_CASES as BASE_CASES

from german_test_cases import GERMAN_CASES
TEST_CASES = list(BASE_CASES) + GERMAN_CASES

MODELS = ["gemma3:1b", "llama3.2:3b"]

# ============================================================
# MINIMAL REGEX — structural PII only, skip code context
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
}

def classify_regex(text):
    if CODE_MARKERS.search(text):
        return "UNCERTAIN", []
    matches = [n for n, p in STRUCTURAL_PII.items() if re.search(p, text)]
    return ("SENSITIVE", matches) if matches else ("UNCERTAIN", [])


# ============================================================
# PROMPT STRATEGIES — iteration 3
# ============================================================

# Baseline: best from v2
PROMPT_C = {
    "name": "C: few-shot+defs (v2 best)",
    "system": """You are a privacy classifier. Respond with SENSITIVE or PUBLIC only.

SENSITIVE: Message contains or reveals personal/private information about a specific real person — health details, financial specifics, legal situations, personal relationships, confessions, contact info, ID numbers, or personal experiences with sensitive topics. Any language.

PUBLIC: General knowledge, technical/coding questions, educational content, or discussions about topics without identifying a specific person's private details. Questions ABOUT sensitive topics without personal context are PUBLIC. Code examples with fake/test data are PUBLIC.""",
    "template": '''Examples:
Message: "My credit score dropped to 620 after missing payments" -> SENSITIVE
Message: "How do I implement a binary search tree in Python?" -> PUBLIC
Message: "Mein Arzt hat bei mir Diabetes diagnostiziert" -> SENSITIVE
Message: "What are the symptoms of type 2 diabetes?" -> PUBLIC

Message: "{text}"
Classification:''',
}

# Iteration 3: extended few-shot targeting the two hard categories
PROMPT_D = {
    "name": "D: 8-shot multilingual+institutional",
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

# Iteration 3 variant: more aggressive — lean SENSITIVE on ambiguity
PROMPT_E = {
    "name": "E: 8-shot + lean-sensitive",
    "system": """You are a privacy classifier protecting user data. Respond with SENSITIVE or PUBLIC only.

SENSITIVE: Contains or reveals personal/private information about any real person — health, financial, legal, relationships, personal experiences, contact info, ID numbers, or third-party messages referencing a named person's private data. When in doubt, choose SENSITIVE. Any language.

PUBLIC: Clearly general knowledge, coding questions, generic tutorials, or educational content with no specific person's private data involved.""",
    "template": '''Examples:
Message: "My credit score dropped to 620 after missing mortgage payments" -> SENSITIVE
Message: "Customer Joe_Schuster53, we need to verify your IMEI for recent banking updates" -> SENSITIVE
Message: "Mein Gehalt beträgt 65.000 Euro im Jahr und ich möchte verhandeln" -> SENSITIVE
Message: "Erreichen Sie mich unter 0176 12345678 nach 17 Uhr" -> SENSITIVE
Message: "Ich wurde wegen Trunkenheit am Steuer angehalten und brauche einen Anwalt" -> SENSITIVE
Message: "How do I implement a binary search tree in Python?" -> PUBLIC
Message: "Was sind die Symptome von Typ-2-Diabetes?" -> PUBLIC
Message: "What are the penalties for a first-time DUI offense?" -> PUBLIC

Message: "{text}"
Classification:''',
}

PROMPTS = [PROMPT_C, PROMPT_D, PROMPT_E]


# ============================================================
# CLASSIFIER
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


def run_experiment(name, classify_fn):
    n_s = sum(1 for _, e in TEST_CASES if e == "SENSITIVE")
    n_p = sum(1 for _, e in TEST_CASES if e == "PUBLIC")
    r = {
        "name": name, "total": len(TEST_CASES),
        "n_sensitive": n_s, "n_public": n_p,
        "correct": 0, "false_negatives": 0, "false_positives": 0,
        "total_latency_ms": 0, "llm_calls": 0,
        "leaks": [], "fps": [],
    }

    for text, expected in TEST_CASES:
        predicted, method, latency = classify_fn(text)
        r["total_latency_ms"] += latency
        if latency > 0:
            r["llm_calls"] += 1

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
            tag = "LEAK" if (expected == "SENSITIVE" and predicted == "PUBLIC") else "FP  "
            print(f"  [{tag}] {expected:>9s}->{predicted:<9s} | {method:<22s} | {text[:65]}")

    r["accuracy"] = round(r["correct"] / r["total"] * 100, 1)
    r["fnr"] = round(r["false_negatives"] / n_s * 100, 1) if n_s else 0
    r["fpr"] = round(r["false_positives"] / n_p * 100, 1) if n_p else 0
    r["avg_latency_ms"] = round(r["total_latency_ms"] / r["llm_calls"], 1) if r["llm_calls"] else 0
    return r


def print_summary(r):
    print(f"\n  {'='*60}")
    print(f"  {r['name']}")
    print(f"  Accuracy: {r['accuracy']}%  FNR: {r['fnr']}% ({r['false_negatives']} leaks)  FPR: {r['fpr']}% ({r['false_positives']} waste)  Latency: {r['avg_latency_ms']}ms")


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
    n_s = sum(1 for _, e in TEST_CASES if e == "SENSITIVE")
    n_p = sum(1 for _, e in TEST_CASES if e == "PUBLIC")
    print("=" * 65)
    print("  PRIVACY CLASSIFIER EXPERIMENT v3 — iteration 3")
    print(f"  Dataset: {len(TEST_CASES)} cases ({n_s} sensitive, {n_p} public)")
    print(f"  Focus: German + institutional messages")
    print("=" * 65)

    all_results = []

    for model in MODELS:
        print(f"\n>>> {model}")
        if not check_model(model):
            print(f"  SKIPPED")
            continue
        print(f"  Warming up...")
        classify_llm("warmup", model, PROMPT_C)

        for prompt in PROMPTS:
            name = f"{model} | {prompt['name']}"
            print(f"\n--- {name}")
            r = run_experiment(name, lambda t, m=model, p=prompt: classify_hybrid(t, m, p))
            print_summary(r)
            all_results.append(r)

    # Final table
    print("\n\n" + "=" * 85)
    print("  FINAL — sorted by FNR (data leaks, lower is better)")
    print("=" * 85)
    print(f"  {'Experiment':<48s} {'Acc':>5s} {'FNR':>5s} {'FPR':>5s} {'ms':>5s}")
    print("  " + "-" * 75)
    for r in sorted(all_results, key=lambda x: x["fnr"]):
        print(f"  {r['name']:<48s} {r['accuracy']:>4.1f}% {r['fnr']:>4.1f}% {r['fpr']:>4.1f}% {r['avg_latency_ms']:>4.0f}ms")

    with open("/tmp/privacy_experiment_v3_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved to /tmp/privacy_experiment_v3_results.json")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Privacy Classifier Experiment v2 — Karpathy autoresearch iteration 2.

Changes from v1:
- Larger dataset with real-world data + German language cases
- 3 prompt strategies (minimal, few-shot, few-shot+definitions)
- No CoT (confirmed hurts small models)
- Stripped regex down to basics — let the LLM do the nuanced work
- Tests: gemma3:1b, llama3.2:3b, gemma3:27b (as accuracy ceiling reference)
"""

import re
import json
import time
import requests
import sys
import os

OLLAMA_URL = "http://localhost:11434"

# Import datasets
sys.path.insert(0, "/tmp")
try:
    from large_test_dataset import TEST_CASES
    DATASET_NAME = "large (real-world)"
except ImportError:
    from privacy_test_dataset import TEST_CASES
    DATASET_NAME = "hand-crafted"

# Add German cases
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from german_test_cases import GERMAN_CASES
    TEST_CASES = list(TEST_CASES) + GERMAN_CASES
    DATASET_NAME += " + German"
except ImportError:
    pass

# ============================================================
# MINIMAL REGEX — only catch obvious structured PII
# ============================================================

STRUCTURAL_PII = {
    "ssn": r'\b\d{3}[-]\d{2}[-]\d{4}\b',
    "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b',
    "credit_card": r'\b(?:4\d{3}|5[1-5]\d{2}|3[47]\d{2})[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
    "iban": r'\b[A-Z]{2}\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{0,2}\b',
    "phone_intl": r'\+\d{1,3}\s?\d{2,4}\s?\d{4,8}',
}

# Code context — if present, skip regex match (it's probably example data)
CODE_MARKERS = re.compile(
    r"(?:regex|pattern|validate|test|fixture|faker|mock|example|schema|CSV|"
    r"template|write a|implement|library|generate|function|def |class |import |"
    r"r'|r\"|`|\{.*:.*\}|how do I (?:write|build|implement|create|design))",
    re.IGNORECASE
)


def classify_regex(text):
    """Minimal regex — only structural PII, skip code context."""
    if CODE_MARKERS.search(text):
        return "UNCERTAIN", []

    matches = []
    for name, pattern in STRUCTURAL_PII.items():
        if re.search(pattern, text):
            matches.append(name)

    if matches:
        return "SENSITIVE", matches
    return "UNCERTAIN", []


# ============================================================
# PROMPT STRATEGIES
# ============================================================

PROMPT_A = {
    "name": "A: minimal zero-shot",
    "system": "Classify as SENSITIVE or PUBLIC. Respond with one word only.",
    "template": 'Message: "{text}"\n\nClassification:',
}

PROMPT_B = {
    "name": "B: few-shot (4 examples)",
    "system": "You are a privacy classifier. Respond with SENSITIVE or PUBLIC only.",
    "template": '''Examples:
Message: "My credit score dropped to 620 after missing payments" -> SENSITIVE
Message: "How do I implement a binary search tree in Python?" -> PUBLIC
Message: "I was diagnosed with diabetes and my doctor prescribed metformin" -> SENSITIVE
Message: "What are the symptoms of type 2 diabetes?" -> PUBLIC

Message: "{text}"
Classification:''',
}

PROMPT_C = {
    "name": "C: few-shot + definitions",
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

PROMPTS = [PROMPT_A, PROMPT_B, PROMPT_C]

# Models to test
MODELS = [
    "gemma3:1b",
    "llama3.2:3b",
    "gemma3:27b",  # accuracy ceiling reference — comment out if slow
]


def classify_llm(text, model, prompt_config):
    """Classify using LLM via Ollama."""
    prompt_text = prompt_config["template"].format(text=text)

    start = time.time()
    try:
        payload = {
            "model": model,
            "prompt": prompt_text,
            "system": prompt_config["system"],
            "stream": False,
            "options": {"temperature": 0, "num_predict": 10}
        }
        resp = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=120)
        raw = resp.json().get("response", "").strip()
        latency = (time.time() - start) * 1000

        # Strip thinking tags (for Qwen etc)
        clean = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip().upper()

        if "SENSITIVE" in clean:
            return "SENSITIVE", latency
        elif "PUBLIC" in clean:
            return "PUBLIC", latency
        else:
            return "UNCERTAIN", latency
    except Exception as e:
        return f"ERROR:{e}", (time.time() - start) * 1000


def classify_hybrid(text, model, prompt_config):
    """Hybrid: minimal regex first, LLM for uncertain."""
    regex_result, matches = classify_regex(text)

    if regex_result == "SENSITIVE":
        return "SENSITIVE", f"regex:{','.join(matches)}", 0.0

    llm_result, latency = classify_llm(text, model, prompt_config)

    if llm_result == "UNCERTAIN" or llm_result.startswith("ERROR"):
        return "SENSITIVE", "failsafe", latency

    return llm_result, f"llm:{llm_result.lower()}", latency


def run_experiment(name, classify_fn, test_cases):
    """Run experiment, return metrics."""
    n_sensitive = sum(1 for _, e in test_cases if e == "SENSITIVE")
    n_public = sum(1 for _, e in test_cases if e == "PUBLIC")

    results = {
        "name": name,
        "total": len(test_cases),
        "n_sensitive": n_sensitive,
        "n_public": n_public,
        "correct": 0,
        "false_negatives": 0,
        "false_positives": 0,
        "total_latency_ms": 0,
        "llm_calls": 0,
        "leaks": [],
        "fps": [],
    }

    for i, (text, expected) in enumerate(test_cases):
        predicted, method, latency = classify_fn(text)
        results["total_latency_ms"] += latency
        if latency > 0:
            results["llm_calls"] += 1

        correct = predicted == expected
        if correct:
            results["correct"] += 1
        else:
            is_leak = expected == "SENSITIVE" and predicted == "PUBLIC"
            if is_leak:
                results["false_negatives"] += 1
                results["leaks"].append({"text": text[:120], "method": method})
            else:
                results["false_positives"] += 1
                results["fps"].append({"text": text[:120], "method": method})

        tag = "OK  " if correct else ("LEAK" if (expected == "SENSITIVE" and predicted == "PUBLIC") else "FP  ")
        # Only print misses to keep output manageable
        if not correct:
            print(f"  [{tag}] {expected:>9s}->{predicted:<9s} | {method:<25s} | {text[:65]}")

    results["accuracy"] = round(results["correct"] / results["total"] * 100, 1)
    results["fnr"] = round(results["false_negatives"] / n_sensitive * 100, 1) if n_sensitive else 0
    results["fpr"] = round(results["false_positives"] / n_public * 100, 1) if n_public else 0
    results["avg_latency_ms"] = round(results["total_latency_ms"] / results["llm_calls"], 1) if results["llm_calls"] else 0
    results["total_time_s"] = round(results["total_latency_ms"] / 1000, 1)

    return results


def print_summary(r):
    print(f"\n  {'='*65}")
    print(f"  {r['name']}")
    print(f"  {'='*65}")
    print(f"  Accuracy:            {r['accuracy']}% ({r['correct']}/{r['total']})")
    print(f"  False Negative Rate: {r['fnr']}%  ({r['false_negatives']} SENSITIVE leaked)")
    print(f"  False Positive Rate: {r['fpr']}%  ({r['false_positives']} PUBLIC routed local)")
    print(f"  LLM calls:           {r['llm_calls']} / {r['total']}")
    print(f"  Avg LLM latency:     {r['avg_latency_ms']}ms")
    print(f"  Total time:          {r['total_time_s']}s")
    if r["leaks"]:
        print(f"\n  DATA LEAKS ({len(r['leaks'])}):")
        for m in r["leaks"]:
            print(f"    [{m['method']}] {m['text']}")
    print()


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
    test_cases = TEST_CASES
    n_s = sum(1 for _, e in test_cases if e == "SENSITIVE")
    n_p = sum(1 for _, e in test_cases if e == "PUBLIC")

    print("=" * 65)
    print("  PRIVACY CLASSIFIER EXPERIMENT v2")
    print(f"  Dataset: {DATASET_NAME}")
    print(f"  Cases: {len(test_cases)} ({n_s} sensitive, {n_p} public)")
    print(f"  Prompt strategies: {len(PROMPTS)}")
    print(f"  Models: {len(MODELS)}")
    print("=" * 65)

    all_results = []

    for model in MODELS:
        print(f"\n>>> Checking model: {model}")
        if not check_model(model):
            print(f"  SKIPPED — {model} not available")
            continue

        # Warm up
        print(f"  Warming up {model}...")
        classify_llm("warmup test", model, PROMPT_A)

        for prompt in PROMPTS:
            exp_name = f"{model} | {prompt['name']}"
            print(f"\n>>> {exp_name}")

            r = run_experiment(
                exp_name,
                lambda t, m=model, p=prompt: classify_hybrid(t, m, p),
                test_cases
            )
            print_summary(r)
            all_results.append(r)

    # Final comparison
    print("\n" + "=" * 95)
    print("  FINAL COMPARISON — sorted by accuracy (FNR is the critical metric)")
    print("=" * 95)
    header = f"  {'Experiment':<45s} {'Acc':>5s} {'FNR':>5s} {'FPR':>5s} {'LLM':>4s} {'AvgMs':>7s} {'TotS':>6s}"
    print(header)
    print("  " + "-" * 90)

    sorted_results = sorted(all_results, key=lambda r: (-r["accuracy"], r["fnr"]))
    for r in sorted_results:
        print(f"  {r['name']:<45s} {r['accuracy']:>4.1f}% {r['fnr']:>4.1f}% {r['fpr']:>4.1f}% {r['llm_calls']:>4d} {r['avg_latency_ms']:>6.0f}ms {r['total_time_s']:>5.1f}s")

    # Save
    out_path = "/tmp/privacy_experiment_v2_results.json"
    save_results = [{k: v for k, v in r.items()} for r in all_results]
    with open(out_path, "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()

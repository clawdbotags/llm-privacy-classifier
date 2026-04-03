#!/usr/bin/env python3
"""
Privacy Classifier Experiment — Karpathy autoresearch style.
Experiment → Evaluate → Keep/Discard → Iterate

Tests regex patterns + small LLM classifiers for SENSITIVE vs PUBLIC classification.
The metric that matters: false negative rate (sensitive leaked as public).
"""

import re
import json
import time
import requests
import sys
from dataclasses import dataclass

OLLAMA_URL = "http://localhost:11434"

# ============================================================
# TEST DATASET — 60 cases, labeled ground truth
# ============================================================

TEST_CASES = [
    # === SENSITIVE: Personal identifiers ===
    ("My SSN is 123-45-6789, can you help me with my tax return?", "SENSITIVE"),
    ("John Smith was born on 03/15/1985 and lives at 42 Oak Street", "SENSITIVE"),
    ("My passport number is AB1234567", "SENSITIVE"),
    ("Please update my driver's license number K12-345-678", "SENSITIVE"),
    ("My employee ID is EMP-2024-0847", "SENSITIVE"),

    # === SENSITIVE: Contact info ===
    ("Send the invoice to john.doe@company.com", "SENSITIVE"),
    ("Call me at (555) 123-4567 to discuss", "SENSITIVE"),
    ("My address is 1234 Maple Drive, Springfield, IL 62704", "SENSITIVE"),
    ("You can reach me at +49 176 12345678", "SENSITIVE"),
    ("Forward it to sarah.miller@gmail.com, she handles billing", "SENSITIVE"),

    # === SENSITIVE: Financial ===
    ("My salary is $85,000 per year and I want to negotiate a raise", "SENSITIVE"),
    ("My credit card number is 4532-1234-5678-9012", "SENSITIVE"),
    ("I have $45,000 in my savings account at Chase", "SENSITIVE"),
    ("My mortgage payment is $2,100/month and I'm struggling", "SENSITIVE"),
    ("My credit score dropped to 620 after the missed payment", "SENSITIVE"),
    ("My bank account is DE89370400440532013000", "SENSITIVE"),
    ("I owe $12,000 in student loans to Sallie Mae", "SENSITIVE"),

    # === SENSITIVE: Health/Medical ===
    ("I was diagnosed with type 2 diabetes last month", "SENSITIVE"),
    ("My doctor prescribed metformin 500mg twice daily", "SENSITIVE"),
    ("My blood pressure reading was 160/95 yesterday", "SENSITIVE"),
    ("I've been seeing a therapist for depression since January", "SENSITIVE"),
    ("My son has ADHD and we're considering medication", "SENSITIVE"),

    # === SENSITIVE: Legal/Private ===
    ("I'm filing for divorce and need to hide some assets", "SENSITIVE"),
    ("My boss Mike harassed me at work last Tuesday", "SENSITIVE"),
    ("I got arrested for DUI last weekend", "SENSITIVE"),
    ("I'm planning to resign from Google next month", "SENSITIVE"),
    ("The custody hearing for my daughter Emma is on Friday", "SENSITIVE"),

    # === SENSITIVE: Named relationships ===
    ("My therapist Dr. Williams recommended I try meditation", "SENSITIVE"),
    ("Tell my wife Anna that I'll be late tonight", "SENSITIVE"),
    ("My accountant Peter filed my taxes wrong", "SENSITIVE"),

    # === SENSITIVE: Edge cases (tricky) ===
    ("I make about six figures but feel underpaid for Seattle", "SENSITIVE"),
    ("After my miscarriage, I took two weeks off work", "SENSITIVE"),
    ("My neighbor's kid keeps stealing from our garden, should I call police?", "SENSITIVE"),
    ("I failed the drug test at my new job", "SENSITIVE"),
    ("I'm worried about my mom's Alzheimer's getting worse", "SENSITIVE"),

    # === PUBLIC: General knowledge ===
    ("What is the capital of France?", "PUBLIC"),
    ("Explain quantum computing in simple terms", "PUBLIC"),
    ("What caused World War II?", "PUBLIC"),
    ("How does photosynthesis work?", "PUBLIC"),
    ("What's the distance from Earth to Mars?", "PUBLIC"),

    # === PUBLIC: Programming/Technical ===
    ("How do I implement a binary search tree in Python?", "PUBLIC"),
    ("What's the difference between TCP and UDP?", "PUBLIC"),
    ("Explain the observer pattern with a code example", "PUBLIC"),
    ("How do I set up a Kubernetes cluster?", "PUBLIC"),
    ("Write a regex to validate email format", "PUBLIC"),
    ("What's the time complexity of quicksort?", "PUBLIC"),
    ("How do I use async/await in Rust?", "PUBLIC"),

    # === PUBLIC: Generic health/science (no personal context) ===
    ("What are the symptoms of type 2 diabetes?", "PUBLIC"),
    ("How does metformin work in the body?", "PUBLIC"),
    ("What is a normal blood pressure range?", "PUBLIC"),
    ("What are common treatments for depression?", "PUBLIC"),

    # === PUBLIC: Edge cases (tricky) ===
    ("Write a function that validates SSN format: XXX-XX-XXXX", "PUBLIC"),
    ("Generate test data with fake names and addresses", "PUBLIC"),
    ("What are the legal requirements for filing divorce in California?", "PUBLIC"),
    ("How much does a software engineer typically make in the US?", "PUBLIC"),
    ("What's the average credit score in America?", "PUBLIC"),
    ("Example: John Doe, 123 Main St — parse this address format", "PUBLIC"),
    ("How do IBAN numbers work in European banking?", "PUBLIC"),
    ("What medications are used to treat ADHD in children?", "PUBLIC"),
    ("Write a mock user profile generator with email and phone fields", "PUBLIC"),
    ("What are my rights if I get arrested?", "PUBLIC"),
]

# ============================================================
# REGEX PATTERNS — production PII detection
# ============================================================

PII_PATTERNS = {
    "ssn": r'\b(?!000|666|9\d{2})\d{3}[-\s]?\d{2}[-\s]?\d{4}\b',
    "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b',
    "phone_us": r'(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
    "phone_intl": r'\+\d{1,3}\s?\d{2,4}\s?\d{4,8}',
    "credit_card": r'\b(?:4\d{3}|5[1-5]\d{2}|3[47]\d{2}|6(?:011|5\d{2}))[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
    "iban": r'\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}(?:[A-Z0-9]?){0,16}\b',
    "ip_address": r'\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b',
    "passport": r'\b[A-Z]{1,2}\d{6,9}\b',
    "address": r'\b\d{1,5}\s+(?:[A-Z][a-z]+\s){1,3}(?:Street|St|Drive|Dr|Avenue|Ave|Road|Rd|Lane|Ln|Boulevard|Blvd|Way|Court|Ct|Place|Pl)\b',
    "dob_mmddyyyy": r'\b(?:0[1-9]|1[0-2])[/-]\d{2}[/-](?:19|20)\d{2}\b',
    "salary_amount": r'(?:salary|income|make|earn|paid|wage|compensation).*?\$[\d,]+|\$[\d,]+.*?(?:salary|income|per year|annually|a year)',
    "my_personal": r'\b(?:my|I\'m|I am|I was|I have|I got|I failed|I owe)\b.*(?:diagnosed|prescribed|therapist|doctor|medication|blood pressure|credit score|mortgage|arrested|divorced|custody|resign|salary|bank account|savings|debt|loan)',
}

# Context patterns that suggest personal (not general) discussion
PERSONAL_CONTEXT_PATTERNS = {
    "possessive_health": r'\b(?:my|his|her|their)\s+(?:doctor|therapist|diagnosis|medication|condition|symptoms|blood pressure|prescription)',
    "possessive_financial": r'\b(?:my|his|her|their)\s+(?:salary|income|bank|mortgage|credit|debt|loan|savings|tax)',
    "possessive_legal": r'\b(?:my|his|her|their)\s+(?:lawyer|attorney|arrest|divorce|custody|hearing|case)',
    "named_person_context": r'\b(?:my\s+(?:boss|wife|husband|partner|friend|neighbor|son|daughter|mom|dad|mother|father|brother|sister|accountant|therapist|doctor))\s+[A-Z][a-z]+',
}


def classify_regex(text: str) -> tuple[str, list[str]]:
    """Classify using regex patterns. Returns (classification, matched_patterns)."""
    matches = []

    for name, pattern in {**PII_PATTERNS, **PERSONAL_CONTEXT_PATTERNS}.items():
        if re.search(pattern, text, re.IGNORECASE):
            matches.append(name)

    if matches:
        return "SENSITIVE", matches
    return "UNCERTAIN", []


def classify_llm(text: str, model: str) -> tuple[str, float]:
    """Classify using LLM via Ollama. Returns (classification, latency_ms)."""
    prompt = f"""Classify this message as SENSITIVE or PUBLIC.

SENSITIVE means it contains or references personal/private information about a specific person:
- Personal identifiers, contact info, financial details, health info, legal matters
- Named personal relationships ("my boss Mike", "my doctor")
- Personal experiences ("I was diagnosed", "my salary", "I got arrested")

PUBLIC means it's a general knowledge question, technical/coding question, or discusses topics without personal context.

Message: "{text}"

Reply with ONLY one word: SENSITIVE or PUBLIC"""

    start = time.time()
    try:
        resp = requests.post(f"{OLLAMA_URL}/api/generate", json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0, "num_predict": 10}
        }, timeout=30)
        result = resp.json().get("response", "").strip().upper()
        latency = (time.time() - start) * 1000

        if "SENSITIVE" in result:
            return "SENSITIVE", latency
        elif "PUBLIC" in result:
            return "PUBLIC", latency
        else:
            return "UNCERTAIN", latency
    except Exception as e:
        return f"ERROR:{e}", 0


def classify_hybrid(text: str, model: str) -> tuple[str, list[str], float]:
    """Hybrid: regex first, LLM for uncertain cases. Returns (classification, regex_matches, llm_latency_ms)."""
    regex_result, matches = classify_regex(text)

    if regex_result == "SENSITIVE":
        return "SENSITIVE", matches, 0.0

    # Regex uncertain → ask LLM
    llm_result, latency = classify_llm(text, model)

    if llm_result == "UNCERTAIN" or llm_result.startswith("ERROR"):
        # Fail-safe: default to SENSITIVE
        return "SENSITIVE", ["failsafe"], latency

    return llm_result, ["llm"], latency


def run_experiment(name: str, classify_fn, test_cases=TEST_CASES):
    """Run a full experiment and return metrics."""
    results = {
        "name": name,
        "total": len(test_cases),
        "correct": 0,
        "wrong": 0,
        "false_negatives": 0,  # SENSITIVE classified as PUBLIC (DATA LEAK!)
        "false_positives": 0,  # PUBLIC classified as SENSITIVE (wastes money)
        "total_latency_ms": 0,
        "llm_calls": 0,
        "errors": [],
        "details": []
    }

    for text, expected in test_cases:
        if "hybrid" in name.lower() or "llm" in name.lower():
            if "hybrid" in name.lower():
                predicted, info, latency = classify_fn(text)
                results["total_latency_ms"] += latency
                if latency > 0:
                    results["llm_calls"] += 1
            else:
                predicted, latency = classify_fn(text)
                results["total_latency_ms"] += latency
                results["llm_calls"] += 1
                info = ["llm-only"]
        else:
            predicted, info = classify_fn(text)
            latency = 0

        correct = predicted == expected
        if correct:
            results["correct"] += 1
        else:
            results["wrong"] += 1
            if expected == "SENSITIVE" and predicted == "PUBLIC":
                results["false_negatives"] += 1
            elif expected == "PUBLIC" and predicted == "SENSITIVE":
                results["false_positives"] += 1

        detail = {
            "text": text[:80],
            "expected": expected,
            "predicted": predicted,
            "correct": correct,
            "info": info if isinstance(info, list) else [info],
            "latency_ms": round(latency, 1)
        }
        results["details"].append(detail)

        status = "OK" if correct else "MISS"
        if not correct and expected == "SENSITIVE":
            status = "LEAK"
        print(f"  [{status}] {expected:9s} → {predicted:9s} | {text[:70]}")

    results["accuracy"] = round(results["correct"] / results["total"] * 100, 1)
    results["fnr"] = round(results["false_negatives"] / sum(1 for _, e in test_cases if e == "SENSITIVE") * 100, 1)
    results["fpr"] = round(results["false_positives"] / sum(1 for _, e in test_cases if e == "PUBLIC") * 100, 1)
    if results["llm_calls"] > 0:
        results["avg_llm_latency_ms"] = round(results["total_latency_ms"] / results["llm_calls"], 1)
    else:
        results["avg_llm_latency_ms"] = 0

    return results


def print_summary(results: dict):
    """Print experiment summary."""
    print(f"\n{'='*60}")
    print(f"  {results['name']}")
    print(f"{'='*60}")
    print(f"  Accuracy:           {results['accuracy']}% ({results['correct']}/{results['total']})")
    print(f"  False Negative Rate: {results['fnr']}% (SENSITIVE leaked as PUBLIC)")
    print(f"  False Positive Rate: {results['fpr']}% (PUBLIC routed as SENSITIVE)")
    print(f"  LLM calls:          {results['llm_calls']}")
    print(f"  Avg LLM latency:    {results['avg_llm_latency_ms']}ms")
    print()

    if results['wrong'] > 0:
        print("  MISCLASSIFIED:")
        for d in results['details']:
            if not d['correct']:
                leak = " *** DATA LEAK ***" if d['expected'] == 'SENSITIVE' else ""
                print(f"    [{d['expected']} → {d['predicted']}] {d['text']}{leak}")
        print()


def main():
    print("=" * 60)
    print("  PRIVACY CLASSIFIER EXPERIMENT")
    print("  Karpathy autoresearch: experiment → evaluate → iterate")
    print("=" * 60)

    all_results = []

    # --- Experiment 1: Regex only ---
    print("\n>>> Experiment 1: REGEX ONLY")
    r1 = run_experiment("Regex Only", lambda t: classify_regex(t))
    print_summary(r1)
    all_results.append(r1)

    # --- Experiment 2: Gemma 1B only ---
    print("\n>>> Experiment 2: GEMMA3 1B (LLM only)")
    r2 = run_experiment("Gemma3 1B (LLM only)", lambda t: classify_llm(t, "gemma3:1b"))
    print_summary(r2)
    all_results.append(r2)

    # --- Experiment 3: Hybrid regex + Gemma 1B ---
    print("\n>>> Experiment 3: HYBRID regex + Gemma3 1B")
    r3 = run_experiment("Hybrid: regex + Gemma3 1B", lambda t: classify_hybrid(t, "gemma3:1b"))
    print_summary(r3)
    all_results.append(r3)

    # --- Comparison ---
    print("\n" + "=" * 60)
    print("  FINAL COMPARISON")
    print("=" * 60)
    print(f"  {'Experiment':<30s} {'Acc':>6s} {'FNR':>6s} {'FPR':>6s} {'LLM':>5s} {'Lat':>8s}")
    print(f"  {'-'*28:<30s} {'---':>6s} {'---':>6s} {'---':>6s} {'---':>5s} {'---':>8s}")
    for r in all_results:
        print(f"  {r['name']:<30s} {r['accuracy']:>5.1f}% {r['fnr']:>5.1f}% {r['fpr']:>5.1f}% {r['llm_calls']:>5d} {r['avg_llm_latency_ms']:>6.0f}ms")

    # Save results
    with open("/tmp/privacy_experiment_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\n  Results saved to /tmp/privacy_experiment_results.json")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Privacy Classifier Experiment — Karpathy autoresearch style.
Experiment -> Evaluate -> Keep/Discard -> Iterate

Tests regex patterns + small LLM classifiers for SENSITIVE vs PUBLIC classification.
The metric that matters: false negative rate (sensitive leaked as public = DATA LEAK).
"""

import re
import json
import time
import requests
import sys

from privacy_test_dataset import TEST_CASES

OLLAMA_URL = "http://localhost:11434"

# Models to test (will skip if not available)
MODELS_TO_TEST = [
    "gemma3:1b",
    "qwen3:1.7b",
    "llama3.2:3b",
    "qwen3:4b",
]

# ============================================================
# REGEX PATTERNS
# ============================================================

PII_PATTERNS = {
    "ssn": r'\b(?!000|666|9\d{2})\d{3}[-\s]?\d{2}[-\s]?\d{4}\b',
    "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b',
    "phone_us": r'(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
    "phone_intl": r'\+\d{1,3}\s?\d{2,4}\s?\d{4,8}',
    "credit_card": r'\b(?:4\d{3}|5[1-5]\d{2}|3[47]\d{2}|6(?:011|5\d{2}))[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
    "iban": r'\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}(?:[A-Z0-9]?){0,16}\b',
    "passport": r'\b[A-Z]{1,2}\d{6,9}\b',
    "address_street": r'\b\d{1,5}\s+(?:[A-Z][a-z]+\s){1,3}(?:Street|St|Drive|Dr|Avenue|Ave|Road|Rd|Lane|Ln|Boulevard|Blvd|Way|Court|Ct|Place|Pl|Terrace)\b',
    "dob_mmddyyyy": r'\b(?:0[1-9]|1[0-2])[/-]\d{2}[/-](?:19|20)\d{2}\b',
    "dl_number": r'\b(?:DL|D\.L\.)[-\s]?\w{2,4}[-\s]?\d{3,7}[-\s]?\d{2,7}\b',
    "routing_number": r'\b(?:routing|ABA)\s*(?:number|#|no\.?)?\s*\d{9}\b',
    "account_number": r'\b(?:account|acct)\s*(?:number|#|no\.?)?\s*\d{6,17}\b',
}

PERSONAL_CONTEXT_PATTERNS = {
    "possessive_health": r'\b(?:my|his|her|their|I\'m|I am)\b.*\b(?:doctor|therapist|diagnosis|diagnosed|medication|prescribed|blood pressure|prescription|surgery|checkup|exam|colonoscopy|tumor|miscarriage|psych ward|rehab|sober|relapsed|addicted|bulimia)',
    "possessive_financial": r'\b(?:my|I|I\'m|our|we)\b.*\b(?:salary|income|bank account|mortgage|credit score|credit card|401k|savings|debt|loan|owe|inheritance|earn|make \$|garnish)',
    "possessive_legal": r'\b(?:my|I|I\'m|I was|I\'ve)\b.*\b(?:arrested|fired|resign|divorce|custody|sued|lawsuit|restraining order|probation|evict|whistleblow|court date|deport|asylum)',
    "named_person_context": r'\b(?:my\s+(?:boss|wife|husband|partner|friend|neighbor|son|daughter|mom|dad|mother|father|brother|sister|accountant|therapist|doctor|coworker|manager|supervisor|ex|ex-wife|ex-husband|ex-girlfriend|ex-boyfriend|spouse|landlord|teenager))\s+[A-Z][a-z]+',
    "personal_confession": r'\b(?:I\'m|I am|I was|I have|I\'ve been|I feel|I think I\'m|I cheated|I lied|I failed|I tested positive|I\'m pregnant|I\'m addicted|I\'m undocumented|I\'m afraid|I\'m scared|I\'m ashamed)',
    "specific_dollar_personal": r'\b(?:I|my|we|our)\b.*\$[\d,]+',
}

# Anti-patterns: things that LOOK like PII but aren't (code, examples, generic questions)
CODE_CONTEXT_PATTERNS = [
    r'(?:regex|pattern|match|validate|function|test|fixture|faker|mock|example|schema|CSV|columns|template|write a|design a|implement|build a|how do I|library|generate fake)',
    r"(?:r'|r\"|`|\{.*:.*\}|def |class |import |from )",
    r'(?:what (?:is|are) the|how (?:do|does|should)|explain|what happens|what are typical|what is the (?:average|median|process|statute))',
]


def classify_regex(text):
    """Classify using regex patterns. Returns (classification, matched_patterns)."""
    matches = []

    # Check for code/example context first — these are likely PUBLIC
    is_code_context = any(re.search(p, text, re.IGNORECASE) for p in CODE_CONTEXT_PATTERNS)

    # Check structured PII patterns
    for name, pattern in PII_PATTERNS.items():
        if re.search(pattern, text, re.IGNORECASE):
            matches.append(name)

    # If we found structured PII but it looks like code/example context, mark uncertain
    if matches and is_code_context:
        return "UNCERTAIN", matches + ["code_context"]

    if matches:
        return "SENSITIVE", matches

    # Check personal context patterns (these don't have code anti-patterns)
    for name, pattern in PERSONAL_CONTEXT_PATTERNS.items():
        if re.search(pattern, text, re.IGNORECASE):
            matches.append(name)

    if matches:
        # Personal context in code examples is still uncertain
        if is_code_context:
            return "UNCERTAIN", matches + ["code_context"]
        return "SENSITIVE", matches

    return "UNCERTAIN", []


def classify_llm(text, model):
    """Classify using LLM via Ollama. Returns (classification, latency_ms)."""
    prompt = f"""You are a privacy classifier. Classify this message as SENSITIVE or PUBLIC.

SENSITIVE: Contains or reveals personal/private information about a specific real person.
Examples: personal health details, financial specifics, legal situations, named relationships, confessions, contact info, ID numbers.

PUBLIC: General knowledge, technical/coding questions, educational content, or discussions about topics without identifying a specific person's private details.
Key: Questions ABOUT sensitive topics (e.g., "What are divorce laws?") are PUBLIC. Personal experiences WITH those topics (e.g., "I'm getting divorced") are SENSITIVE.
Key: Code examples with fake PII data are PUBLIC.

Message: "{text}"

Answer with ONLY one word: SENSITIVE or PUBLIC"""

    start = time.time()
    try:
        resp = requests.post(f"{OLLAMA_URL}/api/generate", json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0, "num_predict": 10}
        }, timeout=60)
        raw = resp.json().get("response", "").strip()
        latency = (time.time() - start) * 1000

        # Parse — handle thinking models that wrap in <think> tags
        clean = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip().upper()

        if "SENSITIVE" in clean:
            return "SENSITIVE", latency
        elif "PUBLIC" in clean:
            return "PUBLIC", latency
        else:
            return "UNCERTAIN", latency
    except Exception as e:
        return f"ERROR:{e}", (time.time() - start) * 1000


def classify_hybrid(text, model):
    """Hybrid: regex first, LLM for uncertain. Returns (classification, method, latency_ms)."""
    regex_result, matches = classify_regex(text)

    if regex_result == "SENSITIVE":
        return "SENSITIVE", f"regex:{','.join(matches)}", 0.0

    # Regex uncertain or no match -> ask LLM
    llm_result, latency = classify_llm(text, model)

    if llm_result == "UNCERTAIN" or llm_result.startswith("ERROR"):
        return "SENSITIVE", "failsafe", latency  # fail-safe

    return llm_result, f"llm:{llm_result.lower()}", latency


def run_experiment(name, classify_fn):
    """Run experiment, return metrics."""
    n_sensitive = sum(1 for _, e in TEST_CASES if e == "SENSITIVE")
    n_public = sum(1 for _, e in TEST_CASES if e == "PUBLIC")

    results = {
        "name": name,
        "total": len(TEST_CASES),
        "n_sensitive": n_sensitive,
        "n_public": n_public,
        "correct": 0,
        "false_negatives": 0,
        "false_positives": 0,
        "total_latency_ms": 0,
        "llm_calls": 0,
        "misses": [],
    }

    for i, (text, expected) in enumerate(TEST_CASES):
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
            else:
                results["false_positives"] += 1
            results["misses"].append({
                "idx": i,
                "text": text[:100],
                "expected": expected,
                "predicted": predicted,
                "method": method,
                "leak": is_leak,
            })

        tag = "OK  " if correct else ("LEAK" if (expected == "SENSITIVE" and predicted == "PUBLIC") else "FP  ")
        print(f"  [{tag}] {expected:>9s}->{predicted:<9s} | {method:<30s} | {text[:60]}")

    results["accuracy"] = round(results["correct"] / results["total"] * 100, 1)
    results["fnr"] = round(results["false_negatives"] / n_sensitive * 100, 1) if n_sensitive else 0
    results["fpr"] = round(results["false_positives"] / n_public * 100, 1) if n_public else 0
    results["avg_latency_ms"] = round(results["total_latency_ms"] / results["llm_calls"], 1) if results["llm_calls"] else 0
    results["total_time_s"] = round(results["total_latency_ms"] / 1000, 1)

    return results


def print_summary(r):
    print(f"\n{'='*70}")
    print(f"  {r['name']}")
    print(f"{'='*70}")
    print(f"  Accuracy:            {r['accuracy']}% ({r['correct']}/{r['total']})")
    print(f"  False Negative Rate: {r['fnr']}%  ({r['false_negatives']} SENSITIVE leaked as PUBLIC)")
    print(f"  False Positive Rate: {r['fpr']}%  ({r['false_positives']} PUBLIC routed local)")
    print(f"  LLM calls:           {r['llm_calls']} / {r['total']}")
    print(f"  Avg LLM latency:     {r['avg_latency_ms']}ms")
    print(f"  Total LLM time:      {r['total_time_s']}s")

    if r["misses"]:
        leaks = [m for m in r["misses"] if m["leak"]]
        fps = [m for m in r["misses"] if not m["leak"]]
        if leaks:
            print(f"\n  DATA LEAKS ({len(leaks)}):")
            for m in leaks:
                print(f"    [{m['method']}] {m['text']}")
        if fps:
            print(f"\n  FALSE POSITIVES ({len(fps)}):")
            for m in fps:
                print(f"    [{m['method']}] {m['text']}")
    print()


def check_model_available(model):
    try:
        resp = requests.post(f"{OLLAMA_URL}/api/generate", json={
            "model": model, "prompt": "hi", "stream": False,
            "options": {"num_predict": 1}
        }, timeout=120)
        return resp.status_code == 200
    except:
        return False


def main():
    print("=" * 70)
    print("  PRIVACY CLASSIFIER EXPERIMENT")
    print(f"  Dataset: {len(TEST_CASES)} cases")
    print(f"  Sensitive: {sum(1 for _,e in TEST_CASES if e=='SENSITIVE')}")
    print(f"  Public:    {sum(1 for _,e in TEST_CASES if e=='PUBLIC')}")
    print("=" * 70)

    all_results = []

    # --- Experiment 1: Regex only ---
    print("\n>>> EXPERIMENT 1: REGEX ONLY")
    def regex_only(t):
        cls, matches = classify_regex(t)
        if cls == "UNCERTAIN":
            return "PUBLIC", "regex:no_match", 0.0
        return cls, f"regex:{','.join(matches)}", 0.0
    r1 = run_experiment("Regex Only", regex_only)
    print_summary(r1)
    all_results.append(r1)

    # --- Experiments 2+: Hybrid with each model ---
    for model in MODELS_TO_TEST:
        print(f"\n>>> Checking model: {model}")
        if not check_model_available(model):
            print(f"  SKIPPED — {model} not available")
            continue

        # Warm up
        print(f"  Warming up {model}...")
        classify_llm("test warmup", model)

        print(f"\n>>> EXPERIMENT: HYBRID regex + {model}")
        r = run_experiment(f"Hybrid: regex + {model}",
            lambda t, m=model: classify_hybrid(t, m))
        print_summary(r)
        all_results.append(r)

    # --- Final comparison ---
    print("\n" + "=" * 70)
    print("  FINAL COMPARISON")
    print("=" * 70)
    header = f"  {'Experiment':<35s} {'Acc':>5s} {'FNR':>5s} {'FPR':>5s} {'LLM':>4s} {'AvgMs':>7s} {'TotalS':>7s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in all_results:
        print(f"  {r['name']:<35s} {r['accuracy']:>4.1f}% {r['fnr']:>4.1f}% {r['fpr']:>4.1f}% {r['llm_calls']:>4d} {r['avg_latency_ms']:>6.0f}ms {r['total_time_s']:>5.1f}s")

    # Save full results
    out_path = "/tmp/privacy_experiment_results.json"
    # Remove non-serializable items
    for r in all_results:
        r.pop("misses_detail", None)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Full results saved to {out_path}")


if __name__ == "__main__":
    main()

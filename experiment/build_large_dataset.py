#!/usr/bin/env python3
"""
Build a large (2000+) privacy classifier test dataset from real sources.
Sources: ai4privacy (EN+DE), StackOverflow, GermanQuAD, reddit_tifu, hand-crafted.
"""

import json
import random
import hashlib
import re
from datasets import load_dataset

random.seed(42)

def clean(text):
    """Clean and truncate text."""
    text = text.strip().replace("\n", " ").replace("\r", " ")
    text = re.sub(r'\s+', ' ', text)
    return text[:500] if len(text) > 500 else text


def get_ai4privacy(n_en=300, n_de=150):
    """Get PII-containing texts from ai4privacy, filtered by language."""
    print(f"Loading ai4privacy/pii-masking-200k (EN:{n_en}, DE:{n_de})...")
    ds = load_dataset("ai4privacy/pii-masking-200k", split="train", streaming=True)

    en_samples, de_samples = [], []
    seen = set()

    for item in ds:
        text = item.get("source_text", "") or item.get("text", "")
        lang = item.get("language", "")
        if not text or len(text) < 30 or len(text) > 500:
            continue

        h = hashlib.md5(text.encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h)

        t = clean(text)
        if lang == "English" and len(en_samples) < n_en:
            en_samples.append((t, "SENSITIVE"))
        elif lang == "German" and len(de_samples) < n_de:
            de_samples.append((t, "SENSITIVE"))

        if len(en_samples) >= n_en and len(de_samples) >= n_de:
            break

    print(f"  EN: {len(en_samples)}, DE: {len(de_samples)}")
    return en_samples + de_samples


def get_ai4privacy_400k(n_en=200, n_de=150):
    """Try the larger 400k dataset."""
    print(f"Loading ai4privacy/pii-masking-400k (EN:{n_en}, DE:{n_de})...")
    try:
        ds = load_dataset("ai4privacy/pii-masking-400k", split="train", streaming=True)
        en_samples, de_samples = [], []
        seen = set()

        for item in ds:
            text = item.get("source_text", "") or item.get("text", "")
            lang = item.get("language", "")
            if not text or len(text) < 30 or len(text) > 500:
                continue

            h = hashlib.md5(text.encode()).hexdigest()
            if h in seen:
                continue
            seen.add(h)

            t = clean(text)
            if lang == "English" and len(en_samples) < n_en:
                en_samples.append((t, "SENSITIVE"))
            elif lang == "German" and len(de_samples) < n_de:
                de_samples.append((t, "SENSITIVE"))

            if len(en_samples) >= n_en and len(de_samples) >= n_de:
                break

        print(f"  EN: {len(en_samples)}, DE: {len(de_samples)}")
        return en_samples + de_samples
    except Exception as e:
        print(f"  400k failed: {e}")
        return []


def get_stackoverflow(n=200):
    """Real coding questions from StackOverflow — PUBLIC."""
    print(f"Loading StackOverflow questions ({n})...")
    try:
        ds = load_dataset("koutch/stackoverflow_python", split="train", streaming=True)
        samples = []
        seen = set()
        for item in ds:
            title = item.get("title", "")
            if not title or len(title) < 15 or len(title) > 300:
                continue
            h = hashlib.md5(title.encode()).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            samples.append((clean(title), "PUBLIC"))
            if len(samples) >= n:
                break
        print(f"  Got {len(samples)}")
        return samples
    except Exception as e:
        print(f"  StackOverflow failed: {e}")
        return []


def get_germanquad(n=150):
    """German QA questions — PUBLIC."""
    print(f"Loading GermanQuAD ({n})...")
    try:
        ds = load_dataset("deepset/germanquad", split="test", streaming=True)
        samples = []
        seen = set()
        for item in ds:
            q = item.get("question", "")
            if not q or len(q) < 10 or len(q) > 300:
                continue
            h = hashlib.md5(q.encode()).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            samples.append((clean(q), "PUBLIC"))
            if len(samples) >= n:
                break
        print(f"  Got {len(samples)}")
        return samples
    except Exception as e:
        print(f"  GermanQuAD failed: {e}")
        return []


def get_reddit_tifu(n=200):
    """Reddit TIFU posts — personal stories → SENSITIVE."""
    print(f"Loading Reddit TIFU ({n})...")
    try:
        ds = load_dataset("reddit_tifu", "short", split="train", streaming=True)
        samples = []
        seen = set()
        for item in ds:
            text = item.get("documents", "") or item.get("title", "")
            if not text or len(text) < 30 or len(text) > 500:
                continue
            h = hashlib.md5(text.encode()).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            t = clean(text)
            # Reddit TIFU are personal stories — SENSITIVE
            samples.append((t, "SENSITIVE"))
            if len(samples) >= n:
                break
        print(f"  Got {len(samples)}")
        return samples
    except Exception as e:
        print(f"  Reddit TIFU failed: {e}")
        return []


def get_ultrachat(n=200):
    """Chat conversations — mostly PUBLIC technical/educational."""
    print(f"Loading UltraChat ({n})...")
    try:
        ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
        samples = []
        seen = set()
        for item in ds:
            msgs = item.get("messages", [])
            # Get first user message
            for msg in msgs:
                if msg.get("role") == "user":
                    text = msg.get("content", "")
                    if text and 20 < len(text) < 400:
                        h = hashlib.md5(text.encode()).hexdigest()
                        if h not in seen:
                            seen.add(h)
                            samples.append((clean(text), "PUBLIC"))
                    break
            if len(samples) >= n:
                break
        print(f"  Got {len(samples)}")
        return samples
    except Exception as e:
        print(f"  UltraChat failed: {e}")
        return []


# Fallback German PUBLIC questions
GERMAN_PUBLIC_EXTRA = [
    ("Was ist der Unterschied zwischen Bundesrat und Bundestag?", "PUBLIC"),
    ("Wie funktioniert das deutsche Gesundheitssystem?", "PUBLIC"),
    ("Erkläre die Funktionsweise einer Wärmepumpe.", "PUBLIC"),
    ("Was ist der Unterschied zwischen GmbH und AG?", "PUBLIC"),
    ("Wie schreibt man einen Businessplan?", "PUBLIC"),
    ("Was sind die wichtigsten Exportgüter Deutschlands?", "PUBLIC"),
    ("Erkläre den Unterschied zwischen Rente und Pension.", "PUBLIC"),
    ("Wie funktioniert die Krankenversicherung in Deutschland?", "PUBLIC"),
    ("Was ist der Numerus Clausus?", "PUBLIC"),
    ("Wie funktioniert das Wahlsystem in Deutschland?", "PUBLIC"),
    ("Was sind die Vorteile von erneuerbaren Energien?", "PUBLIC"),
    ("Erkläre den Unterschied zwischen HTTP und HTTPS.", "PUBLIC"),
    ("Wie funktioniert ein Verbrennungsmotor?", "PUBLIC"),
    ("Was ist der Unterschied zwischen Haupt- und Realschule?", "PUBLIC"),
    ("Wie funktioniert ein Wechselkurs?", "PUBLIC"),
    ("Was ist die Europäische Zentralbank und was macht sie?", "PUBLIC"),
    ("Erkläre die Grundlagen der Relativitätstheorie.", "PUBLIC"),
    ("Wie funktioniert ein Algorithmus zur Bilderkennung?", "PUBLIC"),
    ("Was sind die häufigsten Programmierfehler in Python?", "PUBLIC"),
    ("Erkläre den MapReduce-Algorithmus.", "PUBLIC"),
    ("Wie funktioniert ein neuronales Netz?", "PUBLIC"),
    ("Was ist der Unterschied zwischen Docker und einer virtuellen Maschine?", "PUBLIC"),
    ("Erkläre das Konzept der Objektorientierung.", "PUBLIC"),
    ("Wie funktioniert Git Branching?", "PUBLIC"),
    ("Was sind die SOLID-Prinzipien in der Softwareentwicklung?", "PUBLIC"),
    ("Wie setzt man einen Linux-Server auf?", "PUBLIC"),
    ("Was ist Kubernetes und wofür wird es verwendet?", "PUBLIC"),
    ("Erkläre den Unterschied zwischen SQL und NoSQL.", "PUBLIC"),
    ("Wie funktioniert ein Load Balancer?", "PUBLIC"),
    ("Was sind Microservices und wann sollte man sie einsetzen?", "PUBLIC"),
]


def build_dataset():
    all_cases = []

    # Real PII data
    all_cases.extend(get_ai4privacy(300, 150))
    all_cases.extend(get_ai4privacy_400k(200, 150))

    # Real coding questions (PUBLIC)
    all_cases.extend(get_stackoverflow(200))

    # German QA (PUBLIC)
    all_cases.extend(get_germanquad(150))

    # Reddit personal stories (SENSITIVE)
    all_cases.extend(get_reddit_tifu(200))

    # Chat conversations (PUBLIC)
    all_cases.extend(get_ultrachat(200))

    # German PUBLIC extras
    all_cases.extend(GERMAN_PUBLIC_EXTRA)

    # Hand-crafted edge cases
    try:
        sys_path_backup = list(sys.path)
        import sys
        sys.path.insert(0, "/tmp")
        from privacy_test_dataset import TEST_CASES
        from german_test_cases import GERMAN_CASES
        all_cases.extend(TEST_CASES)
        all_cases.extend(GERMAN_CASES)
        sys.path = sys_path_backup
    except ImportError:
        print("  Hand-crafted datasets not found, skipping")

    # Deduplicate
    seen = set()
    unique = []
    for text, label in all_cases:
        h = hashlib.md5(text.strip().encode()).hexdigest()
        if h not in seen and len(text.strip()) > 10:
            seen.add(h)
            unique.append((text.strip(), label))

    random.shuffle(unique)

    n_s = sum(1 for _, l in unique if l == "SENSITIVE")
    n_p = sum(1 for _, l in unique if l == "PUBLIC")

    print(f"\n{'='*60}")
    print(f"Final dataset: {len(unique)} cases")
    print(f"  SENSITIVE: {n_s} ({n_s/len(unique)*100:.1f}%)")
    print(f"  PUBLIC:    {n_p} ({n_p/len(unique)*100:.1f}%)")
    print(f"{'='*60}")

    # Save
    out_path = "/tmp/large_test_dataset_v2.py"
    with open(out_path, "w") as f:
        f.write("# Auto-generated privacy classifier test dataset v2\n")
        f.write(f"# {len(unique)} cases: {n_s} SENSITIVE, {n_p} PUBLIC\n\n")
        f.write("TEST_CASES = [\n")
        for text, label in unique:
            escaped = text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")
            f.write(f'    ("{escaped}", "{label}"),\n')
        f.write("]\n")

    print(f"Saved to {out_path}")
    return unique


if __name__ == "__main__":
    import sys
    build_dataset()

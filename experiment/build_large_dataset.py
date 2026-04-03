#!/usr/bin/env python3
"""
Build a large (2000+) privacy classifier test dataset from real sources.

Sources:
- ai4privacy/pii-masking-openpii-1m (1.4M samples, 23 languages incl DE) → SENSITIVE
- ai4privacy/pii-masking-300k (300k, 6 languages incl DE) → SENSITIVE
- OpenAssistant/oasst2 (real user conversations, multilingual incl DE) → mixed
- ctr4si/reddit_tifu (personal stories) → SENSITIVE
- jonathanli/legal-advice-reddit (personal legal situations) → SENSITIVE
- koutch/stackoverflow_python (coding questions) → PUBLIC
- deepset/germanquad (German factual QA) → PUBLIC
- HuggingFaceH4/ultrachat_200k (chat conversations) → PUBLIC
- sroecker/aya_german-sharegpt (German conversations) → PUBLIC
- Hand-crafted edge cases (EN + DE) → mixed
"""

import json
import random
import hashlib
import re
import sys
import os
from datasets import load_dataset

random.seed(42)


def clean(text):
    """Clean and truncate text."""
    text = text.strip().replace("\n", " ").replace("\r", " ")
    text = re.sub(r'\s+', ' ', text)
    return text[:500] if len(text) > 500 else text


def safe_load(name, func, *args, **kwargs):
    """Try loading a dataset, return empty list on failure."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        print(f"  [{name}] FAILED: {e}")
        return []


def dedupe_collect(ds_iter, text_fn, label, seen, n, lang_filter=None, lang_fn=None):
    """Generic collector with dedup and optional language filter."""
    samples = []
    for item in ds_iter:
        if lang_filter and lang_fn:
            if lang_fn(item) != lang_filter:
                continue
        text = text_fn(item)
        if not text or len(text) < 25 or len(text) > 500:
            continue
        h = hashlib.md5(text.encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        samples.append((clean(text), label))
        if len(samples) >= n:
            break
    return samples


# ============================================================
# SENSITIVE SOURCES
# ============================================================

def get_openpii_1m(n_en=400, n_de=200):
    """ai4privacy/pii-masking-openpii-1m — 1.4M samples, 23 languages."""
    print(f"Loading ai4privacy/pii-masking-openpii-1m (EN:{n_en}, DE:{n_de})...")
    ds = load_dataset("ai4privacy/pii-masking-openpii-1m", split="train", streaming=True)

    en_samples, de_samples = [], []
    seen = set()

    for item in ds:
        text = item.get("source_text", "") or item.get("text", "") or item.get("unmasked_text", "")
        lang = (item.get("language", "") or item.get("lang", "")).lower()
        if not text or len(text) < 30 or len(text) > 500:
            continue

        h = hashlib.md5(text.encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h)

        t = clean(text)
        if ("en" in lang or "english" in lang) and len(en_samples) < n_en:
            en_samples.append((t, "SENSITIVE"))
        elif ("de" in lang or "german" in lang) and len(de_samples) < n_de:
            de_samples.append((t, "SENSITIVE"))

        if len(en_samples) >= n_en and len(de_samples) >= n_de:
            break

    print(f"  EN: {len(en_samples)}, DE: {len(de_samples)}")
    return en_samples + de_samples


def get_pii_300k(n_en=200, n_de=150):
    """ai4privacy/pii-masking-300k — 300k samples, 6 languages."""
    print(f"Loading ai4privacy/pii-masking-300k (EN:{n_en}, DE:{n_de})...")
    try:
        ds = load_dataset("ai4privacy/pii-masking-300k", split="train", streaming=True)
        en_samples, de_samples = [], []
        seen = set()

        for item in ds:
            text = item.get("source_text", "") or item.get("text", "") or item.get("unmasked_text", "")
            lang = (item.get("language", "") or item.get("lang", "")).lower()
            if not text or len(text) < 30 or len(text) > 500:
                continue

            h = hashlib.md5(text.encode()).hexdigest()
            if h in seen:
                continue
            seen.add(h)

            t = clean(text)
            if ("en" in lang or "english" in lang) and len(en_samples) < n_en:
                en_samples.append((t, "SENSITIVE"))
            elif ("de" in lang or "german" in lang) and len(de_samples) < n_de:
                de_samples.append((t, "SENSITIVE"))

            if len(en_samples) >= n_en and len(de_samples) >= n_de:
                break

        print(f"  EN: {len(en_samples)}, DE: {len(de_samples)}")
        return en_samples + de_samples
    except Exception as e:
        print(f"  300k failed: {e}")
        return []


def get_reddit_tifu(n=200):
    """Reddit TIFU — personal stories → SENSITIVE."""
    print(f"Loading reddit_tifu ({n})...")
    try:
        ds = load_dataset("ctr4si/reddit_tifu", "short", split="train", streaming=True)
        samples, seen = [], set()
        for item in ds:
            text = item.get("documents", "") or item.get("title", "")
            if not text or len(text) < 30 or len(text) > 500:
                continue
            h = hashlib.md5(text.encode()).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            samples.append((clean(text), "SENSITIVE"))
            if len(samples) >= n:
                break
        print(f"  Got {len(samples)}")
        return samples
    except Exception as e:
        print(f"  reddit_tifu failed: {e}, trying 'reddit_tifu' directly")
        try:
            ds = load_dataset("reddit_tifu", "short", split="train", streaming=True)
            samples, seen = [], set()
            for item in ds:
                text = item.get("documents", "") or item.get("title", "")
                if not text or len(text) < 30:
                    continue
                h = hashlib.md5(text.encode()).hexdigest()
                if h in seen:
                    continue
                seen.add(h)
                samples.append((clean(text)[:500], "SENSITIVE"))
                if len(samples) >= n:
                    break
            print(f"  Got {len(samples)}")
            return samples
        except Exception as e2:
            print(f"  Also failed: {e2}")
            return []


def get_legal_advice_reddit(n=150):
    """Reddit legal advice — personal legal situations → SENSITIVE."""
    print(f"Loading legal-advice-reddit ({n})...")
    try:
        ds = load_dataset("jonathanli/legal-advice-reddit", split="train", streaming=True)
        samples, seen = [], set()
        for item in ds:
            text = item.get("title", "") or item.get("text", "")
            if not text or len(text) < 20 or len(text) > 500:
                continue
            h = hashlib.md5(text.encode()).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            samples.append((clean(text), "SENSITIVE"))
            if len(samples) >= n:
                break
        print(f"  Got {len(samples)}")
        return samples
    except Exception as e:
        print(f"  legal-advice-reddit failed: {e}")
        return []


def get_oasst2_sensitive(n_en=100, n_de=100):
    """OpenAssistant conversations — filter for personal/sensitive content."""
    print(f"Loading OpenAssistant/oasst2 for sensitive content (EN:{n_en}, DE:{n_de})...")
    try:
        ds = load_dataset("OpenAssistant/oasst2", split="train", streaming=True)
        # Keywords that suggest personal content
        sensitive_keywords = re.compile(
            r'\b(my doctor|my salary|I was diagnosed|my therapist|I earn|my debt|'
            r'I was fired|my divorce|mein Arzt|mein Gehalt|meine Schulden|'
            r'ich wurde|mein Anwalt|I owe|my credit|my boss)\b',
            re.IGNORECASE
        )
        en_samples, de_samples = [], []
        seen = set()

        for item in ds:
            text = item.get("text", "")
            lang = item.get("lang", "")
            role = item.get("role", "")
            if role != "prompter" or not text or len(text) < 25 or len(text) > 500:
                continue
            if not sensitive_keywords.search(text):
                continue

            h = hashlib.md5(text.encode()).hexdigest()
            if h in seen:
                continue
            seen.add(h)

            t = clean(text)
            if lang == "en" and len(en_samples) < n_en:
                en_samples.append((t, "SENSITIVE"))
            elif lang == "de" and len(de_samples) < n_de:
                de_samples.append((t, "SENSITIVE"))

            if len(en_samples) >= n_en and len(de_samples) >= n_de:
                break

        print(f"  EN: {len(en_samples)}, DE: {len(de_samples)}")
        return en_samples + de_samples
    except Exception as e:
        print(f"  oasst2 failed: {e}")
        return []


# ============================================================
# PUBLIC SOURCES
# ============================================================

def get_stackoverflow(n=250):
    """StackOverflow coding questions → PUBLIC."""
    print(f"Loading StackOverflow ({n})...")
    try:
        ds = load_dataset("koutch/stackoverflow_python", split="train", streaming=True)
        samples, seen = [], set()
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


def get_germanquad(n=200):
    """German QA — factual questions → PUBLIC."""
    print(f"Loading GermanQuAD ({n})...")
    try:
        ds = load_dataset("deepset/germanquad", split="test", streaming=True)
        samples, seen = [], set()
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


def get_ultrachat(n=250):
    """UltraChat conversations — general/technical → PUBLIC."""
    print(f"Loading UltraChat ({n})...")
    try:
        ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
        samples, seen = [], set()
        for item in ds:
            msgs = item.get("messages", [])
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


def get_german_sharegpt(n=150):
    """German ShareGPT conversations → PUBLIC."""
    print(f"Loading German ShareGPT ({n})...")
    try:
        ds = load_dataset("sroecker/aya_german-sharegpt", split="train", streaming=True)
        samples, seen = [], set()
        for item in ds:
            convs = item.get("conversations", [])
            for msg in convs:
                if msg.get("from") == "human":
                    text = msg.get("value", "")
                    if text and 15 < len(text) < 400:
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
        print(f"  German ShareGPT failed: {e}")
        return []


def get_oasst2_public(n_en=100, n_de=100):
    """OpenAssistant conversations — filter for general/technical content → PUBLIC."""
    print(f"Loading OpenAssistant/oasst2 for public content (EN:{n_en}, DE:{n_de})...")
    try:
        ds = load_dataset("OpenAssistant/oasst2", split="train", streaming=True)
        public_keywords = re.compile(
            r'\b(explain|how does|what is|difference between|implement|algorithm|'
            r'write a|code|function|erkläre|was ist|wie funktioniert|unterschied|'
            r'programmier|tutorial|beispiel)\b',
            re.IGNORECASE
        )
        sensitive_keywords = re.compile(
            r'\b(my |I was |I have |I am |mein |ich bin |ich habe |ich wurde )',
            re.IGNORECASE
        )

        en_samples, de_samples = [], []
        seen = set()

        for item in ds:
            text = item.get("text", "")
            lang = item.get("lang", "")
            role = item.get("role", "")
            if role != "prompter" or not text or len(text) < 20 or len(text) > 400:
                continue
            if not public_keywords.search(text):
                continue
            if sensitive_keywords.search(text):
                continue

            h = hashlib.md5(text.encode()).hexdigest()
            if h in seen:
                continue
            seen.add(h)

            t = clean(text)
            if lang == "en" and len(en_samples) < n_en:
                en_samples.append((t, "PUBLIC"))
            elif lang == "de" and len(de_samples) < n_de:
                de_samples.append((t, "PUBLIC"))

            if len(en_samples) >= n_en and len(de_samples) >= n_de:
                break

        print(f"  EN: {len(en_samples)}, DE: {len(de_samples)}")
        return en_samples + de_samples
    except Exception as e:
        print(f"  oasst2 public failed: {e}")
        return []


# German PUBLIC extras (hand-crafted)
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

    # SENSITIVE sources
    print("\n=== SENSITIVE SOURCES ===")
    all_cases.extend(safe_load("openpii-1m", get_openpii_1m, 400, 200))
    all_cases.extend(safe_load("pii-300k", get_pii_300k, 200, 150))
    all_cases.extend(safe_load("reddit-tifu", get_reddit_tifu, 200))
    all_cases.extend(safe_load("legal-advice", get_legal_advice_reddit, 150))
    all_cases.extend(safe_load("oasst2-sensitive", get_oasst2_sensitive, 100, 100))

    # PUBLIC sources
    print("\n=== PUBLIC SOURCES ===")
    all_cases.extend(safe_load("stackoverflow", get_stackoverflow, 250))
    all_cases.extend(safe_load("germanquad", get_germanquad, 200))
    all_cases.extend(safe_load("ultrachat", get_ultrachat, 250))
    all_cases.extend(safe_load("german-sharegpt", get_german_sharegpt, 150))
    all_cases.extend(safe_load("oasst2-public", get_oasst2_public, 100, 100))

    # Hand-crafted
    print("\n=== HAND-CRAFTED ===")
    all_cases.extend(GERMAN_PUBLIC_EXTRA)

    # Try importing hand-crafted edge cases
    for path in ["/tmp", os.path.dirname(os.path.abspath(__file__))]:
        sys.path.insert(0, path)
    try:
        from privacy_test_dataset import TEST_CASES
        all_cases.extend(TEST_CASES)
        print(f"  Added {len(TEST_CASES)} hand-crafted EN cases")
    except ImportError:
        print("  privacy_test_dataset not found")
    try:
        from german_test_cases import GERMAN_CASES
        all_cases.extend(GERMAN_CASES)
        print(f"  Added {len(GERMAN_CASES)} hand-crafted DE cases")
    except ImportError:
        print("  german_test_cases not found")

    # Deduplicate
    seen = set()
    unique = []
    for text, label in all_cases:
        t = text.strip()
        if len(t) < 10:
            continue
        h = hashlib.md5(t.encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append((t, label))

    random.shuffle(unique)

    n_s = sum(1 for _, l in unique if l == "SENSITIVE")
    n_p = sum(1 for _, l in unique if l == "PUBLIC")

    print(f"\n{'='*60}")
    print(f"  FINAL DATASET: {len(unique)} cases")
    print(f"  SENSITIVE: {n_s} ({n_s/len(unique)*100:.1f}%)")
    print(f"  PUBLIC:    {n_p} ({n_p/len(unique)*100:.1f}%)")
    print(f"{'='*60}")

    # Save
    out_path = os.environ.get("OUTPUT_PATH", "/tmp/large_test_dataset_v2.py")
    with open(out_path, "w") as f:
        f.write("# Auto-generated privacy classifier test dataset v2\n")
        f.write(f"# {len(unique)} cases: {n_s} SENSITIVE, {n_p} PUBLIC\n")
        f.write(f"# Sources: ai4privacy (EN+DE), reddit_tifu, legal-advice-reddit,\n")
        f.write(f"#          OpenAssistant (EN+DE), StackOverflow, GermanQuAD,\n")
        f.write(f"#          UltraChat, German ShareGPT, hand-crafted edge cases\n\n")
        f.write("TEST_CASES = [\n")
        for text, label in unique:
            escaped = text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")
            f.write(f'    ("{escaped}", "{label}"),\n')
        f.write("]\n")

    print(f"  Saved to {out_path}")
    return unique


if __name__ == "__main__":
    build_dataset()

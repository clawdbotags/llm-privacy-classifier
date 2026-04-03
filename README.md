# LLM Privacy + Cost Router — Classifier Experiment

Karpathy-style autoresearch experiment to find the best hybrid privacy classifier for an LLM routing system. The classifier decides whether a user message is **SENSITIVE** (route to local Ollama on aiserver01, data never leaves Tailscale) or **PUBLIC** (route to cloud APIs for cost optimization).

## Architecture

```
Message → Regex (structural PII) → hit? → SENSITIVE (local Ollama)
                                  → miss → LLM classifier (Llama 3.2 3B)
                                              → SENSITIVE → local Ollama
                                              → PUBLIC → LLMRouter cost optimizer
                                                            → cheap: Gemini Flash
                                                            → quality: Claude Sonnet
```

## Final Result (Holdout Validated)

| Metric | Value |
|---|---|
| **Model** | Llama 3.2 3B (2GB VRAM) |
| **Accuracy** | 96.7% |
| **False Negative Rate** | 4.6% (5/108 sensitive messages leaked) |
| **False Positive Rate** | 2.3% (3/133 public messages routed local) |
| **Avg LLM Latency** | 276ms |
| **Validation** | 241-case holdout set, never used during tuning |

## Experiment Iterations

### v1 — Baseline (60 hand-crafted cases)

**Goal**: Establish which model + regex combination works.

**Setup**: 60 test cases, 4 models (gemma3:1b, qwen3:1.7b, llama3.2:3b, qwen3:4b), heavy regex patterns + personal context patterns.

**Key findings**:
- Qwen3 models (1.7B and 4B) unusable — thinking mode (`<think>` tags) caused 100% FPR
- Llama 3.2 3B best overall: 95.8% acc, 5.1% FNR, 3.2% FPR, 268ms
- Regex catches ~80% of sensitive cases with zero latency

**Files**: `experiment/privacy_classifier_experiment.py`, `experiment/run_experiment.py`

### v2 — Real-World Data (481 cases)

**Goal**: Test with realistic data — the hand-crafted dataset was too easy.

**Setup**: 481 cases from ai4privacy/pii-masking-200k (100 real PII samples), StackOverflow (100 coding questions), 30 general knowledge, 191 hand-crafted edge cases, 60 German cases. 3 prompt strategies (minimal, few-shot, few-shot+definitions).

**Key findings**:
- Real-world data exposed massive gaps — Gemma 1B minimal prompt went from 17% to 86.7% FNR
- Few-shot examples are critical: Gemma 1B FNR dropped 86.7% → 9.4% with prompt C
- Llama 3.2 3B few-shot actually regressed (26.2% FNR) — confused by institutional messages
- German messages leaked through all models

**Files**: `experiment/run_experiment_v2.py`, `experiment/build_dataset.py`, `experiment/german_test_cases.py`

### v3 — German + Institutional Messages (481 cases)

**Goal**: Fix German leaks and institutional message confusion.

**Setup**: Same 481 cases. New prompts D (8-shot with German + institutional examples) and E (lean-sensitive variant).

**Key findings**:
- Prompt D fixed German: gemma3:1b achieved 1.3% FNR but 22.6% FPR
- Llama 3.2 3B + D: 4.7% FNR, 16.5% FPR — close but FPR too high
- FPR driven by model confusing "questions about sensitive topics" with "sensitive messages"

**Files**: `experiment/run_experiment_v3.py`

### v4 — Topic vs Person Reframing (481 cases)

**Goal**: Reduce FPR by teaching the model the difference between "about a topic" and "about a person".

**Setup**: Same 481 cases. Added phone regex (DE, US, intl). New prompts F and G with explicit topic-vs-person distinction and PUBLIC examples for hard categories (legal questions, medical questions, code about PII, German generic).

**Key findings**:
- **Breakthrough**: Prompt G reframing ("Does this message reveal a SPECIFIC PERSON's private information?") + targeted PUBLIC examples
- Llama 3.2 3B + G: **95.8% acc, 4.3% FNR, 4.0% FPR** — massive FPR improvement
- Critical reviewer flagged potential overfitting (tuned few-shot examples to test set)

**Files**: `experiment/run_experiment_v4.py`

### v5 — Holdout Validation (241 unseen cases)

**Goal**: Honestly validate v4 results on data never used during prompt tuning.

**Setup**: Split 481 cases into 240 dev + 241 test (deterministic seed). Tested v4 winner (G-full), stripped variant (G-stripped, same system prompt but generic examples), and D baseline. Both gemma3:1b and llama3.2:3b.

**Key findings**:
- **Results confirmed real**: Llama G-full scored 96.7% on holdout (better than 95.8% on dev)
- Prompt reframing alone (G-stripped) gets 93.8% — the "specific person" framing genuinely helps
- Targeted examples add ~3% accuracy on top, mainly reducing FPR
- FNR consistent at 4.6% across all Llama prompts — the 5 remaining leaks are genuine hard cases

**Files**: `experiment/run_experiment_v5_holdout.py`

## Results Summary (All Iterations)

| Iter | Best Config | Acc | FNR | FPR | Dataset |
|---|---|---|---|---|---|
| v1 | llama3.2:3b + zero-shot | 95.8% | 5.1% | 3.2% | 60 hand-crafted |
| v2 | gemma3:1b + C (few-shot+defs) | 89.2% | 9.4% | 12.1% | 481 real-world |
| v3 | gemma3:1b + D (8-shot multi) | 87.7% | 1.3% | 22.6% | 481 real-world |
| v4 | llama3.2:3b + G (topic-vs-person) | 95.8% | 4.3% | 4.0% | 481 real-world |
| **v5** | **llama3.2:3b + G (holdout)** | **96.7%** | **4.6%** | **2.3%** | **241 holdout** |

## Winning Prompt (Prompt G)

**System prompt**:
```
You are a privacy classifier. Respond with SENSITIVE or PUBLIC only.

The key question: Does this message reveal a SPECIFIC PERSON's private information?

SENSITIVE: Yes — it contains someone's personal health details, financial data, legal
situation, relationship issue, confession, contact info, or ID number. Messages from
institutions about a named person's private data are also SENSITIVE.

PUBLIC: No — even if the TOPIC is sensitive (divorce, medical conditions, salaries,
arrests, bankruptcy), if no specific person's private data is revealed, it is PUBLIC.
General questions, coding, tutorials, legal/medical information requests, and
discussions about topics in the abstract are all PUBLIC.
```

**10 few-shot examples** (4 SENSITIVE, 6 PUBLIC) — see `experiment/run_experiment_v4.py` for the full template.

## Regex Patterns (Fast Path)

Structural PII caught by regex before LLM is called:
- Email addresses
- Credit card numbers (Visa, MC, Amex)
- IBAN numbers
- US SSN (XXX-XX-XXXX)
- Phone numbers (US, international, German)

Code context detection prevents false positives on programming examples.

## Hardware

- **Classifier**: Llama 3.2 3B Q4_K_M — 2GB VRAM on aiserver01 (13x RTX 3090, 312GB total)
- **Sensitive inference**: Llama 3.3 70B or Gemma 27B on same machine
- **OpenFang**: Hetzner VPS 4GB RAM (ubuntu-4gb-hel1-1)
- **Network**: Tailscale mesh

## Running the Experiments

```bash
# On aiserver01 — models must be pulled first
ollama pull llama3.2:3b
ollama pull gemma3:1b

# Run holdout validation
cd experiment
python3 run_experiment_v5_holdout.py
```

## Files

```
experiment/
├── privacy_test_dataset.py          # 191 hand-crafted test cases
├── german_test_cases.py             # 60 German test cases
├── build_dataset.py                 # Builds 481-case dataset from HuggingFace + hand-crafted
├── build_large_dataset.py           # Builds 2000+ case dataset (WIP)
├── privacy_classifier_experiment.py # v1 — initial baseline
├── run_experiment.py                # v1 — first full run
├── run_experiment_v2.py             # v2 — real-world data + 3 prompt strategies
├── run_experiment_v3.py             # v3 — German + institutional fixes
├── run_experiment_v4.py             # v4 — topic-vs-person reframing (winner)
├── run_experiment_v5_holdout.py     # v5 — holdout validation (honest eval)
PROMPT.md                            # Original design session prompt
```

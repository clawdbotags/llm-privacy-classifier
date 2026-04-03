# LLM Privacy + Cost Router — Design Session Prompt

Paste this into a new Claude Code session to continue the design.

---

## Context

We migrated from OpenClaw to OpenFang (Rust-based Agent OS) on a Hetzner VPS (ubuntu-4gb-hel1-1). We want to build a privacy-aware + cost-optimized LLM router that sits between OpenFang and the LLM providers.

## Current Setup

- **OpenFang v0.5.5** running on Hetzner VPS (4GB RAM, Ubuntu 24.04)
- **Tailscale network**: VPS (100.127.206.107), aiserver01 (13x RTX 3090, 312GB VRAM, AMD EPYC 7402)
- **Agents**: thesis, coach, engineer, assistant (on Claude Code/Sonnet)
- **Hands**: watchdog, platform-watcher (on OpenRouter/Gemini Flash)
- **Matrix**: @openfang:ubuntu-4gb-hel1-1 for messaging
- **Ollama on aiserver01**: DeepSeek R1 671B, Llama 3.3 70B, Gemma 27B, Mixtral 8x22B, Gemma 3 1B (classifier)
- **Dashboard**: http://100.127.206.107:4200
- **Config**: ~/.openfang/config.toml
- **Source repo**: ~/.openfang/hands/platform-watcher/repo/ (OpenFang source clone)

## What We Want to Build

A router with TWO dimensions:

### 1. Privacy Routing (custom)
- Classify each message as SENSITIVE or PUBLIC
- SENSITIVE → route to local Ollama on aiserver01 (data never leaves Tailscale)
- PUBLIC → pass to cost router

### 2. Cost Routing (LLMRouter built-in)
- Simple public queries → cheap model (Gemini Flash via OpenRouter)
- Complex public queries → expensive model (Claude Sonnet via Claude Code)

### Flow
```
Message → Privacy classifier (regex + Gemma 1B on aiserver01)
  ├── SENSITIVE → aiserver01 Ollama (Llama 3.3 70B or Gemma 27B)
  └── PUBLIC → LLMRouter cost optimizer
                ├── Simple → OpenRouter/Gemini Flash (cheap)
                └── Complex → Claude Code/Sonnet (quality)
```

## Research Findings

### Privacy Classification — Tested & Working
- **Hybrid approach**: regex (PII patterns) + Gemma 3 1B (ambiguous cases)
- **Accuracy**: 100% on 16-case test suite
- **Latency**: regex <0.1ms, Gemma 1B ~420ms (warm)
- **VRAM**: 794 MiB for Gemma 1B (0.25% of aiserver01)
- **Fail-safe**: default to SENSITIVE if uncertain
- Test script exists on aiserver01 at /tmp/test_hybrid.py

### Sensitive Classification Schema
**SENSITIVE** (route local):
- Personal identifiers (names in context, SSN, DOB, passport)
- Contact info (email, phone, address)
- Financial (salary, bank accounts, tax, mortgage, credit score)
- Health (personal diagnoses, medications, blood pressure, conditions)
- Legal/private (arrests, lawsuits, divorce, custody, resignation)
- Named personal relationships ("my boss Mike", "my doctor")

**PUBLIC** (route cloud):
- General knowledge, science, history
- Programming, coding, technical questions
- Generic health/science without personal context
- Research, tutorials, explanations

### LLMRouter (UIUC) — github.com/ulab-uiuc/LLMRouter
- 16 routing strategies for cost/quality optimization
- Plugin system: implement custom `MetaRouter` subclass
- `calculate_strong_win_rate()` can be repurposed for routing decisions
- Handles failover, retries, load balancing, metrics

### Existing Tools
- **LLM-Shield**: closest existing project (Presidio + route mode), small/immature
- **LiteLLM**: tag-based routing + Presidio guardrail, not wired together
- **Microsoft Presidio**: gold standard PII detection, open source
- **OpenFang built-in ModelRouter**: complexity scoring only (routing.rs), no privacy dimension
- **OpenFang taint tracking**: TaintLabel/TaintSet in source — could be leveraged

### Key Architectural Decision
Build a **Python proxy service** that:
1. Receives OpenAI-compatible requests from OpenFang
2. Runs hybrid privacy classifier (regex + Gemma 1B via Ollama on aiserver01)
3. If SENSITIVE: forwards to Ollama on aiserver01 (Llama 3.3 70B or Gemma 27B)
4. If PUBLIC: forwards to LLMRouter which picks cheap vs expensive model
5. Returns response to OpenFang

OpenFang talks to this proxy as a single "OpenAI-compatible provider".

## Questions to Resolve in Design Session

1. Where does the proxy run? VPS or aiserver01?
2. Use LLMRouter library or build routing logic from scratch?
3. How to handle streaming responses through the proxy?
4. Should the proxy log routing decisions for auditing?
5. How to configure per-agent overrides (e.g., thesis always local)?
6. What local model for sensitive queries? Llama 3.3 70B vs Gemma 27B
7. How does this integrate with OpenFang's config.toml?
8. Should it be a Hand, an agent, or a standalone service?
9. Fallback strategy when aiserver01 is unreachable?
10. How to handle the classifier being wrong (false public → data leak)?

## Files to Reference

- OpenFang config: `~/.openfang/config.toml`
- OpenFang quirks: `~/.claude/projects/-home-ags/memory/reference_openfang_quirks.md`
- OpenFang ModelRouter source: `~/.openfang/hands/platform-watcher/repo/crates/openfang-runtime/src/routing.rs`
- OpenFang driver factory: `~/.openfang/hands/platform-watcher/repo/crates/openfang-runtime/src/drivers/mod.rs`
- Watchdog Hand spec: `~/docs/superpowers/specs/2026-04-02-watchdog-hand-design.md`
- Migration spec: `~/docs/superpowers/specs/2026-04-01-openclaw-to-openfang-migration-design.md`
- Platform skill: `~/.openfang/skills/openfang-platform/SKILL.md`

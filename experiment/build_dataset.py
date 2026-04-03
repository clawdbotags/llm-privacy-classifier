#!/usr/bin/env python3
"""
Build a large, realistic test dataset from real-world sources.
Combines: ai4privacy PII data, StackOverflow coding questions,
and hand-crafted edge cases.
"""

import json
import random
import hashlib
from datasets import load_dataset

random.seed(42)

def get_ai4privacy_samples(n=100):
    """Get real PII-containing texts from ai4privacy dataset."""
    print(f"Loading ai4privacy/pii-masking-200k (sampling {n})...")
    ds = load_dataset("ai4privacy/pii-masking-200k", split="train", streaming=True)

    samples = []
    seen = set()
    for item in ds:
        text = item.get("source_text", "") or item.get("text", "")
        if not text or len(text) < 30 or len(text) > 500:
            continue

        # Deduplicate
        h = hashlib.md5(text.encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h)

        # These contain real PII patterns -> SENSITIVE
        samples.append((text.strip(), "SENSITIVE"))
        if len(samples) >= n:
            break

    print(f"  Got {len(samples)} ai4privacy samples")
    return samples


def get_stackoverflow_samples(n=100):
    """Get real coding questions from StackOverflow."""
    print(f"Loading StackOverflow questions (sampling {n})...")
    try:
        ds = load_dataset("koutch/stackoverflow_python", split="train", streaming=True)
        samples = []
        seen = set()
        for item in ds:
            title = item.get("title", "")
            if not title or len(title) < 20 or len(title) > 300:
                continue

            h = hashlib.md5(title.encode()).hexdigest()
            if h in seen:
                continue
            seen.add(h)

            samples.append((title.strip(), "PUBLIC"))
            if len(samples) >= n:
                break

        print(f"  Got {len(samples)} StackOverflow samples")
        return samples
    except Exception as e:
        print(f"  StackOverflow dataset failed: {e}")
        return get_fallback_coding_questions(n)


def get_fallback_coding_questions(n=100):
    """Fallback coding questions if StackOverflow dataset unavailable."""
    questions = [
        "How to merge two dictionaries in Python 3.9+?",
        "What's the difference between list comprehension and generator expression?",
        "How to handle circular imports in Python?",
        "Best way to read large CSV files in pandas without running out of memory?",
        "How to implement retry logic with exponential backoff in Python?",
        "What's the difference between __str__ and __repr__?",
        "How to use asyncio.gather with error handling?",
        "Best practices for Python logging in a multi-module application?",
        "How to create a custom context manager in Python?",
        "What's the difference between deepcopy and shallow copy?",
        "How do I unit test a function that calls an external API?",
        "How to profile Python code for memory usage?",
        "What's the best way to parse command line arguments in Python?",
        "How to implement a singleton pattern in Python?",
        "How to handle timezone-aware datetime objects in Python?",
        "What's the difference between multiprocessing and threading in Python?",
        "How to use type hints with generic classes in Python?",
        "Best way to handle database migrations with SQLAlchemy?",
        "How to implement a LRU cache in Python?",
        "What's the difference between pip install -e and regular pip install?",
        "How to set up pre-commit hooks for a Python project?",
        "How to implement a binary search in Rust?",
        "What's the difference between Box, Rc, and Arc in Rust?",
        "How to handle errors idiomatically in Go?",
        "What's the difference between goroutines and threads?",
        "How to implement a REST API with authentication in FastAPI?",
        "How to use Docker multi-stage builds to reduce image size?",
        "What's the difference between kubectl apply and kubectl create?",
        "How to set up a GitHub Actions CI/CD pipeline for a Node.js app?",
        "How to optimize SQL queries with proper indexing?",
        "What's the N+1 query problem and how to fix it?",
        "How to implement WebSocket connections in a React app?",
        "What's the difference between server-side rendering and static site generation?",
        "How to implement OAuth 2.0 authorization code flow?",
        "How to set up monitoring with Prometheus and Grafana?",
        "What's the best way to handle secrets in Kubernetes?",
        "How to implement a distributed lock using Redis?",
        "How to use Terraform to provision AWS infrastructure?",
        "What's the difference between gRPC and REST?",
        "How to implement pagination in a GraphQL API?",
        "How to write integration tests for a microservice architecture?",
        "What's the difference between eventual consistency and strong consistency?",
        "How to implement a message queue with RabbitMQ?",
        "How to set up a reverse proxy with Nginx?",
        "What are the SOLID principles in object-oriented design?",
        "How to implement the observer pattern in TypeScript?",
        "How to handle database connection pooling in Node.js?",
        "What's the difference between SQL and NoSQL databases?",
        "How to implement rate limiting in an API gateway?",
        "How to use Git rebase vs merge for feature branches?",
        "How to implement a trie data structure for autocomplete?",
        "What's the difference between symmetric and asymmetric encryption?",
        "How to implement CORS properly in a REST API?",
        "How to use environment variables in a Next.js application?",
        "What's the best way to structure a monorepo with Turborepo?",
        "How to implement a pub/sub system in Python?",
        "How do I debug memory leaks in a Node.js application?",
        "What's the difference between TCP and UDP for real-time applications?",
        "How to implement circuit breaker pattern in microservices?",
        "How to set up SSL/TLS certificates for a production server?",
        "What's the difference between process.env and dotenv in Node.js?",
        "How to implement a state machine in Python?",
        "How to use Redis for caching in a Django application?",
        "What's the difference between horizontal and vertical scaling?",
        "How to implement blue-green deployment with Kubernetes?",
        "How to handle file uploads in a FastAPI application?",
        "What's the difference between JWT and session-based authentication?",
        "How to implement a content delivery network strategy?",
        "How to use PostgreSQL full-text search vs Elasticsearch?",
        "How to implement a rate limiter using the token bucket algorithm?",
        "What's the difference between Apache Kafka and RabbitMQ?",
        "How to set up a development environment with Docker Compose?",
        "How to implement a custom allocator in Rust?",
        "What's the CAP theorem and how does it affect database selection?",
        "How to use pandas groupby with multiple aggregation functions?",
        "How to implement dependency injection without a framework?",
        "What's the difference between abstract classes and interfaces in TypeScript?",
        "How to handle long-running tasks in a web application?",
        "How to implement a bloom filter for efficient set membership testing?",
        "How to set up a Kubernetes ingress controller?",
        "What's the difference between optimistic and pessimistic locking?",
        "How to implement server-sent events for real-time updates?",
        "How to use GitHub Copilot effectively for code generation?",
        "What's the best way to handle errors in a React application?",
        "How to implement a custom middleware in Express.js?",
        "How to use async generators in Python for streaming data?",
        "What's the difference between GraphQL subscriptions and WebSockets?",
        "How to implement a plugin system in Python?",
        "How to set up automated testing for a Chrome extension?",
        "What's the difference between Alpine and Debian-based Docker images?",
        "How to implement canary deployments with Istio?",
        "How to use SQLAlchemy 2.0 with async support?",
        "What's the difference between Celery and Dramatiq for task queues?",
        "How to implement a custom hooks system in React?",
        "How to handle binary data in a REST API?",
        "What's the difference between Podman and Docker?",
        "How to implement a changelog generation system?",
        "How to use OpenTelemetry for distributed tracing?",
        "What's the best strategy for database sharding?",
        "How to implement a feature flag system from scratch?",
        "How to handle concurrent writes in a distributed database?",
    ]
    return [(q, "PUBLIC") for q in questions[:n]]


# Additional general knowledge / research questions (PUBLIC)
GENERAL_KNOWLEDGE = [
    ("Explain the difference between supervised and unsupervised learning.", "PUBLIC"),
    ("What is the greenhouse effect and how does it contribute to climate change?", "PUBLIC"),
    ("How do neural networks learn through backpropagation?", "PUBLIC"),
    ("What are the main differences between IPv4 and IPv6?", "PUBLIC"),
    ("Explain the concept of opportunity cost in economics.", "PUBLIC"),
    ("What is the difference between classical and operant conditioning?", "PUBLIC"),
    ("How does CRISPR gene editing technology work?", "PUBLIC"),
    ("What are the principles of clean architecture?", "PUBLIC"),
    ("Explain the Byzantine Generals Problem in distributed computing.", "PUBLIC"),
    ("What is the difference between deductive and inductive reasoning?", "PUBLIC"),
    ("How do black holes form and what happens at the event horizon?", "PUBLIC"),
    ("What are the key differences between Keynesian and Austrian economics?", "PUBLIC"),
    ("Explain how public key cryptography works.", "PUBLIC"),
    ("What is the halting problem and why is it undecidable?", "PUBLIC"),
    ("How does natural language processing handle ambiguity?", "PUBLIC"),
    ("What are the pros and cons of microservice architecture?", "PUBLIC"),
    ("Explain the concept of entropy in information theory.", "PUBLIC"),
    ("What is the difference between correlation and causation?", "PUBLIC"),
    ("How do recommendation systems work?", "PUBLIC"),
    ("What are the main types of machine learning bias?", "PUBLIC"),
    ("Explain the concept of eventual consistency in distributed databases.", "PUBLIC"),
    ("What is the difference between compiled and interpreted languages?", "PUBLIC"),
    ("How does a blockchain consensus mechanism work?", "PUBLIC"),
    ("What are design patterns and when should you use them?", "PUBLIC"),
    ("Explain the concept of technical debt.", "PUBLIC"),
    ("What is the difference between monolithic and microservice architecture?", "PUBLIC"),
    ("How does DNS resolution work step by step?", "PUBLIC"),
    ("What are the ACID properties in database transactions?", "PUBLIC"),
    ("Explain MapReduce and how it enables distributed computing.", "PUBLIC"),
    ("What is the difference between strong and weak AI?", "PUBLIC"),
]


def build_dataset():
    """Build combined dataset."""
    all_cases = []

    # 1. Real PII data from ai4privacy
    try:
        pii_samples = get_ai4privacy_samples(100)
        all_cases.extend(pii_samples)
    except Exception as e:
        print(f"  ai4privacy failed: {e}")

    # 2. Real coding questions
    coding_samples = get_stackoverflow_samples(100)
    all_cases.extend(coding_samples)

    # 3. General knowledge
    all_cases.extend(GENERAL_KNOWLEDGE)

    # 4. Import hand-crafted edge cases
    from privacy_test_dataset import TEST_CASES
    all_cases.extend(TEST_CASES)

    # Deduplicate
    seen = set()
    unique = []
    for text, label in all_cases:
        h = hashlib.md5(text.strip().encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append((text.strip(), label))

    # Shuffle
    random.shuffle(unique)

    n_sensitive = sum(1 for _, l in unique if l == "SENSITIVE")
    n_public = sum(1 for _, l in unique if l == "PUBLIC")

    print(f"\nFinal dataset: {len(unique)} cases")
    print(f"  SENSITIVE: {n_sensitive} ({n_sensitive/len(unique)*100:.1f}%)")
    print(f"  PUBLIC:    {n_public} ({n_public/len(unique)*100:.1f}%)")

    # Save as Python module
    with open("/tmp/large_test_dataset.py", "w") as f:
        f.write("# Auto-generated privacy classifier test dataset\n")
        f.write(f"# {len(unique)} cases: {n_sensitive} SENSITIVE, {n_public} PUBLIC\n\n")
        f.write("TEST_CASES = [\n")
        for text, label in unique:
            escaped = text.replace("\\", "\\\\").replace('"', '\\"')
            f.write(f'    ("{escaped}", "{label}"),\n')
        f.write("]\n")

    print(f"Saved to /tmp/large_test_dataset.py")
    return unique


if __name__ == "__main__":
    build_dataset()

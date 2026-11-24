"""Configuration settings for the customer service application."""

import os

# Ollama API settings
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "smollm2:135m")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

# Safety guard: "llama_guard" (app/guardrails_classifier_model.py) or "keyword" (app/guardrails.py)
SAFETY_GUARD = os.getenv("SAFETY_GUARD", "llama_guard")

# Guard classifier settings (Llama Guard served by its own Ollama instance)
GUARD_BASE_URL = os.getenv("GUARD_BASE_URL", "http://localhost:11435")
GUARD_MODEL = os.getenv("GUARD_MODEL", "llama-guard3:1b")
GUARD_TIMEOUT = float(os.getenv("GUARD_TIMEOUT", "30"))

# Application settings
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", "8000"))

# Phoenix observability settings
PHOENIX_ENDPOINT = os.getenv("PHOENIX_ENDPOINT", "http://localhost:6006")
PHOENIX_ENABLED = os.getenv("PHOENIX_ENABLED", "false").lower() == "true"

# System prompt for the customer service LLM
SYSTEM_PROMPT = """You are a helpful customer service assistant for an e-commerce retail company. 
You assist customers with:
- Order status and tracking
- Shipping and delivery questions
- Returns and refunds
- Product information
- Account management

Provide clear, concise, and helpful responses. If you cannot help with a request, 
politely explain why."""


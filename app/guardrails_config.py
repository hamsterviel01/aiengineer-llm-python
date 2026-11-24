"""Selects the safety guard used by the app from the SAFETY_GUARD setting."""

from app import guardrails_classifier_model, guardrails
from app.config import SAFETY_GUARD

# Safety guards selectable via SAFETY_GUARD; each provides classify_input/classify_output.
SAFETY_GUARDS = {
    "llama_guard": guardrails_classifier_model,
    "keyword": guardrails,
}
if SAFETY_GUARD not in SAFETY_GUARDS:
    raise ValueError(f"SAFETY_GUARD must be one of {sorted(SAFETY_GUARDS)}, got {SAFETY_GUARD!r}")
guard = SAFETY_GUARDS[SAFETY_GUARD]

from __future__ import annotations

from typing import Dict, Iterable, Tuple

from .models import ScoringTemplate, SupportAction

_EPSILON = 1e-3


def _normalize(text: str | None) -> str:
    return (text or "").strip().lower()


def _contains_any(text: str, phrases: Iterable[str]) -> float:
    phrases = [phrase for phrase in phrases if phrase]
    if not phrases:
        return 1.0
    hits = sum(1 for phrase in phrases if phrase.lower() in text)
    return hits / len(phrases)


def _keyword_penalty(text: str, keywords: Iterable[str]) -> float:
    keywords = [keyword for keyword in keywords if keyword]
    if not keywords:
        return 0.0
    hits = sum(1 for keyword in keywords if keyword.lower() in text)
    return hits / len(keywords)


def grade_action(task: ScoringTemplate, action: SupportAction) -> Tuple[float, Dict[str, float]]:
    components: Dict[str, float] = {}
    label = _normalize(action.label)
    queue = _normalize(action.queue)
    priority = _normalize(action.priority)
    response = _normalize(action.response_text or action.customer_reply)
    escalation = _normalize(action.escalation_reason or action.internal_note)
    resolution = _normalize(action.resolution_code)

    components["action_type"] = 1.0 if action.action_type == task.expected_action_type else 0.0
    components["label"] = 1.0 if label == task.expected_label else 0.0
    components["queue"] = 1.0 if queue == task.expected_queue else 0.0
    components["priority"] = 1.0 if priority == task.expected_priority else 0.0
    components["response"] = _contains_any(response, task.response_keywords)
    components["escalation"] = _contains_any(escalation, task.escalation_keywords)
    components["internal"] = _contains_any(escalation, task.required_internal_keywords)
    components["resolution"] = 1.0 if resolution else 0.4

    penalty = _keyword_penalty(response, task.forbidden_keywords) * 0.15
    if action.confidence is not None:
        if 0.0 <= action.confidence <= 1.0:
            components["confidence"] = float(action.confidence)
        else:
            components["confidence"] = 0.0
            penalty += 0.05
    else:
        components["confidence"] = 0.5

    weights = {
        "action_type": 0.14,
        "label": 0.21,
        "queue": 0.19,
        "priority": 0.16,
        "response": 0.12,
        "escalation": 0.10,
        "internal": 0.05,
        "resolution": 0.03,
    }

    score = sum(components[key] * weight for key, weight in weights.items())
    score = max(_EPSILON, min(1.0 - _EPSILON, score - penalty))
    return score, components

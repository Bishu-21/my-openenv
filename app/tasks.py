from __future__ import annotations

from typing import Dict, List

from .models import ScoringTemplate, Ticket


TASKS: Dict[str, ScoringTemplate] = {
    "account_access_classification": ScoringTemplate(
        name="account_access_classification",
        difficulty="easy",
        description=(
            "Classify a login/access issue, route it to the identity queue, and send a helpful reset-focused reply."
        ),
        ticket=Ticket(
            ticket_id="T-1001",
            subject="Can't log in after password reset",
            body="I reset my password twice and still cannot access the dashboard. Please help me get back in.",
            customer_tier="standard",
            sentiment="frustrated",
            product_area="account_access",
            channel="email",
        ),
        expected_label="account_access",
        expected_queue="identity",
        expected_priority="medium",
        expected_action_type="classify",
        response_keywords=("reset", "access", "verify", "login"),
        forbidden_keywords=("refund", "security incident"),
        customer_goal="Restore account access without unnecessary escalation.",
        risk_level="low",
        max_steps=3,
        success_threshold=0.90,
    ),
    "billing_routing": ScoringTemplate(
        name="billing_routing",
        difficulty="medium",
        description=(
            "Identify a double-charge dispute, route it to billing, set the right priority, and acknowledge the refund request."
        ),
        ticket=Ticket(
            ticket_id="T-2002",
            subject="Charged twice for the same invoice",
            body="We were billed twice this month and need a refund review for invoice 1842.",
            customer_tier="pro",
            sentiment="angry",
            product_area="billing",
            channel="portal",
        ),
        expected_label="billing_refund",
        expected_queue="billing",
        expected_priority="high",
        expected_action_type="route",
        response_keywords=("invoice", "refund", "billing", "review"),
        forbidden_keywords=("password", "security incident"),
        customer_goal="Route the dispute to billing with a clear refund review path.",
        risk_level="medium",
        max_steps=3,
        success_threshold=0.92,
    ),
    "security_escalation": ScoringTemplate(
        name="security_escalation",
        difficulty="hard",
        description=(
            "Recognize a possible compromise, escalate to security, set urgent priority, and avoid promising resolution before investigation."
        ),
        ticket=Ticket(
            ticket_id="T-3003",
            subject="Suspicious login and possible account compromise",
            body="A login happened from a new country and the customer reports unexpected changes to permissions.",
            customer_tier="enterprise",
            sentiment="angry",
            product_area="security",
            channel="email",
        ),
        expected_label="security_incident",
        expected_queue="security",
        expected_priority="urgent",
        expected_action_type="escalate",
        response_keywords=("security", "investigate", "urgent", "hold"),
        escalation_keywords=("compromise", "security team", "contain", "hold"),
        forbidden_keywords=("safe", "resolved"),
        required_internal_keywords=("investigate", "contain", "hold"),
        customer_goal="Escalate quickly and reduce risk to the account.",
        risk_level="high",
        max_steps=4,
        success_threshold=0.95,
    ),
}


TASK_ORDER: List[str] = list(TASKS.keys())
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


Priority = Literal["low", "medium", "high", "urgent"]
ActionType = Literal["classify", "route", "draft_reply", "escalate", "resolve"]
QueueName = Literal["general_support", "billing", "security", "identity", "vip"]


class Ticket(BaseModel):
    ticket_id: str
    subject: str
    body: str
    customer_tier: Literal["free", "standard", "pro", "enterprise"]
    sentiment: Literal["calm", "frustrated", "angry"]
    product_area: str
    channel: Literal["email", "chat", "portal", "phone"] = "email"
    language: str = "en"


class SupportAction(BaseModel):
    action_type: ActionType = "classify"
    label: Optional[str] = None
    queue: Optional[QueueName] = None
    priority: Optional[Priority] = None
    response_text: Optional[str] = None
    escalation_reason: Optional[str] = None
    customer_reply: Optional[str] = None
    internal_note: Optional[str] = None
    resolution_code: Optional[str] = None
    confidence: Optional[float] = None


class SupportObservation(BaseModel):
    task_name: str
    task_description: str
    ticket: Ticket
    step_count: int
    max_steps: int
    progress: float
    target_queue: QueueName
    target_priority: Priority
    required_action_type: ActionType
    required_labels: List[str] = Field(default_factory=list)
    required_keywords: List[str] = Field(default_factory=list)
    forbidden_keywords: List[str] = Field(default_factory=list)
    risk_level: Literal["low", "medium", "high"] = "low"
    customer_goal: str = ""
    history: List[Dict[str, Any]] = Field(default_factory=list)
    allowed_actions: List[ActionType] = Field(
        default_factory=lambda: ["classify", "route", "draft_reply", "escalate", "resolve"]
    )
    done: bool = False


class SupportReward(BaseModel):
    total: float
    components: Dict[str, float] = Field(default_factory=dict)


class SupportState(BaseModel):
    episode_id: str
    task_name: str
    task_description: str
    step_count: int
    max_steps: int
    progress: float
    completed: bool
    success: bool
    last_action_error: Optional[str] = None
    target_queue: QueueName
    target_priority: Priority


class ResetRequest(BaseModel):
    task_name: Optional[str] = None
    seed: Optional[int] = None


class StepResponse(BaseModel):
    observation: SupportObservation
    reward: SupportReward
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)
    last_action_error: Optional[str] = None


class ResetResponse(BaseModel):
    observation: SupportObservation
    state: SupportState


@dataclass(frozen=True)
class ScoringTemplate:
    name: str
    difficulty: str
    description: str
    ticket: Ticket
    expected_label: str
    expected_queue: QueueName
    expected_priority: Priority
    expected_action_type: ActionType
    response_keywords: tuple[str, ...]
    escalation_keywords: tuple[str, ...] = ()
    forbidden_keywords: tuple[str, ...] = ()
    required_internal_keywords: tuple[str, ...] = ()
    customer_goal: str = ""
    risk_level: Literal["low", "medium", "high"] = "low"
    allowed_actions: tuple[ActionType, ...] = ("classify", "route", "draft_reply", "escalate", "resolve")
    max_steps: int = 3
    success_threshold: float = 0.95
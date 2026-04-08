from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from uuid import uuid4

from .grader import grade_action
from .models import (
    ResetRequest,
    ResetResponse,
    ScoringTemplate,
    StepResponse,
    SupportAction,
    SupportObservation,
    SupportReward,
    SupportState,
)
from .tasks import TASKS, TASK_ORDER


@dataclass
class EpisodeRuntime:
    episode_id: str
    task: ScoringTemplate
    step_count: int = 0
    progress: float = 0.0
    completed: bool = False
    success: bool = False
    last_action_error: Optional[str] = None
    history: list[dict[str, Any]] = field(default_factory=list)
    ticket_snapshot: Dict[str, Any] = field(default_factory=dict)


class SupportTriageEnvironment:
    def __init__(self) -> None:
        self._task_index = -1
        self._episode: Optional[EpisodeRuntime] = None

    def _pick_task(self, requested: Optional[str], seed: Optional[int]) -> ScoringTemplate:
        if requested and requested in TASKS:
            return TASKS[requested]
        if seed is not None:
            self._task_index = seed % len(TASK_ORDER)
        else:
            self._task_index = (self._task_index + 1) % len(TASK_ORDER)
        return TASKS[TASK_ORDER[self._task_index]]

    def _make_observation(self) -> SupportObservation:
        assert self._episode is not None
        task = self._episode.task
        ticket = deepcopy(task.ticket)
        return SupportObservation(
            task_name=task.name,
            task_description=task.description,
            ticket=ticket,
            step_count=self._episode.step_count,
            max_steps=task.max_steps,
            progress=self._episode.progress,
            target_queue=task.expected_queue,
            target_priority=task.expected_priority,
            required_action_type=task.expected_action_type,
            required_labels=[task.expected_label],
            required_keywords=list(task.response_keywords),
            forbidden_keywords=list(task.forbidden_keywords),
            risk_level=task.risk_level,
            customer_goal=task.customer_goal,
            history=deepcopy(self._episode.history),
            allowed_actions=list(task.allowed_actions),
            done=self._episode.completed,
        )

    def _make_state(self) -> SupportState:
        assert self._episode is not None
        return SupportState(
            episode_id=self._episode.episode_id,
            task_name=self._episode.task.name,
            task_description=self._episode.task.description,
            step_count=self._episode.step_count,
            max_steps=self._episode.task.max_steps,
            progress=self._episode.progress,
            completed=self._episode.completed,
            success=self._episode.success,
            last_action_error=self._episode.last_action_error,
            target_queue=self._episode.task.expected_queue,
            target_priority=self._episode.task.expected_priority,
        )

    def reset(self, request: Optional[ResetRequest] = None) -> ResetResponse:
        request = request or ResetRequest()
        task = self._pick_task(request.task_name, request.seed)
        self._episode = EpisodeRuntime(
            episode_id=str(uuid4()),
            task=task,
            ticket_snapshot=task.ticket.model_dump(),
        )
        return ResetResponse(observation=self._make_observation(), state=self._make_state())

    def step(self, action: SupportAction) -> StepResponse:
        if self._episode is None:
            self.reset()
        assert self._episode is not None

        if self._episode.completed:
            self._episode.last_action_error = "episode already completed"
            return StepResponse(
                observation=self._make_observation(),
                reward=SupportReward(total=0.0, components={"terminal": 0.0}),
                done=True,
                info={"success": self._episode.success, "message": "episode already completed"},
                last_action_error=self._episode.last_action_error,
            )

        self._episode.step_count += 1
        score, components = grade_action(self._episode.task, action)
        previous_progress = self._episode.progress
        improvement = max(0.0, score - previous_progress)
        regression = max(0.0, previous_progress - score)
        forbidden_penalty = 0.15 if any(
            keyword.lower() in (action.response_text or action.customer_reply or "").lower()
            for keyword in self._episode.task.forbidden_keywords
        ) else 0.0
        step_penalty = 0.01 * max(0, self._episode.step_count - 1)

        reward_value = (improvement * 0.9) - (regression * 0.2) - step_penalty - forbidden_penalty
        self._episode.progress = max(previous_progress, score)
        self._episode.success = self._episode.progress >= self._episode.task.success_threshold
        self._episode.completed = self._episode.success or self._episode.step_count >= self._episode.task.max_steps
        self._episode.last_action_error = None

        if self._episode.success:
            reward_value += 0.35
        elif self._episode.completed:
            reward_value -= 0.05

        reward_value = max(0.0, min(1.0, reward_value))
        reward = SupportReward(
            total=reward_value,
            components={
                "score": round(score, 4),
                "improvement": round(improvement, 4),
                "regression": round(regression, 4),
                "step_penalty": round(-step_penalty, 4),
                "forbidden_penalty": round(-forbidden_penalty, 4),
                "terminal_bonus": 0.35 if self._episode.success else 0.0,
            },
        )

        self._episode.history.append(
            {
                "step": self._episode.step_count,
                "action": action.model_dump(),
                "score": round(score, 4),
                "reward": round(reward_value, 4),
                "components": components,
            }
        )

        info = {
            "task_name": self._episode.task.name,
            "difficulty": self._episode.task.difficulty,
            "success": self._episode.success,
            "progress": round(self._episode.progress, 4),
            "remaining_steps": max(0, self._episode.task.max_steps - self._episode.step_count),
        }

        return StepResponse(
            observation=self._make_observation(),
            reward=reward,
            done=self._episode.completed,
            info=info,
            last_action_error=None,
        )

    @property
    def state(self) -> SupportState:
        if self._episode is None:
            return SupportState(
                episode_id="",
                task_name="",
                task_description="",
                step_count=0,
                max_steps=0,
                progress=0.0,
                completed=False,
                success=False,
                last_action_error="environment not reset",
                target_queue="general_support",
                target_priority="medium",
            )
        return self._make_state()
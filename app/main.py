from __future__ import annotations

from fastapi import FastAPI

from .environment import SupportTriageEnvironment
from .models import ResetRequest, SupportAction


app = FastAPI(title="OpenEnv Support Triage", version="0.1.0")
environment = SupportTriageEnvironment()


@app.get("/")
def root() -> dict[str, str]:
    return {"status": "ok", "name": "OpenEnv Support Triage"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/reset")
def reset(request: ResetRequest | None = None) -> dict:
    return environment.reset(request).model_dump()


@app.post("/step")
def step(action: SupportAction) -> dict:
    return environment.step(action).model_dump()


@app.get("/state")
def state() -> dict:
    return environment.state.model_dump()
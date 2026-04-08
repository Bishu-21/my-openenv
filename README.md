# OpenEnv Support Triage

A lightweight OpenEnv submission scaffold for a realistic customer support workflow. The agent triages incoming tickets, chooses the right queue and priority, and drafts a response or escalation note.

## What the environment does

Each episode simulates one support ticket. The agent sees the ticket context, then submits structured actions such as classifying the issue, routing it, drafting a reply, or escalating it.

## Action space

`SupportAction` supports these fields:

- `action_type` - `classify`, `route`, `draft_reply`, `escalate`, or `resolve`
- `label` - issue label used by the grader
- `queue` - `general_support`, `billing`, `security`, `identity`, or `vip`
- `priority` - `low`, `medium`, `high`, or `urgent`
- `response_text` - customer-facing reply
- `escalation_reason` - internal escalation summary
- `customer_reply`, `internal_note`, `resolution_code`, `confidence` - optional helper fields

## Observation space

`SupportObservation` includes:

- ticket subject, body, customer tier, sentiment, and channel
- target queue, target priority, and required action type
- required labels, keywords, and forbidden keywords
- episode progress, step count, max steps, and action history
- customer goal and risk level

## Reward design

Rewards are shaped from graded progress, not just terminal success.

- partial credit for the right label, queue, priority, and response content
- small step penalty to discourage looping
- penalty for forbidden or risky wording
- terminal bonus when the task reaches the success threshold

All grader scores and reward values are clipped to the range `0.0` to `1.0`.

## Tasks

- `account_access_classification` - easy, classify a login/access issue and route to identity support
- `billing_routing` - medium, identify a billing dispute and route to billing with the correct priority
- `security_escalation` - hard, recognize a possible compromise and escalate urgently to security

## Files

- `inference.py` - root-level baseline runner using the OpenAI client against the Hugging Face router
- `app/main.py` - FastAPI app with `/`, `/health`, `/reset`, `/step`, and `/state`
- `app/environment.py` - support triage environment logic
- `app/grader.py` - deterministic task scoring
- `app/tasks.py` - task definitions
- `app/models.py` - typed request, response, and state models
- `openenv.yaml` - OpenEnv metadata
- `Dockerfile` - container entrypoint

## Setup

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Run the baseline

```bash
python inference.py
```

## Environment variables

- `API_BASE_URL` - LLM endpoint, defaults to `https://router.huggingface.co/v1`
- `MODEL_NAME` - model name, defaults to `gpt-4.1-mini`
- `HF_TOKEN` - required API key
- `OPENENV_BASE_URL` - environment server URL, defaults to `http://127.0.0.1:8000`
- `BENCHMARK` - optional benchmark label printed in the logs

## Docker

```bash
docker build -t openenv-support-triage .
docker run --rm -p 7860:7860 openenv-support-triage
```

## Validation checklist

- `inference.py` is in the repo root
- `HF_TOKEN` is required and `API_BASE_URL` and `MODEL_NAME` have defaults
- stdout only uses the required `[START]`, `[STEP]`, and `[END]` lines
- the Space responds successfully on `/reset`
- `docker build` works
- `openenv validate` passes
- the baseline script completes on all three tasks

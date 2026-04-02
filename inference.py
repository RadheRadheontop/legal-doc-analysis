#!/usr/bin/env python3
"""Baseline agent runner for LexCrisis.

Execution modes
---------------
- **LLM mode** (default when HF_TOKEN or API_KEY is set and USE_LLM_BASELINE=1):
  The agent queries the model at every step and parses a valid action from the
  JSON it returns.  If the model returns an unparseable response the scripted
  fallback action is used so the episode never stalls.
- **Scripted mode** (default when no API key is present):
  Deterministic reference policy that replays SCRIPTED_BASELINES.  Produces
  identical scores on every run.

Environment variables
---------------------
API_BASE_URL     – defaults to https://router.huggingface.co/v1
MODEL_NAME       – defaults to Qwen/Qwen2.5-72B-Instruct
HF_TOKEN         – preferred credential (checked first)
API_KEY          – fallback credential
USE_LLM_BASELINE – set to 1/true/yes to activate LLM mode
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

from lexcrisis_env.env import BENCHMARK_NAME, LexCrisisEnvironment
from lexcrisis_env.models import Action
from lexcrisis_env.tasks import SCRIPTED_BASELINES, TASK_ACTIONS, TASK_DEFINITIONS

# ---------------------------------------------------------------------------
# Configuration (read from environment – no hardcoded secrets)
# ---------------------------------------------------------------------------
API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY: str = os.environ.get("HF_TOKEN") or os.environ.get("API_KEY", "")
USE_LLM_BASELINE: bool = os.environ.get("USE_LLM_BASELINE", "0").lower() in {
    "1",
    "true",
    "yes",
}

_SYSTEM_PROMPT = (
    "You are a legal operations AI agent solving tasks in the LexCrisis benchmark. "
    "At each step you must return ONLY a single JSON object that matches the Action schema:\n"
    '  {"action_type": "<type>", "parameters": {<key>: <value>, ...}}\n'
    "Select the most appropriate action from the available_actions list in the observation. "
    "Do not add any prose, markdown fences, or extra keys. Return only the JSON object."
)


# ---------------------------------------------------------------------------
# Logging helpers (exact format required by the spec)
# ---------------------------------------------------------------------------

def action_string(action: Dict[str, Any]) -> str:
    return json.dumps(action, sort_keys=True, separators=(",", ":"))


def emit_start(task_id: str) -> None:
    print(f"[START] task={task_id} env={BENCHMARK_NAME} model={MODEL_NAME}")
    sys.stdout.flush()


def emit_step(
    step: int,
    action: Dict[str, Any],
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    error_value = error if error is not None else "null"
    print(
        f"[STEP] step={step} action={action_string(action)} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_value}"
    )
    sys.stdout.flush()


def emit_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_text = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_text}")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------

def build_client() -> Optional[OpenAI]:
    """Return an OpenAI client if credentials are available, else None."""
    if not API_KEY:
        return None
    return OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


# ---------------------------------------------------------------------------
# LLM action selection
# ---------------------------------------------------------------------------

def _parse_action_from_text(text: str, available_actions: List[str]) -> Optional[Dict[str, Any]]:
    """Try to extract a valid JSON action dict from the model's response text."""
    # Strip markdown fences if present
    text = re.sub(r"```(?:json)?\s*", "", text).strip()
    # Try the whole text
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "action_type" in obj:
            act_type = obj.get("action_type", "")
            if act_type in available_actions:
                return obj
    except (json.JSONDecodeError, ValueError):
        pass
    # Try to find the first {...} block
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group())
            if isinstance(obj, dict) and "action_type" in obj:
                act_type = obj.get("action_type", "")
                if act_type in available_actions:
                    return obj
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def get_llm_action(
    client: Optional[OpenAI],
    task_id: str,
    step_index: int,
    observation_dict: Dict[str, Any],
    fallback_action: Dict[str, Any],
) -> Dict[str, Any]:
    """Query the LLM for the next action.

    If the LLM is unavailable or returns unparseable output, the scripted
    fallback action is used to keep the episode on track.
    """
    if client is None or not USE_LLM_BASELINE:
        return fallback_action

    available_actions: List[str] = observation_dict.get("available_actions", [])

    # Compose a compact prompt so the model can act without reading a wall of JSON
    prompt_obs = {
        "task_id": task_id,
        "step": step_index,
        "feedback": observation_dict.get("feedback", ""),
        "available_actions": available_actions,
        "findings_summary": {
            k: v
            for k, v in observation_dict.get("findings", {}).items()
            if v  # omit empty collections
        },
        "active_deadlines": observation_dict.get("active_deadlines", []),
        "ethical_alerts": observation_dict.get("ethical_alerts", []),
    }

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0,
            max_tokens=256,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Current observation:\n{json.dumps(prompt_obs, indent=2)}\n\n"
                        f"Scripted suggestion (use only if helpful): "
                        f"{json.dumps(fallback_action)}"
                    ),
                },
            ],
        )
        raw_text = response.choices[0].message.content or ""
        parsed = _parse_action_from_text(raw_text, available_actions)
        if parsed is not None:
            return parsed
    except Exception:
        pass  # Fall through to scripted baseline

    return fallback_action


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_task(task_id: str, client: Optional[OpenAI]) -> Dict[str, Any]:
    env = LexCrisisEnvironment()
    rewards: List[float] = []
    step_index = 0
    success = False
    final_score = 0.0

    scripted = SCRIPTED_BASELINES[task_id]

    emit_start(task_id)
    try:
        observation = env.reset(task_id=task_id)
        obs_dict: Dict[str, Any] = observation.model_dump(mode="json")

        for idx, raw_action in enumerate(scripted):
            step_index += 1
            agent_action = get_llm_action(
                client,
                task_id,
                step_index,
                obs_dict,
                raw_action,
            )

            error_message: Optional[str] = None
            done = False
            reward = 0.0

            try:
                observation = env.step(Action.model_validate(agent_action))
                state = env.state
                reward = float(state.reward or observation.reward or 0.0)
                done = bool(state.done or observation.done)
                final_score = env.last_score
                obs_dict = observation.model_dump(mode="json")
            except Exception as exc:  # pragma: no cover
                error_message = str(exc)

            rewards.append(round(reward, 2))
            emit_step(step_index, agent_action, reward, done, error_message)

            if error_message is not None:
                break
            if done:
                success = True
                break

        if not success and env.state.done:
            success = True
            final_score = env.last_score
    finally:
        final_score = env.last_score
        env.close()
        emit_end(success, step_index, rewards)

    return {
        "task_id": task_id,
        "task_name": TASK_DEFINITIONS[task_id].name,
        "score": final_score,
        "steps": step_index,
        "success": success,
        "rewards": rewards,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    client = build_client()

    if USE_LLM_BASELINE and client is not None:
        print(f"# LLM mode active — model: {MODEL_NAME}, base: {API_BASE_URL}")
    else:
        print("# Scripted mode — deterministic baseline (no API key / USE_LLM_BASELINE=0)")
    sys.stdout.flush()

    results = [run_task(task_id, client) for task_id in TASK_DEFINITIONS]

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    (output_dir / "baseline_scores.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

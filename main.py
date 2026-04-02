"""Root application entrypoint for the LexCrisis OpenEnv server."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import uvicorn
from fastapi.responses import FileResponse
from openenv.core.env_server import create_app

from lexcrisis_env.env import LexCrisisEnvironment
from lexcrisis_env.models import Action, Observation
from lexcrisis_env.tasks import SCRIPTED_BASELINES, TASK_DEFINITIONS

UI_PATH = Path(__file__).resolve().parent / "server" / "ui.html"

app = create_app(
    env=LexCrisisEnvironment,
    action_cls=Action,
    observation_cls=Observation,
    env_name="LexCrisis",
)


@app.get("/", include_in_schema=False)
def root() -> FileResponse:
    response = FileResponse(UI_PATH)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.get("/tasks")
def tasks() -> Dict[str, Any]:
    return {
        "tasks": [
            {
                "task_id": task.task_id,
                "name": task.name,
                "difficulty": task.difficulty,
                "description": task.description,
                "max_steps": task.max_steps,
                "baseline_steps": len(SCRIPTED_BASELINES[task.task_id]),
            }
            for task in TASK_DEFINITIONS.values()
        ]
    }


@app.get("/baselines")
def baselines() -> Dict[str, Any]:
    return {"baselines": SCRIPTED_BASELINES}


@app.get("/episode")
def episode() -> Dict[str, Any]:
    return LexCrisisEnvironment().episode_info()


def main(host: str = "0.0.0.0", port: int = 7860) -> None:
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()

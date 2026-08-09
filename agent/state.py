from pathlib import Path
import json
from typing import Any


STATE_FILE = Path("memory/research_state.json")


def load_state() -> dict:
    """
    Load the agent's current research state.
    """

    if not STATE_FILE.exists():
        raise FileNotFoundError(
            f"State file not found: {STATE_FILE}"
        )

    with STATE_FILE.open(
        "r",
        encoding="utf-8"
    ) as file:
        state = json.load(file)

    return state


def save_state(state: dict) -> None:
    """
    Persist the agent's current research state.
    """

    STATE_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with STATE_FILE.open(
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            state,
            file,
            indent=2,
            ensure_ascii=False
        )


def increment_step(state: dict) -> dict:
    """
    Increase the action counter by one.
    """

    state["step"] = state.get("step", 0) + 1

    return state


def can_continue(state: dict) -> bool:
    """
    Check whether the agent is still allowed to act.
    """

    step = state.get("step", 0)
    max_steps = state.get("max_steps", 5)

    return (
        step < max_steps
        and state.get("status") not in {
            "completed",
            "failed"
        }
    )


def add_search(
    state: dict,
    query: str
) -> dict:
    """
    Record a search query if it has not already been used.
    """

    searches = state.setdefault(
        "searches",
        []
    )

    if query not in searches:
        searches.append(query)

    return state


def add_source(
    state: dict,
    url: str
) -> dict:
    """
    Record a source URL if it has not already been seen.
    """

    sources = state.setdefault(
        "sources_seen",
        []
    )

    if url not in sources:
        sources.append(url)

    return state


def add_note(
    state: dict,
    note: str
) -> dict:
    """
    Add a research observation to memory.
    """

    notes = state.setdefault(
        "notes",
        []
    )

    notes.append(note)

    return state


def set_best_candidate(
    state: dict,
    candidate: dict
) -> dict:
    """
    Store the current best research candidate.
    """

    state["best_candidate"] = candidate

    return state


def set_status(
    state: dict,
    status: str
) -> dict:
    """
    Update task status.
    """

    allowed_statuses = {
        "not_started",
        "researching",
        "completed",
        "failed"
    }

    if status not in allowed_statuses:
        raise ValueError(
            f"Invalid status: {status}"
        )

    state["status"] = status

    return state

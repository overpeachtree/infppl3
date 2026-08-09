from __future__ import annotations

import json
import os
from typing import Any

from google import genai
from google.genai import types

from agent.prompts import SYSTEM_PROMPT, build_agent_prompt
from agent.state import (
    load_state,
    save_state,
    increment_step,
    can_continue,
    add_search,
    add_source,
    add_note,
    set_best_candidate,
    set_status,
)

from tools.web_search import search_web
from tools.file_reader import read_person_file
from tools.file_writer import write_proposal


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

DEFAULT_MODEL = "gemini-3-flash-preview"

ALLOWED_ACTIONS = {
    "read_person_file",
    "search_web",
    "write_proposal",
}


# ---------------------------------------------------------
# STRUCTURED ACTION SCHEMA
# ---------------------------------------------------------

ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": [
                "read_person_file",
                "search_web",
                "write_proposal"
            ]
        },
        "reason": {
            "type": "string"
        },
        "arguments": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string"
                },
                "query": {
                    "type": "string"
                },
                "content": {
                    "type": "string"
                }
            }
        },
        "best_candidate": {
            "type": ["object", "null"],
            "properties": {
                "title": {
                    "type": "string"
                },
                "url": {
                    "type": "string"
                },
                "confidence": {
                    "type": "number"
                },
                "reason": {
                    "type": "string"
                }
            }
        },
        "note": {
            "type": ["string", "null"]
        }
    },
    "required": [
        "action",
        "reason",
        "arguments",
        "best_candidate",
        "note"
    ]
}


# ---------------------------------------------------------
# GEMINI DECISION
# ---------------------------------------------------------

def ask_agent(
    client: genai.Client,
    model: str,
    person_profile: str,
    state: dict,
    latest_observation: str | None
) -> dict:
    """
    Ask Gemini to choose exactly one next action.
    """

    prompt = build_agent_prompt(
        person_profile=person_profile,
        state=state,
        latest_observation=latest_observation
    )

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_json_schema=ACTION_SCHEMA,
            temperature=0.2
        )
    )

    if not response.text:
        raise RuntimeError(
            "Gemini returned an empty response."
        )

    decision = json.loads(
        response.text
    )

    return decision


# ---------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------

def validate_action(
    decision: dict,
    state: dict
) -> None:
    """
    Enforce harness rules before executing any tool.
    """

    action = decision.get("action")

    if action not in ALLOWED_ACTIONS:
        raise RuntimeError(
            f"Agent requested forbidden action: {action}"
        )

    arguments = decision.get(
        "arguments",
        {}
    )

    if action == "read_person_file":

        filename = arguments.get("filename")

        if not filename:
            raise RuntimeError(
                "read_person_file requires filename."
            )

        expected_file = state.get(
            "person_file"
        )

        if filename != expected_file:
            raise RuntimeError(
                "Agent attempted to read an "
                "unauthorized person file."
            )

    elif action == "search_web":

        query = arguments.get("query")

        if not query:
            raise RuntimeError(
                "search_web requires query."
            )

        if query in state.get(
            "searches",
            []
        ):
            raise RuntimeError(
                "Agent attempted to repeat "
                "an existing search."
            )

    elif action == "write_proposal":

        content = arguments.get(
            "content"
        )

        if not content:
            raise RuntimeError(
                "write_proposal requires content."
            )

        if not state.get(
            "best_candidate"
        ):
            raise RuntimeError(
                "Agent cannot write a proposal "
                "without a best candidate."
            )


# ---------------------------------------------------------
# TOOL EXECUTION
# ---------------------------------------------------------

def execute_action(
    decision: dict,
    state: dict
) -> tuple[str, dict]:
    """
    Execute exactly one approved tool.
    """

    action = decision["action"]
    arguments = decision.get(
        "arguments",
        {}
    )

    print()
    print("=" * 70)
    print(f"AGENT ACTION: {action}")
    print(f"REASON: {decision['reason']}")
    print("=" * 70)

    # -----------------------------------------
    # READ
    # -----------------------------------------

    if action == "read_person_file":

        filename = arguments[
            "filename"
        ]

        content = read_person_file(
            filename
        )

        observation = (
            "Person profile successfully read:\n\n"
            + content
        )

    # -----------------------------------------
    # SEARCH
    # -----------------------------------------

    elif action == "search_web":

        query = arguments[
            "query"
        ]

        results = search_web(
            query
        )

        state = add_search(
            state,
            query
        )

        for result in results:

            url = result.get(
                "url"
            )

            if url:
                state = add_source(
                    state,
                    url
                )

        observation = json.dumps(
            {
                "query": query,
                "results": results
            },
            indent=2,
            ensure_ascii=False
        )

    # -----------------------------------------
    # WRITE
    # -----------------------------------------

    elif action == "write_proposal":

        content = arguments[
            "content"
        ]

        output_path = write_proposal(
            content
        )

        observation = (
            "Proposal successfully written to "
            f"{output_path}."
        )

        state = set_status(
            state,
            "completed"
        )

    else:

        raise RuntimeError(
            f"Unhandled action: {action}"
        )

    return observation, state


# ---------------------------------------------------------
# APPLY AGENT OBSERVATIONS TO STATE
# ---------------------------------------------------------

def update_state_from_decision(
    state: dict,
    decision: dict
) -> dict:

    note = decision.get(
        "note"
    )

    if note:

        state = add_note(
            state,
            note
        )

    candidate = decision.get(
        "best_candidate"
    )

    if candidate:

        state = set_best_candidate(
            state,
            candidate
        )

    return state


# ---------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------

def main():

    api_key = os.getenv(
        "GEMINI_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing."
        )

    model = os.getenv(
        "GEMINI_MODEL",
        DEFAULT_MODEL
    )

    client = genai.Client(
        api_key=api_key
    )

    state = load_state()

    # Start a fresh run if needed
    if state["status"] == "not_started":
        state = set_status(
            state,
            "researching"
        )
        save_state(state)

    # Read canonical person profile ourselves.
    # This is bootstrapping context for the harness.
    person_profile = read_person_file(
        state["person_file"]
    )

    latest_observation = (
        "The harness loaded the canonical "
        "person profile. Decide what to do next."
    )

    print()
    print("Starting research agent.")
    print("Goal:", state["goal"])
    print(
        f"Maximum actions: "
        f"{state['max_steps']}"
    )
    print("Model:", model)

    while can_continue(state):

        print()
        print(
            f"STEP {state['step'] + 1} "
            f"OF {state['max_steps']}"
        )

        decision = ask_agent(
            client=client,
            model=model,
            person_profile=person_profile,
            state=state,
            latest_observation=latest_observation
        )

        print()
        print("AGENT DECISION:")
        print(
            json.dumps(
                decision,
                indent=2,
                ensure_ascii=False
            )
        )

        validate_action(
            decision,
            state
        )

        state = update_state_from_decision(
            state,
            decision
        )

        latest_observation, state = execute_action(
            decision,
            state
        )

        state = increment_step(
            state
        )

        save_state(
            state
        )

        print()
        print("OBSERVATION:")
        print(latest_observation)

        if (
            state.get("status")
            == "completed"
        ):
            break

    # -----------------------------------------
    # Stopping boundary
    # -----------------------------------------

    if (
        state.get("status")
        != "completed"
    ):

        state = set_status(
            state,
            "failed"
        )

        state = add_note(
            state,
            "Agent reached the maximum "
            "number of actions before "
            "completing the research goal."
        )

        save_state(
            state
        )

        print()
        print(
            "STOPPED: maximum action "
            "limit reached."
        )

    else:

        print()
        print(
            "COMPLETED: research proposal "
            "created for human review."
        )


if __name__ == "__main__":
    main()

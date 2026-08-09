from pathlib import Path


OUTPUT_DIRECTORY = Path("outputs")


def write_proposal(content: str):
    """
    Write the agent's proposed research update.

    The agent is NOT allowed to directly modify
    canonical files in people/.
    """

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )

    path = OUTPUT_DIRECTORY / "proposed_update.md"

    print(f"\n[TOOL] write_proposal")
    print(f"Writing: {path}")

    path.write_text(
        content,
        encoding="utf-8"
    )

    return str(path)

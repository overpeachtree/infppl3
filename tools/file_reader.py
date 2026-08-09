from pathlib import Path


ALLOWED_DIRECTORY = Path("people")


def read_person_file(filename: str):
    """
    Read a Markdown file from the people directory.
    """

    path = ALLOWED_DIRECTORY / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Person file not found: {path}"
        )

    print(f"\n[TOOL] read_person_file")
    print(f"Reading: {path}")

    return path.read_text(
        encoding="utf-8"
    )

"""O02 collection containment. This policy is not an execution sandbox."""
from pathlib import Path
import os
import pytest

_ROOT = Path(__file__).resolve().parent


def _ignored_path(path):
    """Anchor resolved paths at this repo; never match a substring elsewhere."""
    candidate = Path(str(path).replace("\\", os.sep)).resolve()
    try:
        relative = candidate.relative_to(_ROOT)
    except ValueError:
        return False
    parts = tuple(part.casefold() for part in relative.parts)
    if not parts:
        return False
    return parts[0] == "outputs" or (
        len(parts) == 1 and parts[0].startswith("test_v092_")
        and parts[0].endswith(".py")
    )


def pytest_ignore_collect(collection_path, config):
    # Called before module import during recursive discovery.
    return True if _ignored_path(collection_path) else None


@pytest.hookimpl(tryfirst=True)
def pytest_collection(session):
    # pytest bypasses ignore_collect for explicit initial paths. Reject these
    # before its built-in collector runs; never import a prohibited module.
    for argument in session.config.args:
        raw_path = argument.split("::", 1)[0].replace("\\", os.sep)
        path = Path(raw_path)
        if not path.is_absolute():
            path = session.config.invocation_params.dir / path
        if _ignored_path(path):
            raise pytest.UsageError("O02 collection blocked for retired scratch/output path")

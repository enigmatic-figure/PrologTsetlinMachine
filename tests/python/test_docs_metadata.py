from __future__ import annotations

import pytest

from scripts.check_docs import _normalized_repo_path


def test_repository_relative_posix_path_is_accepted() -> None:
    failures: list[str] = []
    value = _normalized_repo_path(
        "docs/manual/how-to/install.md",
        prefix="inventory:2",
        field="destination",
        failures=failures,
    )
    assert value is not None
    assert value.as_posix() == "docs/manual/how-to/install.md"
    assert failures == []


@pytest.mark.parametrize(
    "value",
    (
        r"docs\manual\how-to\install.md",
        "../docs/manual/index.md",
        "/docs/manual/index.md",
        "C:/docs/manual/index.md",
        "docs/./manual/index.md",
        "docs//manual/index.md",
        ".",
    ),
)
def test_noncanonical_or_escaping_inventory_path_is_rejected(value: str) -> None:
    failures: list[str] = []
    assert (
        _normalized_repo_path(
            value,
            prefix="inventory:2",
            field="destination",
            failures=failures,
        )
        is None
    )
    assert failures == [
        "inventory:2: destination must be normalized repository-relative POSIX"
    ]

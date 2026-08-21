from __future__ import annotations

import subprocess
import sys


def test_core_import_does_not_load_textual() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import prolog_tsetlin; "
                "assert not any(name == 'textual' or name.startswith('textual.') "
                "for name in sys.modules)"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr

"""Phase 0 smoke tests: the package imports and the CLI entry point runs."""

import subprocess
import sys

from typer.testing import CliRunner

import utriage
from utriage.cli import app


def test_package_imports() -> None:
    assert utriage.__version__


def test_cli_version() -> None:
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert utriage.__version__ in result.output


def test_module_help_runs() -> None:
    proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, "-m", "utriage", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "utriage" in proc.stdout.lower()

"""Tests for Phase 8 orchestration pipeline and execution chain."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from experiments.run_phase8 import (
    CONFIG_CAPABLE,
    EXPERIMENT_SCRIPTS,
    QUICK_CAPABLE,
    Phase8Config,
    parse_args,
    run_phase8,
)

EXPECTED_STAGES: tuple[str, ...] = (
    "experiments/run_leakage_audit.py",
    "experiments/run_loco.py",
    "experiments/run_anomaly.py",
    "experiments/run_sim_generalization.py",
    "experiments/run_cross_dataset.py",
    "experiments/run_hybrid.py",
    "experiments/run_physics_model.py",
    "experiments/run_false_call_analysis.py",
    "experiments/run_fit_diagnostics.py",
    "experiments/run_failures.py",
    "experiments/run_hardware.py",
    "scripts/make_data_inventory.py",
)


def _setup_mock_run(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Configure mocked subprocess and matrix helpers."""
    mock_run = MagicMock()
    monkeypatch.setattr("subprocess.run", mock_run)
    monkeypatch.setattr("experiments.run_phase8.build_matrix", lambda _: {})
    monkeypatch.setattr("experiments.run_phase8.plot_matrix", lambda *_: None)
    return mock_run


def _get_called_commands(mock_run: MagicMock) -> list[list[str]]:
    """Extract command arguments list from subprocess mock calls."""
    return [call.args[0] for call in mock_run.call_args_list if call.args]


def _get_called_scripts(mock_run: MagicMock) -> list[str]:
    """Extract script filenames from command arguments list."""
    cmds = _get_called_commands(mock_run)
    return [cmd[1].replace("\\", "/") for cmd in cmds if len(cmd) > 1]


def test_experiment_scripts_constant_order() -> None:
    """Verify EXPERIMENT_SCRIPTS contains new stages in required order."""
    assert EXPERIMENT_SCRIPTS == EXPECTED_STAGES
    c_idx = EXPERIMENT_SCRIPTS.index("experiments/run_cross_dataset.py")
    h_idx = EXPERIMENT_SCRIPTS.index("experiments/run_hybrid.py")
    p_idx = EXPERIMENT_SCRIPTS.index("experiments/run_physics_model.py")
    f_idx = EXPERIMENT_SCRIPTS.index("experiments/run_false_call_analysis.py")
    d_idx = EXPERIMENT_SCRIPTS.index("experiments/run_fit_diagnostics.py")
    assert c_idx < h_idx < p_idx < f_idx < d_idx


def test_stage_order_when_all_exist(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Ensure run_phase8 executes all stages in exact sequence when files exist."""
    mock_run = _setup_mock_run(monkeypatch)
    monkeypatch.setattr(
        "experiments.run_phase8.Path.is_file",
        lambda self: True,
    )

    cfg = Phase8Config(results_dir=tmp_path)
    run_phase8(cfg)

    executed = _get_called_scripts(mock_run)
    assert executed[: len(EXPECTED_STAGES)] == list(EXPECTED_STAGES)


def test_missing_script_skipped_with_note(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Missing script like run_false_call_analysis is skipped with printed note."""
    mock_run = _setup_mock_run(monkeypatch)
    cfg = Phase8Config(results_dir=tmp_path)
    run_phase8(cfg)

    executed = _get_called_scripts(mock_run)
    assert "experiments/run_false_call_analysis.py" not in executed

    captured = capsys.readouterr().out
    assert "Skipping experiments/run_false_call_analysis.py" in captured
    assert "does not exist" in captured


@pytest.mark.parametrize(
    ("script_name", "expect_quick"),
    [
        ("experiments/run_leakage_audit.py", False),
        ("experiments/run_loco.py", True),
        ("experiments/run_anomaly.py", True),
        ("experiments/run_sim_generalization.py", True),
        ("experiments/run_cross_dataset.py", True),
        ("experiments/run_hybrid.py", True),
        ("experiments/run_physics_model.py", True),
        ("experiments/run_false_call_analysis.py", True),
        ("experiments/run_fit_diagnostics.py", True),
        ("experiments/run_failures.py", True),
        ("experiments/run_hardware.py", False),
        ("scripts/make_data_inventory.py", False),
    ],
)
def test_quick_flag_routing(script_name: str, expect_quick: bool) -> None:
    """Verify quick capability configuration matches experiment scripts."""
    base_name = Path(script_name).name
    assert (base_name in QUICK_CAPABLE) is expect_quick


def _check_script_config_flags(
    cmd: list[str],
    is_capable: bool,
    cfg_val: str,
    set_vals: tuple[str, ...],
) -> None:
    """Assert presence or absence of config and set flags in command."""
    if is_capable:
        assert "--config" in cmd
        assert cmd[cmd.index("--config") + 1] == cfg_val
        for s in set_vals:
            assert s in cmd
    else:
        assert "--config" not in cmd
        assert "--set" not in cmd


@pytest.mark.parametrize("script", EXPECTED_STAGES)
def test_config_and_set_passthrough(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    script: str,
) -> None:
    """Check pass-through of --config and --set to capable scripts only."""
    mock_run = _setup_mock_run(monkeypatch)
    monkeypatch.setattr("experiments.run_phase8.Path.is_file", lambda self: True)

    cfg_file = Path("test_detector.yaml")
    overrides = ("detector.threshold=0.6", "physics.margin=1.2")
    cfg = Phase8Config(
        results_dir=tmp_path,
        config=cfg_file,
        set=overrides,
    )
    run_phase8(cfg)

    cmds = {c[1].replace("\\", "/"): c for c in _get_called_commands(mock_run)}
    is_capable = Path(script).name in CONFIG_CAPABLE
    _check_script_config_flags(cmds[script], is_capable, str(cfg_file), overrides)


def test_assemble_only_skips_experiments(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Assemble-only mode executes no experiment generation scripts."""
    mock_run = _setup_mock_run(monkeypatch)
    cfg = Phase8Config(assemble_only=True, results_dir=tmp_path)
    run_phase8(cfg)

    cmds = _get_called_commands(mock_run)
    assert not any("experiments/" in c[1].replace("\\", "/") for c in cmds)


def test_skip_flag_bypasses_scripts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Explicit skip entries are excluded from execution."""
    mock_run = _setup_mock_run(monkeypatch)
    monkeypatch.setattr("experiments.run_phase8.Path.is_file", lambda self: True)
    cfg = Phase8Config(
        results_dir=tmp_path,
        skip=("run_hybrid.py", "run_loco.py"),
    )
    run_phase8(cfg)

    executed = _get_called_scripts(mock_run)
    assert "experiments/run_hybrid.py" not in executed
    assert "experiments/run_loco.py" not in executed


def test_parse_args_options() -> None:
    """CLI argument parser handles config, repeatable set, and standard flags."""
    args = [
        "--config",
        "custom_detectors.yaml",
        "--set",
        "key1=val1",
        "--set",
        "key2=val2",
        "--quick",
        "--assemble-only",
        "--skip",
        "run_anomaly.py",
    ]
    cfg = parse_args(args)
    assert cfg.config == Path("custom_detectors.yaml")
    assert cfg.set == ("key1=val1", "key2=val2")
    assert cfg.quick is True
    assert cfg.assemble_only is True
    assert "run_anomaly.py" in cfg.skip

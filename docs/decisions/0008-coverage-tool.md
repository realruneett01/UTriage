# 0008. Dev dependency: pytest-cov for test coverage

- Status: Accepted
- Date: 2026-10-27

## Context

Phase 7.6 of the project plan requires maintaining at least 80% test coverage across `src/` as a quality gate.
To prevent regressions and verify compliance in developer environments and continuous integration (CI), coverage measurement must be automated and executable in a single command alongside the existing test suite.

`CONTRIBUTING.md` requires that any dependency outside the core list requires an Architecture Decision Record (ADR). Adding `pytest-cov` as a developer-only dependency introduces this tooling.

## Decision

1. **Add `pytest-cov` as a dev dependency**:
   - Add `pytest-cov` under `[project.optional-dependencies]` in the `dev` extra of `pyproject.toml`.
   - `pytest-cov` is licensed under the MIT license, compatible with the project license.
   - It is strictly a development and CI dependency; it is never required for production, plant evaluation, or offline release bundles.

2. **Configuration in `pyproject.toml`**:
   - `[tool.coverage.run]`: target `source = ["src/utriage"]` and omit `["*/__main__.py"]`.
   - `[tool.coverage.report]`: enforce `fail_under = 80` and `show_missing = true`.

3. **CI integration**:
   - Update `.github/workflows/ci.yml` test step to execute `pytest --cov --cov-report=term`.
   - CI runs will fail if test coverage drops below 80%.

## Alternatives Considered

- **Using `coverage.py` directly (`coverage run -m pytest` and `coverage report`)**:
  `coverage.py` is the underlying engine used by `pytest-cov`. While functional without a pytest plugin, it requires multiple commands, separate configuration handling, and more complex command invocations in CI workflows and local testing. `pytest-cov` integrates directly with the pytest runner, allowing coverage collection and terminal reporting in a single invocation (`pytest --cov --cov-report=term`) while honoring settings in `pyproject.toml`.

## Consequences

- Automated verification of the project plan 7.6 coverage requirement (>= 80% on `src/utriage`).
- CI workflows fail automatically if coverage drops below the 80% threshold.
- No impact on production or runtime environments, maintaining offline standalone deployability.

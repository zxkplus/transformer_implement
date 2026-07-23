# Repository Guidelines

## Project Structure & Module Organization

```
transformer_implement/
  main.py              # Core Transformer implementation (single-file, flat layout)
  test_attention.py    # Pytest-based tests for each built component
  .vscode/settings.json  # VS Code: conda env manager, pytest runner
```

Source code lives in `main.py`. As components grow, move each module (attention, embeddings, encoder, etc.) into a `src/` package with `__init__.py` re-exports. Tests mirror the module under test in name and scope.

## Build, Test, and Development Commands

| Command | Purpose |
|---|---|
| `pytest` | Run all tests in the project root |
| `pytest test_attention.py -v` | Run attention tests with verbose output |
| `python main.py` | Smoke-test the current implementation (add manual checks here) |

**Conda environment**: `pytorch_env_v1` (Python 3.12.4, PyTorch 2.6.0+cu124). Activate before running code:

```bash
conda activate pytorch_env_v1
```

## Coding Style & Naming Conventions

- **Indentation**: 4 spaces. No tabs.
- **Naming**: `snake_case` for functions and variables (`scaled_dot_product_attention`), `PascalCase` for `nn.Module` subclasses (`MultiHeadAttention`).
- **Imports**: Standard library first, then third-party (`torch`, `torch.nn`), then local. Keep imports at the top of each file.
- **Type hints**: Not required but encouraged for function signatures as the codebase matures.
- No formatter or linter configured yet; keep code manually consistent with existing style.

## Testing Guidelines

- **Framework**: pytest (configured in `.vscode/settings.json`)
- **Coverage**: One test file per major component. Add new tests when a new module is completed.
- **Naming**: `test_<module>.py` for test files, `test_<NN>_<description>` for functions.
- **Test style**: Prefer self-checking assertions that report expected vs. actual values clearly. Each test prints a descriptive label before asserting.
- Run `pytest` before committing. All tests must pass.

## Commit & Pull Request Guidelines

- **Commit format**: [Conventional Commits](https://www.conventionalcommits.org/) — `type: description`
  - `feat:` for new components (e.g., `feat: add multi-head attention`)
  - `test:` for test additions and improvements
  - `docs:` for README and documentation updates
  - `refactor:` for restructuring without behavioral changes
- **Scope**: Keep commits focused on one component at a time. One commit per completed learning module.
- **No PR template** yet; reference the README roadmap in commit descriptions when helpful.

## Agent-Specific Instructions

- This is a **teaching repository**. Do not write code for the student unless explicitly asked. Provide hints, review their code, and suggest next steps.
- The learning roadmap lives in [README.md](README.md). Consult it to track progress and determine what component comes next.
- After each component is implemented by the student, run `pytest` to verify correctness before moving on.
- When offering a code review, ground feedback in concrete line references and frame suggestions as questions or nudges rather than corrections.

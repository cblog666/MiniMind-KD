# Contributing

Keep changes small, testable and explicit about research provenance. New architecture claims must cite a primary source and state whether the code is exact, scaled or approximate.

Before opening a pull request:

```bash
python scripts/scan_secrets.py
ruff check .
pytest
```

Do not add hosted API keys, credential loaders, private datasets, unlicensed checkpoints, or unsandboxed execution of model-generated programs.

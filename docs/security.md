# Security and privacy

## Repository policy

- No hosted-model SDK or API credential is required.
- Tokenizers, datasets, checkpoints and teacher models are supplied as local paths.
- `.env`, common key formats, datasets, checkpoints, weights and experiment logs are ignored.
- `scripts/scan_secrets.py` runs locally and in CI before tests.
- Checkpoints are written and loaded only as `safetensors`; there is no pickle-format fallback.
- The built-in code-domain reward never executes generated code. Add execution rewards only inside a separately secured sandbox.

## Before publishing

Run:

```bash
python scripts/scan_secrets.py
git status --short
git diff --cached
```

Never commit raw private conversations, proprietary training data, local absolute paths, access tokens, private keys, model-provider request logs, or checkpoints whose license does not permit redistribution.

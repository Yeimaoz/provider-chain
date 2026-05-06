# Contributing

Thanks for your interest!

## Development setup

```bash
git clone https://github.com/Yeimaoz/provider-chain.git
cd provider-chain
python -m venv venv
source venv/bin/activate
pip install -e '.[dev]'
```

## Running tests

```bash
pytest -v
```

## PR process

1. Fork + branch off `main`.
2. Add tests covering your change.
3. Run `pytest` locally to confirm green.
4. Open PR with description of motivation + scope.

## Adding a new provider

See `chain.py` `_chat_once` and `PROVIDER_ENDPOINTS`. PRs welcome.

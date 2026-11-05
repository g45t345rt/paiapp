# Contributing to Pai

Thanks for your interest in contributing.

## Development setup

1. Install Python 3.12–3.14.
2. Run `run.bat` (Windows) or `sh run.sh` (Linux/macOS) once to create the private `.venv` and install the dependencies.
3. Launch the app with the same command; it reuses the environment.

Use `--core-only` if you are working on the paint UI and do not need the AI engine.

## Tests

Tests are plain scripts (no test runner needed). Run one directly with the project interpreter, for example:

```
.venv\Scripts\python.exe tests\test_blend_modes.py
```

Each script prints `ok`/`FAIL` lines and sets a non-zero exit code on failure.

## Code layout

- `paintapp/` – the Tkinter UI and painting logic
- `engine/` – the local AI generation, upscaling and background removal engine
- `installer/` – packaging and installers
- `tests/` – test scripts

## Pull requests

Keep changes focused, run the relevant tests, and describe what you changed and why. Note that Pai is GPL-3.0; by contributing you agree your work is licensed the same way.

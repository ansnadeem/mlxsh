# Contributing

Bug reports and patches are welcome. mlxsh is one file, `mlxsh.py`, with tests
in `tests/`.

## Getting set up

```sh
git clone https://github.com/ansnadeem/mlxsh
cd mlxsh
python3 -m venv .venv
.venv/bin/pip install mlx-lm mlx-vlm huggingface_hub
./mlxsh.py shell
```

## Before opening a pull request

```sh
python3 -m unittest discover -s tests   # no network, no models needed
ruff check .
python3 -m py_compile mlxsh.py
```

CI runs the same on macOS with Python 3.10, 3.12 and 3.13, plus a build of the
wheel.

## House style

- Standard library only in `mlxsh.py`. `mlx_lm`, `mlx_vlm` and
  `huggingface_hub` are imported lazily inside the functions that need them, so
  the tool still runs when they are missing.
- Plain output: no em dashes, no decorative glyphs, no chatty asides. A test
  enforces the characters.
- Comments say what a line does when that is not obvious, or say nothing.
- Anything a user might want different belongs in `SETTINGS`, not in a
  constant.
- New behaviour needs a test. The tests never touch the network, download a
  model, or need a terminal.

## Reporting a bug

Include the output of `mlxsh doctor`, the command you ran, and what happened.
`MLXSH_DEBUG=1` turns an unexpected error into a traceback.

## Recording the demo

The README animation is made with [vhs](https://github.com/charmbracelet/vhs):

```sh
brew install vhs
vhs demo.tape          # writes demo.gif
```

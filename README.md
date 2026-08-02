# mlxsh

[![CI](https://github.com/ansnadeem/mlxsh/actions/workflows/ci.yml/badge.svg)](https://github.com/ansnadeem/mlxsh/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

Run local [MLX](https://github.com/ml-explore/mlx) models on Apple silicon.
mlxsh serves a model at an OpenAI-compatible endpoint, keeps several loaded at
once, switches a model between text-only and multimodal, and browses and
downloads from the Hugging Face Hub. One file, standard library only; it drives
`mlx-lm` and `mlx-vlm`.

```
mlxsh> lm gemma-4-26B             serve it at http://127.0.0.1:41277/v1
mlxsh> lm qwen3.6                 a second model, the first keeps running
mlxsh> ask --on qwen "hello"      no extra copy is loaded
mlxsh> stop all
```

macOS on Apple silicon, Python 3.10 or newer. Driving mlxsh from a script or a
coding agent: [AGENTS.md](AGENTS.md).

## Install

```sh
curl -LsSf https://raw.githubusercontent.com/ansnadeem/mlxsh/main/install.sh | sh
mlxsh setup
```

The first line installs [uv](https://github.com/astral-sh/uv) if it is missing,
then mlxsh with both engines. The second is only needed if something is
missing: it installs `mlx-lm`, `mlx-vlm` and `huggingface_hub` into whichever
environment mlxsh runs from, about 300 MB. `mlxsh doctor` shows what it found.

State lives in `~/.mlxsh/`: the registry, one file per running server, logs and
history. Override with `MLXSH_HOME`.

## The two modes

| mode | engine | what loads |
|---|---|---|
| `lm` | `mlx_lm` | text weights only, no vision tower |
| `vision` | `mlx_vlm` | the full multimodal stack |

Most MLX repos are vision-capable, so the mode picks the engine, not the model.
Loading a model that is already running in the other mode swaps the engine on
the same port.

## Several models at once

One model per port. When the configured port is busy the next model takes the
next free port, so nothing running is stopped by accident.

```
mlxsh> status
  lm      gemma-4-26B-A4B-it-qat-4bit    15.0 GB   up 01:15   http://127.0.0.1:41277/v1
  vision  gemma-4-31b-it-4bit            18.4 GB   up 00:59   http://127.0.0.1:41278/v1
  2 servers, 33.4 GB resident now, about 33.4 GB of weights, machine has 68.7 GB
```

`stop`, `bench`, `ask`, `chat` and `log` take a port, a model substring or a
mode: `stop 41278`, `bench gemma`, `ask --on vision "..."`, `stop all`. Leave
the argument off in the shell and you get a picker. `--new` forces a new port,
`--replace` reuses the configured one, and `config when_busy` sets the default.

Before adding a model mlxsh adds up the weights already loaded and asks if the
new one will not fit. `-y` skips the question.

## Using the endpoint

Any OpenAI client works. There is no API key.

```sh
curl -s http://127.0.0.1:41277/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"default_model","messages":[{"role":"user","content":"hello"}]}'
```

`default_model` means whatever that port was started with, so a client pinned
to a port needs no repo id. An exact repo id works too, and `mlxsh status
--json` reports it.

`GET /v1/models` on a port lists that one model. mlx_lm builds that list by
scanning the Hugging Face cache, so a server started against the real cache
advertises every model on the machine and clients cannot tell which is loaded;
mlxsh gives each server a cache view holding only its own model. That also
stops a stray request from swapping the model out or downloading another one.
`config pin_model off` restores mlx_lm's behaviour.

The servers have no authentication and bind to `127.0.0.1`;
`config host 0.0.0.0` exposes them to your network.

## Commands

```
mlxsh                    this list
mlxsh shell              the list, then the interactive shell
mlxsh <command> ...      run one command and exit

lm [model]               serve text only
vision [model]           serve multimodal
stop [port|model|all]    stop a server and free the memory
status [--json]          what is running
bench [port|model] [n]   measure tok/s, saved to the registry
log [port|model] [n]     tail one server's log

models                   pick a model, then pick an action
ls [--json]              the same list as plain text
use [model] [mode]       set the default for lm or vision
rm [model] [-y]          delete from disk

browse [filters]         live list from hugging face, space marks downloads
get [n|repo] [-y]        download from that list, or by repo id

ask <text>               single prompt, --image PATH sends a picture
chat                     a chat loop
--on <port|model>        which running model answers
--load                   load a private copy instead of using a server

config                   settings, and where each value comes from
doctor                   versions, paths, machine
setup                    install the MLX packages
edit                     open the registry in $EDITOR
help                     this list

--port N, --host ADDR    override the address for one run
--foreground             run a server attached to this terminal
```

`model` is a repo id, a number from `ls`, or a unique substring: `lm 3`,
`lm qwen3.6`, `vision gemma-4-31b`. A command that prints `error:` exits
non-zero.

`mlxsh <command> -h` explains one command with examples. Errors go to stderr
and exit non-zero, so a command composes in a pipeline.

Every command works both one-shot and in the shell. The shell adds what only
makes sense while you are sitting in it: a picker whenever you leave an
argument off, a command list on an empty line, tab completion, history, a chat
loop, and a line pinned to the top showing what is loaded (`config status_bar
off` hides it). One-shot runs never open a picker, so scripts stay
predictable.

In a picker: up/down or j/k move, type to filter, enter selects, space marks
several, esc cancels.

`browse` estimates each repo's memory from its safetensors dtypes, so 4-bit
packing is counted correctly, and hides what will not fit unless you ask for
`all`. Filters: `vision`, `text`, `trending`, `popular`, `new`, `all`, any
other word searches, and `org/` scopes to an org.

## Settings

```sh
config                   list settings and their source
config port 8080         change one
config reset port        back to the default
```

Flag beats environment beats registry beats default.

| setting | env | default |
|---|---|---|
| `host` | `MLXSH_HOST` | `127.0.0.1` |
| `port` | `MLXSH_PORT` | `41277` |
| `org` | `MLXSH_ORG` | `mlx-community` |
| `mem_comfy` | `MLXSH_MEM_COMFY` | `0.7` |
| `mem_tight` | `MLXSH_MEM_TIGHT` | `0.85` |
| `browse_limit` | `MLXSH_BROWSE_LIMIT` | `25` |
| `start_timeout` | `MLXSH_START_TIMEOUT` | `900` |
| `reply_timeout` | `MLXSH_REPLY_TIMEOUT` | `600` |
| `when_busy` | `MLXSH_WHEN_BUSY` | `new` |
| `pin_model` | `MLXSH_PIN_MODEL` | `on` |
| `status_bar` | `MLXSH_STATUS_BAR` | `on` |
| `bar_interval` | `MLXSH_BAR_INTERVAL` | `2.0` |

Also read: `MLXSH_HOME`, `MLXSH_PYTHON`, `MLXSH_DEBUG`, `HF_TOKEN`, `NO_COLOR`.

## Troubleshooting

| symptom | what to do |
|---|---|
| `port 41277 is held by pid N, not an mlx server` | `config port <n>`, or stop that process |
| a model exits during startup | `log` shows the last lines: usually an unsupported architecture or memory |
| everything is slow with several models loaded | they compete for bandwidth, `stop` one |
| a thinking model answers nothing | it spent its budget reasoning, raise `config ask_args` |
| downloads are rate limited | set `HF_TOKEN` |

`doctor` is the first thing to include in a bug report. `MLXSH_DEBUG=1` turns
an unexpected error into a traceback.

## Uninstall

```sh
uv tool uninstall mlxsh
rm -rf ~/.mlxsh
```

Models live in the Hugging Face cache, not in `~/.mlxsh`.

## Tests

```sh
python3 -m unittest discover -s tests
```

No network, no models, no MLX packages needed.

## More

- [AGENTS.md](AGENTS.md) for scripts and coding agents
- [CHANGELOG.md](CHANGELOG.md), [CONTRIBUTING.md](CONTRIBUTING.md)
- MIT, see [LICENSE](LICENSE)

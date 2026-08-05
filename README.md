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

There are two ways to put several models behind a single URL, and they trade
memory against latency:

| | a gateway in front of several servers | one server, `pin_model off` |
|---|---|---|
| the client | one URL, models by id | one URL, models by id |
| resident | all of them | one |
| switching | free, they are all warm | a full reload, seconds to a minute |
| memory | the sum of them | one model |

The second uses mlx_lm's own behaviour, where the model named in a request is
loaded in place of the current one. Measured here at 2.7 seconds to swap a
12.6 GB model with a warm page cache.

Before adding a model mlxsh adds up the weights already loaded and asks if the
new one will not fit. `-y` skips the question.

## Using the endpoint

Any OpenAI client works. There is no API key.

```sh
curl -s http://127.0.0.1:41277/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"mlx-community/gemma-4-26B-A4B-it-qat-4bit",
       "messages":[{"role":"user","content":"hello"}]}'
```

The repo id comes from `mlxsh status --json` and works against both engines.
On a `lm` server you can send `default_model` instead, an alias for whatever
that port was started with; `mlx_vlm` rejects it, so `status --json` reports
the alias only where it applies.

`GET /v1/models` on a port lists that one model. mlx_lm builds that list by
scanning the Hugging Face cache, so a server started against the real cache
advertises every model on the machine and clients cannot tell which is loaded;
mlxsh gives each server a cache view holding only its own model. That also
stops a stray request from swapping the model out or downloading another one.
`config pin_model off` restores mlx_lm's behaviour.

The servers have no authentication and bind to `127.0.0.1`;
`config host 0.0.0.0` exposes them to your network.

## Reaching it from elsewhere

The model servers have no authentication, which is why they only listen on
`127.0.0.1`. To use them from another machine, put the gateway in front: one
endpoint, a bearer key, routed to whichever loaded model the request names.

```
your app  ->  cloudflare edge  ->  cloudflared  ->  gateway  ->  model servers
              TLS, your            started by      the key,     127.0.0.1:41277
              hostname             mlxsh           routing      127.0.0.1:41278
```

```sh
mlxsh gateway                        prints the URL and the key
mlxsh tunnel setup llm.example.com   once: login, create, route DNS
mlxsh tunnel                         start it, and the gateway if needed
```

Then anything that speaks the OpenAI API works, unchanged:

```sh
OPENAI_BASE_URL=https://llm.example.com/v1
OPENAI_API_KEY=mlxsh-...              from: mlxsh gateway key
```

`/v1/models` there lists every loaded model, and a request naming one is routed
to its port. Load another with `mlxsh lm` and it appears in the list, with no
change on the client side. `mlxsh gateway key --new` rotates the key.

Worth knowing before you expose anything:

- The key is the only lock. Anything holding it uses your GPU.
- The gateway has no TLS of its own. The tunnel provides it, which is why the
  key is safe in transit. Binding the gateway to your LAN instead needs
  `--expose`, and the key then crosses that network in cleartext.
- A named tunnel needs a domain in a Cloudflare account. Quick tunnels are not
  used: they hand out a new name each run and do not support server-sent
  events, so streaming would break.
- Cloudflare's proxy read timeout is 125 seconds and only Enterprise can raise
  it. A streamed reply is fine because bytes start immediately; a long
  **non-streaming** completion can return 524.
- Your Mac has to be awake. `caffeinate -s mlxsh tunnel` keeps it up while the
  tunnel runs.
- `cloudflared service install` makes the tunnel survive reboots, if you want
  that.

Prefer something else? `config tunnel_cmd "..."` replaces cloudflared entirely,
with `{port}` substituted: Tailscale Funnel for a permanent
`<mac>.<tailnet>.ts.net` with no domain to buy (macOS needs the open-source
build, not the App Store one), [chisel](https://github.com/jpillora/chisel) on
a box you already have, or `ssh -R` to your own server.

## Commands

```
mlxsh                    this list
mlxsh shell              the list, then the interactive shell
mlxsh <command> ...      run one command and exit

lm [model]               serve text only
vision [model]           serve multimodal
gateway [stop|key]       an authenticated endpoint for other machines
tunnel [setup <host>]    a permanent public address for the gateway
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
| `gateway_port` | `MLXSH_GATEWAY_PORT` | `41377` |
| `tunnel_name` | `MLXSH_TUNNEL_NAME` | `mlxsh` |
| `tunnel_hostname` | `MLXSH_TUNNEL_HOSTNAME` | empty |
| `tunnel_cmd` | `MLXSH_TUNNEL_CMD` | empty |
| `status_bar` | `MLXSH_STATUS_BAR` | `on` |
| `bar_interval` | `MLXSH_BAR_INTERVAL` | `2.0` |

Also read: `MLXSH_HOME`, `MLXSH_PYTHON`, `MLXSH_DEBUG`, `MLXSH_API_KEY`,
`HF_TOKEN`, `NO_COLOR`.

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

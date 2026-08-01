# mlxsh

[![PyPI](https://img.shields.io/pypi/v/mlxsh)](https://pypi.org/project/mlxsh/)
[![CI](https://github.com/ansnadeem/mlxsh/actions/workflows/ci.yml/badge.svg)](https://github.com/ansnadeem/mlxsh/actions/workflows/ci.yml)
[![Python](https://img.shields.io/pypi/pyversions/mlxsh)](https://pypi.org/project/mlxsh/)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

**Run local LLMs on your Mac, from a lightweight CLI.** mlxsh is a launcher and
shell for [MLX](https://github.com/ml-explore/mlx) models on Apple silicon:
pick a model, serve it at an OpenAI-compatible endpoint, and point any client
at it. Keep several models loaded at once, switch a model between text-only and
multimodal, and browse and download from the Hugging Face Hub without leaving
the terminal.

No app to install, no daemon, no model store of its own: one 90 KB script that
starts in a tenth of a second and gets out of the way.

```
 mlxsh   lm gemma-4-26B-A4B :41277 15.0G 04:12   [1 loaded, 15.0G]
mlxsh> lm qwen3.6                 load a second model, the first keeps running
mlxsh> ask --on qwen "hello"      talk to it, nothing extra is loaded
mlxsh> bench gemma                58.1 tok/s
mlxsh> stop all
```

One file, standard library only. It drives `mlx-lm` and `mlx-vlm` rather than
bundling an engine, so `mlxsh help` works before anything else is installed.

Requires macOS on Apple silicon and Python 3.10 or newer. MLX is a Python
stack, so Python is unavoidable, but you do not have to manage it: `uv tool
install mlxsh` fetches a suitable interpreter itself. The Python that ships
with macOS is 3.9 and too old for MLX.

**Quickstart**

```sh
uv tool install git+https://github.com/ansnadeem/mlxsh   # or see Install below
mlxsh browse            # pick a model that fits your machine, enter downloads it
mlxsh lm                # serve it at http://127.0.0.1:41277/v1
mlxsh ask "hello"       # or point any OpenAI client at that URL
mlxsh stop
```

Driving mlxsh from a script or a coding agent? See [AGENTS.md](AGENTS.md).

## Why this and not something else

[Ollama](https://ollama.com), LM Studio and llama.cpp run GGUF models through
their own engines. mlxsh runs MLX models through Apple's own stack, which is
usually faster on Apple silicon and is the only way to run the MLX-quantised
repos on `mlx-community`. Beyond the engine, the difference is weight:

| | mlxsh | a typical desktop LLM tool |
|---|---|---|
| what you install | one 90 KB script, no dependencies of its own | an app bundle or a service, hundreds of MB |
| running when idle | nothing | a daemon, waiting |
| where models live | the Hugging Face cache you already have | a private store, downloaded again |
| models loaded at once | as many as memory allows, one per port | usually one |
| startup | 55 ms to answer, 200 ms to report every server | app launch |
| what it is | a launcher: the servers are plain `mlx_lm` and `mlx_vlm` processes you could have started yourself | an engine plus a runtime plus a UI |
| interface | terminal, pickers when you want them, plain text when piped | GUI, or a client library |

The whole program is `mlxsh.py`: standard library only, no imports beyond it at
module level, and every heavier thing (`mlx-lm`, `mlx-vlm`, `huggingface_hub`)
imported lazily inside the function that needs it. That is why `mlxsh help`
works before anything else is installed, and why `curl`ing one file is a
complete install.

To be fair about it: the engines are not free. `mlxsh setup` installs about
300 MB of `mlx-lm`, `mlx-vlm` and `huggingface_hub`, which is what any MLX tool
needs to run a model, and the models themselves are tens of gigabytes. mlxsh is
the 90 KB on top, and it is the part you can read in an afternoon.

**Contents**: [Install](#install), [The two modes](#the-two-modes),
[Using the endpoint](#using-the-endpoint),
[More than one model](#more-than-one-model), [Commands](#commands),
[Settings](#settings), [Finding models](#finding-models),
[Troubleshooting](#troubleshooting)

## Install

Pick one. All of them end with `mlxsh` on your PATH.

```sh
# 1. the file itself: mlxsh is one script with no dependencies of its own
curl -LsSf https://raw.githubusercontent.com/ansnadeem/mlxsh/main/mlxsh.py \
  -o ~/.local/bin/mlxsh && chmod +x ~/.local/bin/mlxsh

# 2. one line, no prerequisites: installs uv if needed, then mlxsh
curl -LsSf https://raw.githubusercontent.com/ansnadeem/mlxsh/main/install.sh | sh

# 3. if you already have uv or pipx
uv tool install mlxsh          # or: pipx install mlxsh
uvx mlxsh status               # or run it once without installing

# 4. from a clone
git clone https://github.com/ansnadeem/mlxsh && cd mlxsh && ./mlxsh.py
```

Option 1 is the whole program: one readable file you can inspect before running
it, upgraded by curling it again over itself. If `curl | sh` makes you uneasy,
prefer it, or read `install.sh` first.

Then, unless you installed with uv or pipx (which bring the engines along):

```sh
mlxsh setup        # mlx-lm, mlx-vlm and huggingface_hub into ~/.mlxsh/.venv
mlxsh doctor       # confirms what it found
```

`setup` uses uv when it is there and falls back to `python -m venv`. It costs
about 300 MB and lives on its own, so you can upgrade or delete it without
touching mlxsh. mlxsh looks for that environment, a `.venv` next to the
script, or `$MLXSH_PYTHON`, and re-execs into whichever it finds; if the MLX
packages are already importable, none of that happens.

Vision mode is included by default. `mlxsh setup --no-vision` skips mlx-vlm,
and `uv tool install "mlxsh[vision]"` adds it to a uv install.

State lives in `~/.mlxsh/`: `models.json` (the registry), `servers/<port>.json`
(one per running server), `logs/<port>.log`, and shell history. Nothing is
written next to the script. Override the lot with `MLXSH_HOME`.

## The two modes

| mode | engine | what loads | use it when |
|---|---|---|---|
| `lm` | `mlx_lm` | text weights only, no vision tower | you don't need images: less memory, simpler path |
| `vision` | `mlx_vlm` | the full multimodal stack | you're sending images, video or audio |

Many MLX repos are vision-capable, so the same model can run in either mode:
the mode picks the engine, not the model. Loading a model that is already
running in the other mode swaps the engine on the same port and reports the
memory it freed, so `lm` and `vision` trade places cleanly.

Both modes serve the same address, so clients never need reconfiguring:

    http://127.0.0.1:41277/v1        from a container: http://host.docker.internal:41277/v1

## Using the endpoint

Anything that speaks the OpenAI API works. There is no API key; pass any
non-empty string if your client insists on one.

```sh
export OPENAI_BASE_URL=http://127.0.0.1:41277/v1
export OPENAI_API_KEY=not-needed
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:41277/v1", api_key="not-needed")
print(client.chat.completions.create(
    model="mlx-community/gemma-4-26B-A4B-it-qat-4bit",   # what `status` shows
    messages=[{"role": "user", "content": "hello"}],
).choices[0].message.content)
```

```sh
curl -s http://127.0.0.1:41277/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"mlx-community/gemma-4-26B-A4B-it-qat-4bit",
       "messages":[{"role":"user","content":"hello"}]}'
```

Streaming, images and multi-model routing are covered in [AGENTS.md](AGENTS.md).

## More than one model

A model per port, and **nothing running is ever stopped by accident**. Serving
a model when the configured port is busy moves it to the next free port.

```
mlxsh> lm gemma-4-26B                 41277
mlxsh> lm qwen3.6                     41278, the first one keeps running
mlxsh> vision gemma-4-31b             41279
mlxsh> status
  lm      gemma-4-26B-A4B-it-qat-4bit         15.0 GB   up 01:15     http://127.0.0.1:41277/v1
  lm      Qwen3.6-35B-A3B-4bit                20.4 GB   up 01:02     http://127.0.0.1:41278/v1
  vision  gemma-4-31b-it-4bit                  4.1 GB   up 00:59     http://127.0.0.1:41279/v1
  3 servers, 39.5 GB resident now, about 52.8 GB of weights, machine has 68.7 GB
```

`stop`, `bench` and the rest take a port, a model substring, or a mode:
`stop 41278`, `bench gemma-4-31b`, `stop vision`, `stop all`. In the shell,
leave the argument off and you get a picker.

### Which port a model lands on

| you type | what happens |
|---|---|
| `lm modelA` with the port free | serves on the configured port |
| `lm modelB` with the port busy | next free port, `when_busy` decides |
| `vision modelA` while `lm modelA` runs | swaps the engine in place, same port |
| `lm modelB --new` | always the next free port |
| `lm modelB --replace` | always the configured port, stopping what is there |
| `lm modelB --port 41300` | exactly that port, replacing an mlxsh server on it |

`when_busy` is a setting: `new` (default, never stops anything), `replace` (the
old behaviour of reusing the configured port) or `ask` (a picker each time, and
`new` when there is no terminal).

```
config when_busy replace
```

### Prompting a particular model

`ask` and `chat` talk to a model that is already loaded, over the endpoint, so
nothing extra is loaded and the reply is immediate. Name the model with `--on`,
or leave it off and pick from the list:

```
ask --on gemma "explain unified memory in one sentence"
ask --on 41278 --image shot.png "what is in this picture"
chat --on qwen3.6
```

Every server gets its own log too: `log --on 41278`, or `log` with a picker.

With one model loaded, `--on` is unnecessary. With none loaded, or with
`--load`, both commands fall back to loading a private copy the way they always
did. Images need a server in `vision` mode; against an `lm` server mlxsh says so
rather than failing obscurely.

The shell pins a line to the top of the terminal showing what is loaded, so the
prompt stays out of it:

```
 mlxsh   lm gemma-4-26B-A4B :41277 15.0G 04:12   vision gemma-4-31b :41279 18.4G 01:03   [2 loaded, 33.4G]
mlxsh> _
```

It refreshes every couple of seconds and disappears when you leave. Turn it off
with `config status_bar off`, or change the rate with `config bar_interval 5`.

Before adding a model, mlxsh adds up the weights already loaded and asks if the
new one will not fit:

```
  3 other model(s) hold about 52.8 GB, Qwen3.6-35B-A3B-4bit needs 20.4 GB more,
  this machine has 68.7 GB
  start it anyway? [y/N]
```

Pass `-y` to skip the question. Note that resident size is not the whole story:
macOS pages idle weights out, so a loaded but unused model can report almost no
RSS while still occupying its share when it wakes. The check uses the weight
size on disk, which is the honest number.

## Interactive by default

Leave an argument off and you get an arrow-key picker. Press enter on an empty
line for the command list.

```
mlxsh> models
  models                                       up/down select   enter ok   esc cancel
 >    vision   20.4 GB  Qwen3.6 35B-A3B MoE (4-bit)    61.4 lm  89.1 vision tok/s
      text      0.4 GB  Qwen3-0.6B-4bit                187.2 lm tok/s
```

Up/down or j/k move, PgUp/PgDn/Home/End jump, typing filters, backspace
un-filters, enter selects, space marks several (in `browse` and `rm`), esc
clears the filter or cancels. The picker draws below the prompt and erases
itself, so your scrollback survives.

Every command also works as a one-shot: `mlxsh lm qwen3.6`, `mlxsh stop`. The
three browsing commands (`browse`, `models`, `get`) are interactive by nature,
so from a terminal they open the picker and then leave you at the prompt.
Piping anywhere (`mlxsh browse vision | grep`) gives plain text, so scripts
stay predictable.

## Commands

```
mlxsh                    the command list
mlxsh shell              the list, then the interactive shell
mlxsh <command> ...      run one command and exit

lm [model]               serve text only (mlx_lm)
vision [model]           serve multimodal (mlx_vlm)
stop [port|model|all]    stop a server and free the memory
status [--json]          every server that is running
bench [port|model] [n]   measure tok/s, saved to the registry
log [port|model] [n]     tail one server's log

--new                    force the next free port
--replace                force the configured port, stopping what is on it
--port N, --host ADDR    override for one run
--foreground             run attached to this terminal
-y                       skip the memory warning when adding a server

models                   pick a model, then pick an action
ls [--json]              the same list as plain text
use [model] [mode]       set the default for lm or vision
rm [model] [-y]          delete from disk
edit                     open the registry in $EDITOR

browse [filters]         live list from hugging face, space marks downloads
get [n|repo] [-y]        download from that list, or by repo id

ask <text>               single prompt, --image PATH sends a picture
chat                     a chat loop
--on <port|model>        which running model answers
--load                   load a private copy instead of using a server

config                   settings, and where each value comes from
doctor                   versions, paths, machine
setup [--no-vision]      install the MLX packages into ~/.mlxsh/.venv
```

`status --json` and `ls --json` print stable, parseable output with full repo
ids, ports and endpoints. Scripts should use those rather than reading the
human listing; see [AGENTS.md](AGENTS.md).

`model` can be a repo id, a number from `ls`, or any unique substring:
`lm 3`, `lm qwen3.6`, `vision gemma-4-31b`. A command that prints `error:`
exits non-zero, so `mlxsh` composes in scripts.

State lives in `~/.mlxsh/servers/<port>.json`, one file per running server.
Servers started outside mlxsh, or left behind by a crash, are picked up on the
configured port and shown as such.

## Settings

Everything is configurable from the shell. Nothing needs a file edited by hand.

```
config                   list settings and their source
config port 8080         change the port
config host 0.0.0.0      expose the server to the network
config org lmstudio-community
config reset port        back to the default
```

In the shell, `config` with no argument opens a picker of settings and prompts
for the new value. For a single run, pass flags instead: `mlxsh lm --port 8080`,
`mlxsh vision --host 0.0.0.0`.

Precedence is flag, then environment, then registry, then default, and `config`
shows which one is in force.

| setting | env | default | meaning |
|---|---|---|---|
| `host` | `MLXSH_HOST` | `127.0.0.1` | address the server binds to |
| `port` | `MLXSH_PORT` | `41277` | port the server listens on |
| `org` | `MLXSH_ORG` | `mlx-community` | hub org searched by browse, assumed by pull |
| `mem_comfy` | `MLXSH_MEM_COMFY` | `0.7` | fraction of RAM a model can use freely |
| `mem_tight` | `MLXSH_MEM_TIGHT` | `0.85` | fraction above which browse hides a model |
| `browse_limit` | `MLXSH_BROWSE_LIMIT` | `25` | rows per browse |
| `start_timeout` | `MLXSH_START_TIMEOUT` | `900` | seconds to wait for a server to load |
| `reply_timeout` | `MLXSH_REPLY_TIMEOUT` | `600` | seconds to wait for a reply from a server |
| `when_busy` | `MLXSH_WHEN_BUSY` | `new` | port already has a model: `new`, `replace` or `ask` |
| `status_bar` | `MLXSH_STATUS_BAR` | `on` | pin a live line at the top of the shell |
| `bar_interval` | `MLXSH_BAR_INTERVAL` | `2.0` | seconds between bar refreshes |

Also read from the environment: `MLXSH_HOME` (state directory), `MLXSH_PYTHON`
(interpreter to re-exec into), `MLXSH_DEBUG` (turn an unexpected error into a
traceback), `HF_TOKEN` (faster Hub downloads), `NO_COLOR`.

## Finding models

`browse` pulls a live list from the Hub in one request and opens a scrollable
picker showing the estimated memory footprint, whether the repo is
vision-capable, downloads, and last update. Models already on disk are marked
`downloaded`. Space marks several, enter downloads them, and you stay at the
prompt afterwards. Repos too big for your machine, and non-LLM repos (speech,
embeddings, image generation), are hidden unless you ask for `all`.

```
browse                   trending MLX models that fit this machine
browse vision            multimodal only
browse text popular      text-only, by downloads
browse new               most recently updated
browse qwen3.6 vision    search terms plus a filter
browse lmstudio-community/gemma    scope to another org
browse all               include models too big to run
get 3                    download row 3
```

Sizes come from the repo's safetensors dtype census, so MLX's 4-bit packing
(weights stored in `U32` words) is counted correctly. A naive parameter count
is off by 8x. Checked against five local models, the estimate matched what
landed on disk to within 0.1 GB.

Downloads are registered automatically, with the vision flag read from
`config.json`. Models already in your Hugging Face cache are picked up on the
next run, as long as the installed MLX packages implement their architecture.

## Numbers from one machine (M5 Pro, 64 GB)

`bench`, 128 tokens, short prompt, mlx-lm 0.31.3 and mlx-vlm 0.6.3:

| model | `lm` | `vision` |
|---|---|---|
| Qwen3.6 35B-A3B (MoE) | 61 tok/s | 89 tok/s |
| Gemma 4 26B-A4B (MoE) | 58 tok/s | 73 tok/s |
| gpt-oss 20B | 66 tok/s | text only |
| Gemma 4 31B (dense) | 13 tok/s | 14 tok/s |

Two things worth knowing:

1. Speed comes from the model, not the mode. MoE models (A3B, A4B) run 4 to 5
   times faster than a dense 31B in either mode.
2. `mlx_vlm` was the faster text decoder on MoE models here. Choose `lm` mode
   for the memory saving and the simpler path, not for throughput.

`bench` writes its result into the registry per model per mode, so `ls` and the
pickers show your numbers rather than these. Re-run it after upgrading mlx-lm
or mlx-vlm.

## What it actually runs

mlxsh is a driver, not an inference engine. Serving a model is:

```
python -m mlx_lm server --model <repo> --host 127.0.0.1 --port 41277 --log-level INFO
python -m mlx_vlm.server --model <repo> --host 127.0.0.1 --port 41277 --log-level INFO
```

started in its own process group, with output going to `~/.mlxsh/logs/<port>.log`
and a record in `~/.mlxsh/servers/<port>.json`. `stop` signals that process
group; nothing is killed unless its command line shows it is an mlx server.
Downloads shell out to `hf download`, or `huggingface_hub.snapshot_download`
when that is missing. Everything else is the standard library.

## Security

The servers have no authentication. The default bind address is `127.0.0.1`, so
they are reachable only from this machine. `config host 0.0.0.0` exposes a model
to your whole network, including anyone who can reach the port, and mlx-lm's own
server warns it is not built for production. Put it behind something that does
authentication if you need it off-machine.

Downloaded models run code from the Hub only to the extent that mlx-lm and
mlx-vlm do; mlxsh never passes `--trust-remote-code`.

## Troubleshooting

| symptom | what to do |
|---|---|
| `port 41277 is held by pid N, not an mlx server` | something else has the port; `config port <n>` or stop that process |
| a model exits during startup | `log` shows the last lines; usually an unsupported architecture or not enough memory |
| `no MLX implementation for <family>` | the installed mlx-lm/mlx-vlm cannot run that repo; look for a different quantisation |
| everything is slow with several models loaded | they compete for memory bandwidth; `status` shows the weight total, `stop` one |
| the terminal keeps a stuck top line | `config status_bar off`, then open a new shell |
| downloads are rate limited | set `HF_TOKEN` |
| a thinking model sits quiet, then says it spent its budget reasoning | raise the limit: `config ask_args` / `chat_args`, or use a model without a long chain of thought |

`doctor` prints versions, paths, memory limits and the endpoint, and is the
first thing to include in a bug report.

## Uninstall

```sh
uv tool uninstall mlxsh          # or pipx uninstall mlxsh
rm -rf ~/.mlxsh                  # registry, logs, server state, history
```

Downloaded models live in the Hugging Face cache, not in `~/.mlxsh`. Remove
them with `rm` inside mlxsh, or `hf cache delete`.

## Tests

```sh
python3 -m unittest discover -s tests -v
```

No network, no models, no MLX packages required. They cover the memory
estimator, vision sniffing, cache detection, model resolution, browse filtering
against a fake Hub, settings precedence, server identity checks, log tailing,
and the terminal escape-sequence parser.

## License

MIT, see [LICENSE](LICENSE).

## More

- [AGENTS.md](AGENTS.md) for driving mlxsh from a script or a coding agent
- [CHANGELOG.md](CHANGELOG.md) for what changed
- [CONTRIBUTING.md](CONTRIBUTING.md) if you want to send a patch

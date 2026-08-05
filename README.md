# mlxsh

[![CI](https://github.com/ansnadeem/mlxsh/actions/workflows/ci.yml/badge.svg)](https://github.com/ansnadeem/mlxsh/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

Run local [MLX](https://github.com/ml-explore/mlx) models on Apple silicon, from
one CLI. Find and download models from Hugging Face, serve them at an
OpenAI-compatible endpoint, keep several loaded at once, and put your own model
on the internet behind an API key when something off the machine needs it.

![a model served locally, then reached over the internet](demo.gif)

macOS on Apple silicon, Python 3.10 or newer.

## Install

```sh
curl -LsSf https://raw.githubusercontent.com/ansnadeem/mlxsh/main/install.sh | sh
```

That installs [uv](https://github.com/astral-sh/uv) if it is missing, then mlxsh
with both engines, `mlx-lm` and `mlx-vlm`. Run `mlxsh doctor` if anything looks
wrong, and `mlxsh setup` to install what it reports missing.

## Find a model

```sh
mlxsh browse trending          live list from hugging face, marks what you have
mlxsh get 3                    download number 3 from that list
mlxsh ls                       what is on this machine
```

`browse` reads each repo's real memory footprint from its safetensors, so 4-bit
models are counted correctly, and hides what will not fit in your RAM unless you
ask for `all`. Other filters: `vision`, `text`, `popular`, `new`, an org with
`mlx-community/`, or any word to search.

## Serve it

```sh
mlxsh lm gemma-4-26B           text only, at http://127.0.0.1:41277/v1
mlxsh vision gemma-4-31b       the full multimodal stack
mlxsh status                   what is running, and what it costs in memory
mlxsh stop all
```

A model is a repo id, a number from `ls`, or any unique substring, so `lm 3` and
`lm qwen3.6` both work. The mode picks the engine rather than the model: most
MLX repos can do vision, and `lm` skips the vision tower to get the speed back.

One model per port. Start a second and it takes the next free port instead of
evicting the first, so nothing you are using disappears:

```
mlxsh> status
  lm      gemma-4-26B-A4B-it-qat-4bit    15.0 GB   up 01:15   http://127.0.0.1:41277/v1
  vision  gemma-4-31b-it-4bit            18.4 GB   up 00:59   http://127.0.0.1:41278/v1
  2 servers, 33.4 GB resident now, machine has 68.7 GB
```

Before loading another, mlxsh adds up what is already resident and asks if the
new one will not fit.

## Talk to it

Any OpenAI client works, and locally there is no key:

```sh
curl -s http://127.0.0.1:41277/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"mlx-community/gemma-4-26B-A4B-it-qat-4bit",
       "messages":[{"role":"user","content":"hello"}]}'
```

Or without leaving the terminal:

```sh
mlxsh ask "explain this error" --on qwen
mlxsh ask "what is in this?" --image shot.png
mlxsh chat
```

`--on` picks which running model answers, by port, mode or substring. Each
server's `/v1/models` lists only the model it is actually holding, so a client
cannot be surprised by which one answers.

## Expose it to the internet

The model servers have no authentication, which is why they only listen on
`127.0.0.1`. To let an app elsewhere use them, mlxsh puts a gateway in front:
one endpoint, one bearer key, routed to whichever loaded model the request
names.

```
your app  ->  cloudflare edge  ->  cloudflared  ->  gateway  ->  model servers
              TLS, your            started by      the key,     127.0.0.1:41277
              hostname             mlxsh           routing      127.0.0.1:41278
```

```sh
mlxsh tunnel --quick           a public address in seconds, no account needed
mlxsh tunnel setup llm.example.com && mlxsh tunnel    an address that never moves
```

Either one starts the gateway if it is not already up, then prints the URL and
the key. Both need
[cloudflared](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/),
which ships for macOS, Linux and Windows.

Point anything that speaks the OpenAI API at it, unchanged:

```sh
OPENAI_BASE_URL=https://llm.example.com/v1
OPENAI_API_KEY=mlxsh-...       from: mlxsh gateway key
```

`/v1/models` there lists every model you have loaded, and a request naming one
goes to its port. Load another and it shows up, with nothing to change on the
client. `mlxsh gateway key --new` rotates the key.

Worth knowing before you open it up:

- The key is the only lock. Anything holding it uses your GPU.
- A permanent address needs a domain in a Cloudflare account. `--quick` needs
  neither, but the address changes on every restart and Cloudflare does not
  support server-sent events there, so treat streaming as unreliable.
- Cloudflare's read timeout is 125 seconds. Streamed replies are fine; a long
  non-streaming completion can come back as a 524.
- Your Mac has to stay awake: `caffeinate -s mlxsh tunnel`.
- Prefer another tunnel? `mlxsh config tunnel_cmd "..."` replaces cloudflared,
  with `{port}` substituted, for Tailscale Funnel, `ssh -R`, or anything else.

## The shell

```sh
mlxsh                          the command list
mlxsh shell                    the list, then an interactive prompt
```

Everything works both ways. What the shell adds is what only makes sense while
you are sitting there: leave an argument off and you get a picker (up/down to
move, type to filter, space to mark several), plus tab completion, history, and
a line pinned to the top showing what is loaded. One-shot runs never open a
picker, so scripts stay predictable.

`mlxsh help` lists every command, and `mlxsh <command> -h` explains one with
examples.

## Settings

```sh
mlxsh config                   every setting, and where its value came from
mlxsh config port 8080         change one
```

A flag beats an environment variable, which beats the saved value. The ones
worth knowing: `port` (41277), `gateway_port` (41377), `host` (127.0.0.1),
`org` (mlx-community), and `when_busy`, which decides whether a second model
takes a new port or replaces the running one.

State lives in `~/.mlxsh/`, models in the Hugging Face cache. To remove it all:
`uv tool uninstall mlxsh && rm -rf ~/.mlxsh`.

## When something is wrong

| symptom | what to do |
|---|---|
| a model exits during startup | `mlxsh log` shows the last lines, usually memory or an unsupported architecture |
| `port 41277 is held by pid N` | `mlxsh config port <n>`, or stop that process |
| everything slows down with several models | they compete for memory bandwidth, `stop` one |
| a thinking model answers nothing | it spent its budget reasoning, raise `--max-tokens` in `ask_args` with `mlxsh edit` |
| downloads are rate limited | set `HF_TOKEN` |

`mlxsh doctor` is the first thing to put in a bug report, and `MLXSH_DEBUG=1`
turns an unexpected error into a traceback.

## More

- [AGENTS.md](AGENTS.md) for scripts and coding agents
- [CHANGELOG.md](CHANGELOG.md), [CONTRIBUTING.md](CONTRIBUTING.md)
- MIT, see [LICENSE](LICENSE)

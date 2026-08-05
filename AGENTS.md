# Using mlxsh from an agent

Instructions for automated callers: coding agents, scripts, CI. Humans want
[README.md](README.md).

mlxsh runs local MLX models on Apple silicon and serves them at an
OpenAI-compatible endpoint. An agent uses it for two things:

1. **Drive the CLI** to load, list and unload models.
2. **Call the endpoint** to get completions from a loaded model.

## Requirements

macOS on Apple silicon, Python 3.10 or newer. The stock `/usr/bin/python3` is
3.9 and cannot run MLX; `uv tool install mlxsh` supplies its own interpreter.
`mlxsh doctor` reports the interpreter in use and says when it is too old.

## The contract

- One-shot commands (`mlxsh <command>`) never open a picker and never block on
  a prompt, except the two confirmations below. Pickers only appear in
  `mlxsh shell`, which an agent should not start.
- Output is plain text. Colour is disabled automatically when stdout is not a
  terminal; `NO_COLOR=1` forces it off. Piped `ask` prints the answer and
  nothing else.
- **Exit code 0 means the command worked.** Any line starting with `error:`
  means it did not, and the exit code is 1.
- Two commands can ask a yes/no question: `get`/`pull` before downloading, and
  `lm`/`vision` when the machine is short on memory. Pass `-y` to answer yes,
  or feed `</dev/null` to answer no.
- Every command is idempotent to re-run. Serving a model that is already
  serving prints `already running` and changes nothing.

## Recipes

### Is anything loaded?

**Use `--json`. Do not parse the human output.**

```sh
mlxsh status --json
```

```json
[
  {
    "model": "mlx-community/gemma-4-26B-A4B-it-qat-4bit",
    "mode": "lm",
    "engine": "mlx_lm",
    "host": "127.0.0.1",
    "port": 41277,
    "pid": 30144,
    "endpoint": "http://127.0.0.1:41277/v1",
    "resident_bytes": 15641241228,
    "uptime": "04:12",
    "adopted": false
  }
]
```

An empty array means nothing is loaded, and the exit code is still 0. `model`
is the full repo id, which is exactly what the endpoint wants; the human
listing shows a shortened name that the endpoint will reject.

### What can I load?

```sh
mlxsh ls --json   # every registered model: repo, vision, downloaded, size, tok/s
mlxsh doctor      # versions, paths, memory limits, endpoint
```

```json
{
  "models": [
    {"repo": "mlx-community/gemma-4-26B-A4B-it-qat-4bit", "label": "Gemma 4 26B-A4B",
     "vision": true, "downloaded": true, "size_bytes": 15641241228,
     "speed": {"lm": 58.1, "vision": 73.4}}
  ],
  "defaults": {"lm": "mlx-community/gemma-4-26B-A4B-it-qat-4bit", "vision": ""},
  "endpoint": "http://127.0.0.1:41277/v1"
}
```

### Load a model

```sh
mlxsh lm gemma-4-26B          # text only, fastest
mlxsh vision gemma-4-31b      # multimodal
mlxsh lm qwen3.6 --new -y     # alongside what is already loaded, no prompts
mlxsh lm qwen3.6 --port 8080  # on a port you choose
```

The command returns when the model has loaded and answered a warm-up request,
printing `ready in Ns` and the endpoint. It can take minutes for a large model;
`start_timeout` (default 900s) bounds it. The model argument is a repo id, a
number from `ls`, or any unique substring.

A model is loaded per port. If the configured port is busy the next model goes
to the next free port, so **loading never stops another model** unless you pass
`--replace`, or name an occupied port with `--port`.

### Ask a loaded model

```sh
mlxsh ask --on gemma "summarise this error: ..."
mlxsh ask --on 41279 --image /tmp/shot.png "what is in this picture"
```

`--on` takes a port, a model substring, or a mode (`lm`, `vision`). It is
required whenever more than one model is loaded; without it mlxsh prints the
list and exits 1 rather than guessing. Images need a server in `vision` mode.

For anything beyond a single prompt, call the endpoint directly (below) instead
of shelling out per turn.

### Unload

```sh
mlxsh stop 41279      # by port
mlxsh stop gemma      # by model
mlxsh stop all        # everything
```

Never `kill` a server yourself: `stop` signals the process group and cleans up
the state file. mlxsh refuses to signal a process whose command line does not
show it is an mlx server.

### Get a model that is not here yet

```sh
mlxsh browse qwen3.6 vision   # plain-text list when piped or non-interactive
mlxsh pull mlx-community/Qwen3.6-27B-4bit -y
```

Downloads are tens of gigabytes. Check `mlxsh ls` first, and prefer telling the
user what you are about to fetch.

## Calling the endpoint

Any OpenAI-compatible client works.

### Which model to send

**Send the exact repo id from `mlxsh status --json`.** It works against both
engines.

On a `lm` server you can also send `"model": "default_model"`, an alias mlx_lm
maps to whatever that port was started with, so a client needs no repo id.
`mlx_vlm` has no such alias and answers `400`, so only use it when
`status --json` reports one for that server.

```sh
curl -s http://127.0.0.1:41277/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"default_model","messages":[{"role":"user","content":"hello"}]}'
```

Two things to know, because both mislead agents:

- **`GET /v1/models` returns the one model that port serves.** mlx_lm builds
  that list from the Hugging Face cache, so left alone it advertises every
  model on the machine. mlxsh starts each server against a cache view holding
  only its own model, so the list matches reality. `config pin_model off`
  turns that off and restores the full listing.
- **A pinned server will not swap.** Asking it for another model returns an
  error rather than loading it, so no request can evict what you are using or
  start a download. Unpinned, mlx_lm loads the named model in place of the
  resident one, which costs a full load and can surprise other clients.

Use `default_model`, or the exact repo id from `mlxsh status --json`. Either
works; the alias means a client pinned to a port needs no repo id at all.

```sh
curl -s http://127.0.0.1:41277/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"mlx-community/gemma-4-26B-A4B-it-qat-4bit",
       "messages":[{"role":"user","content":"hello"}],
       "max_tokens":256}'
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:41277/v1", api_key="not-needed")
reply = client.chat.completions.create(
    model="mlx-community/gemma-4-26B-A4B-it-qat-4bit",
    messages=[{"role": "user", "content": "hello"}],
)
```

Streaming (`"stream": true`) returns server-sent events: lines beginning
`data: `, ending with `data: [DONE]`. Images go in the message content as a
`image_url` part with a `data:image/png;base64,...` URL, and require a `vision`
server.

**Reasoning models.** Qwen3, gpt-oss and friends put their chain of thought in
`message.reasoning` (or `delta.reasoning` when streaming) and send **no
`content` at all** until the thinking finishes. Read the answer with
`.get("content")`, never `["content"]`, and expect a long quiet stretch first.
If the token budget runs out mid-thought you get `finish_reason: "length"` and
no content: raise `max_tokens`. `mlxsh ask` handles this for you, showing the
reasoning dimmed on a terminal, keeping it out of piped output, and saying so
when a reply never arrives.

There is no authentication. The default bind address is `127.0.0.1`, so the
endpoint is local-only unless someone sets `host`.

From a container on the same machine use `http://host.docker.internal:41277/v1`.

## Wiring other tools to it

Anything that accepts an OpenAI base URL: set the base URL to
`http://127.0.0.1:<port>/v1`, the API key to any non-empty string, and the
model to the repo id from `mlxsh status`. Common environment variables:

```sh
export OPENAI_BASE_URL=http://127.0.0.1:41277/v1
export OPENAI_API_KEY=not-needed
```

## From another machine

Everything above assumes the agent runs on the Mac. If it does not, it talks to
the gateway instead of the model servers:

```
OPENAI_BASE_URL=https://<the tunnel hostname>/v1
OPENAI_API_KEY=<from: mlxsh gateway key>
```

The gateway is an ordinary OpenAI endpoint: `Authorization: Bearer <key>`,
`/v1/models` listing every loaded model, and requests routed by model name to
the port holding it. `GET /healthz` answers without a key, for readiness
checks. A model that is not loaded returns 404 naming what is available, rather
than loading it.

## Running in isolation

For tests, sandboxes or parallel agents that must not touch a user's setup:

```sh
export MLXSH_HOME=/tmp/agent-mlxsh    # own registry, logs and server state
export MLXSH_PORT=41400               # own port range
export NO_COLOR=1
```

`MLXSH_HOME` fully isolates state. Models still come from the shared Hugging
Face cache, which is what you want: they are large and identical.

## Failure modes

| output | meaning | what to do |
|---|---|---|
| `error: no running server matches 'x'` | `--on` named nothing that is loaded | `mlxsh status`, then use a real port or model |
| `error: N servers running, name one: ask --on <port\|model>` | ambiguous target | pass `--on` |
| `port 41277 is held by pid N, not an mlx server` | something else has the port | `--port` another, or stop that process yourself |
| `server exited during startup` | model failed to load | the last log lines follow; usually unsupported architecture or memory |
| `N other model(s) hold about X` then a prompt | not enough memory | `-y` to force, or `mlxsh stop` something first |
| `error: lm mode needs mlx-lm` | dependency missing | `pip install mlx-lm` into the same environment |
| `warning: no MLX implementation for <family>` | mlx-lm/mlx-vlm cannot run that repo | pick a different model |
| `error: the model spent its whole budget reasoning` | a thinking model used every token before answering | raise `max_tokens`, or use a smaller prompt |
| `401 Repository Not Found` from the endpoint | the `model` field is not an exact repo id | take it from `mlxsh status --json` |

`MLXSH_DEBUG=1` turns an unexpected error into a traceback worth reporting.

## An end-to-end example

```python
import json, subprocess, urllib.request

def servers():
    out = subprocess.run(["mlxsh", "status", "--json"], capture_output=True,
                         text=True, check=True).stdout
    return json.loads(out)

running = servers()
if not running:
    subprocess.run(["mlxsh", "lm", "-y"], check=True)   # blocks until ready
    running = servers()

server = running[0]
body = json.dumps({"model": server["model"],
                   "messages": [{"role": "user", "content": "hello"}],
                   "max_tokens": 256}).encode()
req = urllib.request.Request(server["endpoint"] + "/chat/completions", data=body,
                             headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=600) as r:
    message = json.loads(r.read())["choices"][0]["message"]
    print(message.get("content") or f"(only reasoning: {message.get('reasoning', '')[:80]})")
```

Leave the model loaded when you are done unless you started it for a one-off
task: reloading costs seconds to minutes, and the user may be using it.

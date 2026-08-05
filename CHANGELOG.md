# Changelog

## Unreleased

- `mlxsh gateway`: one authenticated endpoint in front of every loaded model.
  A bearer key, so any OpenAI client works with only a base URL and a key;
  `/v1/models` lists what is loaded; requests are routed by model name to the
  port holding it, with the upstream always receiving its own repo id.
  Streaming is passed through chunk by chunk. Binds localhost unless told
  otherwise.
- Guidance on installing cloudflared links Cloudflare's downloads page rather
  than assuming Homebrew, and `gateway -h` says the gateway alone listens on
  this machine only.
- `mlxsh tunnel --quick` for a throwaway address with no account, domain or
  configuration, reading the assigned name out of cloudflared's own output.
  `setup` and `gateway` both mention cloudflared when it is missing, and offer
  to install it.
- `mlxsh tunnel`: a permanent public address through a named cloudflared
  tunnel. `tunnel setup <hostname>` runs the login, create and DNS steps, each
  skipped when already done; `tunnel` starts the gateway first, so nothing is
  ever exposed without a key. `config tunnel_cmd` replaces cloudflared with
  anything else.

## Unreleased

- `<command> -h` explains one command with examples, and never acts. It used
  to be taken as an argument: `lm --help` stopped a running server and started
  a model, `rm --help` tried to delete a model of that name, and
  `browse --help` searched the Hub for it.
- Errors and hints go to stderr, so piping a command gives you its output
  alone. A typo suggests the nearest command.
- `help <command>` in the shell, and a leading `mlxsh` is ignored there, since
  people type it out of habit.
- `mlxsh` on a machine with no models points at `browse`, and `setup` says to
  start mlxsh again when it installed the packages this process was missing.

- `mlx-vlm` is a dependency rather than an optional extra, so an install can
  always serve both modes. `uv tool install` left it out, and vision mode then
  failed with an error telling you to run pip, which a uv-managed environment
  does not have.
- `setup` installs into the environment mlxsh is running from when that
  environment already works, instead of always building `~/.mlxsh/.venv`
  beside it.

- Each server now sees a Hugging Face cache view holding only its own model, so
  `GET /v1/models` lists that model rather than everything on the machine, and
  a stray request cannot swap the model out or download another one.
  `config pin_model off` restores mlx_lm's behaviour.
- `status --json` reports the `default_model` alias, which mlx_lm maps to
  whatever a server was started with.

## 0.2.1

Fixes from an architecture review, documentation for automated callers, and
several ways to install it.

- `mlxsh setup` installs mlx-lm, mlx-vlm and huggingface_hub into
  `~/.mlxsh/.venv`, the environment mlxsh already re-execs into. Any install
  route can bootstrap itself with it.
- Install by curling the single file, by `install.sh` (which pulls in uv when
  it is missing), or from PyPI. Releases publish on a tag.
- A clear message when the interpreter is older than MLX supports, instead of
  a pip error about an unsupported version. macOS ships Python 3.9.
- `sysctl` is called by absolute path, so the machine name survives a trimmed
  PATH.

- `status --json` and `ls --json` for scripts and agents. The human listing
  shows a shortened model name that the endpoint rejects; the JSON carries the
  full repo id.
- A failed command exits non-zero. `--on` naming a server that is not running
  now fails instead of quietly loading a private copy.
- Reasoning models stream their chain of thought with no content at first.
  `ask` and `chat` show it dimmed on a terminal, keep it out of piped output,
  and say so when the token budget runs out before an answer.
- AGENTS.md: how to drive mlxsh from a script or coding agent.

- A process is only treated as an mlx server when its command line is one of
  the forms that start one. Matching `mlx_lm` and `server` anywhere would also
  match an editor open on a file with that name, and `stop` signals a process
  group.
- The model picker no longer writes its transient `have` flag into
  `models.json`.
- `--port`, `--host` and `--new` apply to the command that carried them
  instead of the rest of the shell session.
- Settings read from a hand-edited registry are coerced to the declared type,
  and fall back to the default with a warning when they cannot be.
- `bench` survives a hung or unreachable server instead of raising.
- `bench_timeout` is now `reply_timeout` and covers `ask` and `chat` too.
- `status` uses one `ps` call for all servers rather than two per server.
- An unexpected error prints a one-line message; `MLXSH_DEBUG=1` restores the
  traceback.

## 0.2.0

Multiple models at once.

- A model per port. Serving no longer stops what is already running: a busy
  port sends the new model to the next free one. `when_busy` chooses between
  `new` (default), `replace` and `ask`; `--new` and `--replace` override per run.
- Loading a model that is already running in the other mode still swaps the
  engine in place rather than loading a second copy.
- `status` lists every server with resident size, uptime and endpoint. `stop`
  and `bench` take a port, a model substring, a mode, or `all`, and open a
  picker in the shell.
- A memory check before adding a model, based on weight size rather than
  resident size, because macOS pages idle weights out.
- `ask` and `chat` talk to a running model over its endpoint instead of loading
  a private copy. `--on <port|model>` picks which one, `--load` forces a copy.
  Images are sent as data URLs and require a vision server.
- One log per server, `~/.mlxsh/logs/<port>.log`, and `log --on <port|model>`.
- A status bar pinned to the top of the shell showing what is loaded, refreshed
  on a timer. The prompt is plain `mlxsh>` again. `config status_bar off` to
  hide it.
- The Hugging Face cache is only scanned by commands that list models.

## 0.1.0

First release, extracted from a set of personal shell scripts.

- `lm` and `vision` modes behind one OpenAI-compatible endpoint, with the
  vision tower skipped in `lm` mode.
- An interactive shell with arrow-key pickers, tab completion and history.
- `browse`, a live list of MLX models from the Hub with a memory estimate
  computed from the safetensors dtype census, so 4-bit packing is counted
  correctly.
- A model registry with measured tok/s written back by `bench`.
- Settings with a `config` command: flag, environment, registry, default.
- Servers are identified by their command line before anything is signalled.

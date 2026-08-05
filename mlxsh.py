#!/usr/bin/env python3
"""
mlxsh: a shell for running local MLX models on Apple silicon.

    mlxsh                 help
    mlxsh shell           help, then the interactive shell
    mlxsh <command> ...   run one command and exit

Two modes behind one OpenAI-compatible endpoint:

  lm      mlx_lm    text only, no vision tower
  vision  mlx_vlm   multimodal

One server at a time, so switching modes frees the previous model first.

MIT licensed. https://github.com/ansnadeem/mlxsh
"""

from __future__ import annotations

import atexit
import difflib
import functools
import json
import os
import platform
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

try:
    import select as _select
    import termios
    import tty

    HAS_TTY = True
except ImportError:
    HAS_TTY = False

__version__ = "0.2.1"

SELF = Path(__file__).resolve()

HOME = Path(os.environ.get("MLXSH_HOME") or Path.home() / ".mlxsh")
LEGACY_HOMES = [Path.home() / ".mlx-shell"]
REGISTRY = HOME / "models.json"
STATE = HOME / "server.json"       # pre-0.2 single server file, migrated away
STATE_DIR = HOME / "servers"       # one file per running server, named by port
HISTORY = HOME / "history"
LOG = HOME / "server.log"
LOG_MAX = 5_000_000

DEFAULTS = {
    "version": 1,
    "host": "127.0.0.1",
    "port": 41277,
    "org": "mlx-community",
    "mem_comfy": 0.70,
    "mem_tight": 0.85,
    "browse_limit": 25,
    "start_timeout": 900,
    "reply_timeout": 600,
    "when_busy": "new",
    "pin_model": True,
    "gateway_port": 41377,
    "tunnel_name": "mlxsh",
    "tunnel_hostname": "",
    "tunnel_cmd": "",
    "status_bar": True,
    "bar_interval": 2.0,
    "serve_args": {"lm": ["--max-tokens", "4096"],
                   "vision": ["--max-tokens", "4096"]},
    "chat_args": ["--max-tokens", "4096"],
    "ask_args": ["--max-tokens", "2048"],
    "defaults": {"lm": "", "vision": ""},
    "models": [],
}

# key -> (env var, type, help)
SETTINGS = {
    "host": ("MLXSH_HOST", str, "address the server binds to"),
    "port": ("MLXSH_PORT", int, "port the server listens on"),
    "org": ("MLXSH_ORG", str, "hub org searched by browse, assumed by pull"),
    "mem_comfy": ("MLXSH_MEM_COMFY", float, "fraction of RAM a model can use freely"),
    "mem_tight": ("MLXSH_MEM_TIGHT", float, "fraction of RAM above which browse hides a model"),
    "browse_limit": ("MLXSH_BROWSE_LIMIT", int, "rows per browse"),
    "start_timeout": ("MLXSH_START_TIMEOUT", int, "seconds to wait for a server to load"),
    "reply_timeout": ("MLXSH_REPLY_TIMEOUT", int,
                      "seconds to wait for a reply from a server"),
    "when_busy": ("MLXSH_WHEN_BUSY", str,
                  "port already has a model: new, replace or ask"),
    "pin_model": ("MLXSH_PIN_MODEL", lambda v: boolean(v),
                  "a server sees only its own model, and cannot swap (on/off)"),
    "gateway_port": ("MLXSH_GATEWAY_PORT", int,
                     "port the authenticated gateway listens on"),
    "tunnel_name": ("MLXSH_TUNNEL_NAME", str, "cloudflared tunnel name"),
    "tunnel_hostname": ("MLXSH_TUNNEL_HOSTNAME", str,
                        "public hostname routed to the gateway"),
    "tunnel_cmd": ("MLXSH_TUNNEL_CMD", str,
                   "run this instead of cloudflared, {port} is substituted"),
    "status_bar": ("MLXSH_STATUS_BAR", lambda v: boolean(v),
                   "pin a live line at the top of the shell (on/off)"),
    "bar_interval": ("MLXSH_BAR_INTERVAL", float,
                     "seconds between status bar refreshes"),
}


def boolean(v) -> bool:
    if isinstance(v, bool):
        return v
    if str(v).lower() in ("on", "true", "yes", "1"):
        return True
    if str(v).lower() in ("off", "false", "no", "0"):
        return False
    raise ValueError(v)

# Set by --host/--port, for this run only.
OVERRIDES: dict[str, object] = {}

# Set by --new or --port auto: pick the next free port for this run.
AUTO_PORT: set[bool] = set()

# Bytes per element. MLX packs 4-bit weights into U32 words, so counting
# parameters alone is off by 8x.
DTYPE_BYTES = {
    "F64": 8, "I64": 8, "F32": 4, "U32": 4, "I32": 4,
    "BF16": 2, "F16": 2, "I16": 2, "U16": 2,
    "F8_E4M3": 1, "F8_E5M2": 1, "U8": 1, "I8": 1, "BOOL": 1, "F4": 0.5,
}

TEXT_TAGS = {"text-generation", "text2text-generation"}
VISION_TAGS = {"image-text-to-text", "visual-question-answering", "any-to-any",
               "video-text-to-text"}


# --------------------------------------------------------------------- machine


def sysctl(name: str) -> str:
    # absolute path: sysctl lives in /usr/sbin, which is missing from the
    # trimmed PATH of a launchd job or a hook
    for exe in ("/usr/sbin/sysctl", "sysctl"):
        try:
            return subprocess.run([exe, "-n", name], capture_output=True,
                                  text=True, check=True).stdout.strip()
        except Exception:
            continue
    return ""


def total_memory() -> int:
    v = sysctl("hw.memsize")
    if v.isdigit():
        return int(v)
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return 16 * 1000 ** 3


def machine_name() -> str:
    chip = sysctl("machdep.cpu.brand_string") or platform.processor() \
        or platform.machine()
    return f"{chip}, {total_memory() / 2 ** 30:.0f} GB"


MEM_TOTAL = total_memory()


def mem_comfy() -> float:
    return MEM_TOTAL * setting("mem_comfy")


def mem_tight() -> float:
    return MEM_TOTAL * setting("mem_tight")


# ---------------------------------------------------------------------- output

TTY = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def c(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if TTY else s


def bold(s: str) -> str:
    return c("1", s)


def dim(s: str) -> str:
    return c("2", s)


def red(s: str) -> str:
    return c("31", s)


def green(s: str) -> str:
    return c("32", s)


def yellow(s: str) -> str:
    return c("33", s)


def blue(s: str) -> str:
    return c("34", s)


def cyan(s: str) -> str:
    return c("36", s)


def gb(n: float | None) -> str:
    return "?" if not n else f"{n / 1e9:.1f} GB"


def short(repo: str) -> str:
    return repo.split("/", 1)[-1]


# Set by warn(), read by cli() so a failed one-shot command exits non-zero.
FAILED: list[str] = []


def warn(msg: str):
    # diagnostics on stderr, so piping a command gives you its output alone
    FAILED.append(msg)
    print(red("error: ") + msg, file=sys.stderr)


def note(msg: str):
    """A hint that belongs with an error, not with the output."""
    print(msg, file=sys.stderr)


def die(msg: str, code: int = 1):
    warn(msg)
    sys.exit(code)


def confirm(prompt: str, default: bool = False) -> bool:
    try:
        a = input(prompt + (" [Y/n] " if default else " [y/N] ")).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return default if not a else a.startswith("y")


# ---------------------------------------------------------------------- picker

ANSI = re.compile(r"\033\[[0-9;]*m")

UP, DOWN, PGUP, PGDN, HOME_KEY, END_KEY = "up", "down", "pgup", "pgdn", "home", "end"

SEQUENCES = {
    "[A": UP, "OA": UP, "[B": DOWN, "OB": DOWN,
    "[5~": PGUP, "[6~": PGDN,
    "[H": HOME_KEY, "OH": HOME_KEY, "[F": END_KEY, "OF": END_KEY,
    "[1~": HOME_KEY, "[4~": END_KEY,
}


def supports_tui() -> bool:
    return (HAS_TTY and sys.stdin.isatty() and sys.stdout.isatty()
            and os.environ.get("TERM") != "dumb")


def visible_len(s: str) -> int:
    return len(ANSI.sub("", s))


def truncate(s: str, width: int) -> str:
    """Cut to width visible columns, keeping colour codes."""
    if visible_len(s) <= width:
        return s
    out, seen, i = [], 0, 0
    while i < len(s) and seen < width - 1:
        m = ANSI.match(s, i)
        if m:
            out.append(m.group())
            i = m.end()
            continue
        out.append(s[i])
        seen += 1
        i += 1
    return "".join(out) + "…\033[0m"


def read_key(fd: int) -> str:
    """Read one keypress from a tty in raw mode."""
    # os.read, not sys.stdin.read: the text layer buffers the rest of an
    # escape sequence out of reach of select(), so arrows arrive as "[A".
    b = os.read(fd, 1)
    if not b:
        return "\x03"
    if b != b"\x1b":
        if b[0] >= 0x80:
            while len(b) < 4 and _select.select([fd], [], [], 0.005)[0]:
                more = os.read(fd, 1)
                if not more:
                    break
                b += more
                try:
                    return b.decode()
                except UnicodeDecodeError:
                    continue
        return b.decode(errors="ignore")
    seq = b""
    while len(seq) < 8 and _select.select([fd], [], [], 0.03)[0]:
        more = os.read(fd, 1)
        if not more:
            break
        seq += more
        if seq in (b"[", b"O"):
            continue
        if seq[-1:].isalpha() or seq.endswith(b"~"):
            break
    if not seq:
        return "\x1b"
    return SEQUENCES.get(seq.decode(errors="ignore"), "")


def pick(rows, title="", render=str, key=None, index=0, multi=False,
         height=None, hint=""):
    """Arrow-key list. Returns the chosen row, a list of rows if multi, or None.

    Draws below the prompt and erases itself on exit.
    """
    if not rows:
        return [] if multi else None
    if not supports_tui():
        return None

    key = key or (lambda r: ANSI.sub("", render(r)))
    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    cols, lines = shutil.get_terminal_size((100, 30))
    view = height or max(3, min(len(rows), lines - 8, 14))

    query = ""
    picked: set[int] = set()
    cur = min(max(index, 0), len(rows) - 1)
    top = 0
    drawn = 0

    def matches() -> list[int]:
        if not query:
            return list(range(len(rows)))
        q = query.lower()
        return [i for i in range(len(rows))
                if all(w in key(rows[i]).lower() for w in q.split())]

    def frame(view_rows: list[int], sel: int) -> list[str]:
        out = []
        head = bold(title) if title else ""
        tips = hint or ("up/down select   space mark   enter ok   esc cancel"
                        if multi else "up/down select   enter ok   esc cancel")
        if query:
            head += "   " + cyan("/" + query)
        pad = cols - visible_len(head) - visible_len(tips) - 2
        out.append(truncate(head + " " * max(pad, 1) + dim(tips), cols))
        if not view_rows:
            out.append("  " + dim("no matches"))
            return out
        for i in view_rows[top:top + view]:
            marker = cyan(">") if i == sel else " "
            mark = ("[x]" if i in picked else "[ ]") if multi else ""
            body = render(rows[i])
            if i == sel:
                body = "\033[7m" + ANSI.sub("", body) + "\033[0m"
            out.append(truncate(f" {marker} {mark} {body}".rstrip(), cols))
        above, below = top, max(0, len(view_rows) - top - view)
        if above or below:
            bits = ([f"{above} above"] if above else []) + \
                   ([f"{below} below"] if below else [])
            out.append("   " + dim(", ".join(bits)))
        return out

    def draw(view_rows: list[int], sel: int):
        nonlocal drawn
        block = frame(view_rows, sel)
        buf = [f"\r\033[{drawn}A"] if drawn else []
        buf += ["\033[2K" + ln + "\r\n" for ln in block]
        extra = max(0, drawn - len(block))
        if extra:
            buf += ["\033[2K\r\n"] * extra + [f"\033[{extra}A"]
        sys.stdout.write("".join(buf))
        sys.stdout.flush()
        drawn = len(block)

    def erase():
        if drawn:
            sys.stdout.write(f"\r\033[{drawn}A" + "\033[2K\033[1B" * drawn
                             + f"\033[{drawn}A")
            sys.stdout.flush()

    result = None
    BAR.pause()
    try:
        sys.stdout.write("\033[?25l")
        tty.setraw(fd)
        while True:
            vis = matches()
            if not vis:
                draw(vis, -1)
            else:
                if cur not in vis:
                    cur = vis[0]
                pos = vis.index(cur)
                if pos < top:
                    top = pos
                elif pos >= top + view:
                    top = pos - view + 1
                top = max(0, min(top, max(0, len(vis) - view)))
                draw(vis, cur)

            k = read_key(fd)
            if not k:
                continue
            if k == "\x03":
                break
            if k == "\x1b":
                if query:
                    query = ""
                    continue
                break
            if k in ("\r", "\n"):
                if multi:
                    result = [rows[i] for i in sorted(picked)] or (
                        [rows[cur]] if vis else [])
                else:
                    result = rows[cur] if vis else None
                break
            if k == " " and multi:
                if vis:
                    picked.symmetric_difference_update({cur})
                    cur = vis[min(vis.index(cur) + 1, len(vis) - 1)]
                continue
            if k in (UP, DOWN, PGUP, PGDN, HOME_KEY, END_KEY, "j", "k"):
                if not vis:
                    continue
                pos = vis.index(cur)
                if k == HOME_KEY:
                    pos = 0
                elif k == END_KEY:
                    pos = len(vis) - 1
                else:
                    step = {UP: -1, "k": -1, DOWN: 1, "j": 1,
                            PGUP: -view, PGDN: view}[k]
                    pos = max(0, min(len(vis) - 1, pos + step))
                cur = vis[pos]
                continue
            if k in ("\x7f", "\b"):
                query = query[:-1]
                continue
            if k.isprintable():
                query += k
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)
        erase()
        sys.stdout.write("\033[?25h")
        sys.stdout.flush()
        BAR.resume()
    return result


# -------------------------------------------------------------------- registry

_registry: dict | None = None


def ensure_home():
    HOME.mkdir(parents=True, exist_ok=True)
    if REGISTRY.exists() or HOME != Path.home() / ".mlxsh":
        return
    for old in [SELF.parent / "models.json",
                *[h / "models.json" for h in LEGACY_HOMES]]:
        if old.exists() and old != REGISTRY:
            shutil.copy2(old, REGISTRY)
            print(dim(f"  copied {old} to {REGISTRY}"))
            return


def load_registry(refresh: bool = False) -> dict:
    global _registry
    if _registry is not None and not refresh:
        return _registry
    ensure_home()
    if REGISTRY.exists():
        try:
            reg = json.loads(REGISTRY.read_text())
        except json.JSONDecodeError as e:
            die(f"{REGISTRY} is not valid json ({e})")
    else:
        reg = {}
    for k, v in DEFAULTS.items():
        reg.setdefault(k, json.loads(json.dumps(v)))
    if not REGISTRY.exists():
        REGISTRY.write_text(json.dumps(reg, indent=2) + "\n")
    _registry = reg
    return reg


def save_registry(reg: dict):
    global _registry
    ensure_home()
    REGISTRY.write_text(json.dumps(reg, indent=2) + "\n")
    _registry = reg


def setting(key: str):
    """Resolve a setting: flag, then env, then registry, then default."""
    value, _ = setting_with_source(key)
    return value


def setting_with_source(key: str) -> tuple[object, str]:
    if key in OVERRIDES:
        return OVERRIDES[key], "flag"
    env_var, kind, _ = SETTINGS.get(key, (None, str, ""))
    raw = os.environ.get(env_var) if env_var else None
    if raw:
        try:
            return kind(raw), "env"
        except ValueError:
            warn(f"{env_var}={raw!r} is not a valid {kind.__name__}, ignoring")
    reg = load_registry()
    if key in reg and reg[key] != DEFAULTS.get(key):
        try:
            return kind(reg[key]), "registry"
        except (ValueError, TypeError):
            warn(f"{key}={reg[key]!r} in the registry is not a valid "
                 f"{getattr(kind, '__name__', 'value')}, using the default")
    return DEFAULTS[key], "default"


def set_setting(key: str, raw: str) -> bool:
    if key not in SETTINGS:
        warn(f"unknown setting {key!r}")
        return False
    kind = SETTINGS[key][1]
    try:
        value = kind(raw)
    except ValueError:
        warn(f"{raw!r} is not a valid {kind.__name__}")
        return False
    if key == "port" and not 1 <= value <= 65535:
        warn("port must be between 1 and 65535")
        return False
    if key in ("mem_comfy", "mem_tight") and not 0 < value <= 1:
        warn(f"{key} must be between 0 and 1")
        return False
    if key == "when_busy" and value not in ("new", "replace", "ask"):
        warn("when_busy must be new, replace or ask")
        return False
    reg = load_registry()
    reg[key] = value
    save_registry(reg)
    return True


def reset_setting(key: str) -> bool:
    if key not in SETTINGS:
        warn(f"unknown setting {key!r}")
        return False
    reg = load_registry()
    reg[key] = DEFAULTS[key]
    save_registry(reg)
    return True


def adopt_cached_models(reg: dict) -> dict:
    """Register cached repos that MLX can run."""
    known = {m["repo"] for m in reg["models"]}
    changed = False
    for repo in sorted(cache_sizes()):
        if repo in known or not is_downloaded(repo):
            continue
        cfg = local_config(repo) or {}
        if not mlx_can_run(cfg):
            continue
        reg["models"].append({
            "repo": repo,
            "label": short(repo),
            "vision": is_vision_config(cfg),
            "note": "found in the hugging face cache",
        })
        changed = True
    if changed:
        save_registry(reg)
        set_missing_defaults(reg)
    return reg


def set_missing_defaults(reg: dict) -> dict:
    changed = False
    for mode in ("lm", "vision"):
        if reg["defaults"].get(mode):
            continue
        pool = [m for m in reg["models"]
                if is_downloaded(m["repo"]) and (m.get("vision") or mode == "lm")]
        if pool:
            reg["defaults"][mode] = pool[0]["repo"]
            changed = True
    if changed:
        save_registry(reg)
    return reg


def find_model(reg: dict, needle: str) -> dict | None:
    """resolve(), but scan the cache before giving up.

    A fresh registry knows nothing, so a name that matches a model already in
    the Hugging Face cache would otherwise be treated as a repo to download.
    """
    entry = resolve(reg, needle)
    if entry or "/" in needle:
        return entry
    return resolve(adopt_cached_models(load_registry(refresh=True)), needle)


def resolve(reg: dict, needle: str) -> dict | None:
    """Find a model by index, repo id, or unique substring."""
    needle = (needle or "").strip()
    if not needle:
        return None
    if needle.isdigit():
        i = int(needle)
        return reg["models"][i - 1] if 1 <= i <= len(reg["models"]) else None
    for m in reg["models"]:
        if m["repo"] == needle:
            return m
    low = needle.lower()
    hits = [m for m in reg["models"]
            if low in m["repo"].lower() or low in (m.get("label") or "").lower()]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        warn(f"{needle!r} matches {len(hits)}: "
             + ", ".join(short(h["repo"]) for h in hits[:5]))
    return None


# ---------------------------------------------------------------- hugging face


def cache_sizes() -> dict[str, int]:
    try:
        from huggingface_hub import scan_cache_dir

        info = scan_cache_dir()
        return {r.repo_id: r.size_on_disk for r in info.repos
                if r.repo_type == "model"}
    except Exception:
        return {}


def hub_cache() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def snapshot_dir(repo: str) -> Path | None:
    d = hub_cache() / ("models--" + repo.replace("/", "--")) / "snapshots"
    if not d.is_dir():
        return None
    snaps = sorted(d.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    return snaps[0] if snaps else None


def is_downloaded(repo: str) -> bool:
    """True if weights are present, not just a cached config.json."""
    snap = snapshot_dir(repo)
    if not snap:
        return False
    return any(any(snap.glob(p)) for p in ("*.safetensors", "*.npz", "*.gguf"))


def local_config(repo: str) -> dict | None:
    snap = snapshot_dir(repo)
    if not snap:
        return None
    try:
        return json.loads((snap / "config.json").read_text())
    except Exception:
        return None


def remote_config(repo: str) -> dict | None:
    try:
        from huggingface_hub import hf_hub_download

        return json.loads(Path(hf_hub_download(repo, "config.json")).read_text())
    except Exception:
        return None


def is_vision_config(cfg: dict) -> bool:
    if not cfg:
        return False
    if "vision_config" in cfg or "vision_tower" in cfg or "image_token_index" in cfg:
        return True
    archs = " ".join(cfg.get("architectures") or [])
    return bool(re.search(r"VL|Vision|ImageText|Conditional", archs))


@functools.lru_cache(maxsize=1)
def mlx_families() -> frozenset[str]:
    """Model families the installed mlx_lm and mlx_vlm implement."""
    out: set[str] = set()
    for mod in ("mlx_lm", "mlx_vlm"):
        try:
            import importlib.util

            spec = importlib.util.find_spec(mod)
            if not spec or not spec.origin:
                continue
            d = Path(spec.origin).parent / "models"
            out |= {p.stem for p in d.glob("*.py") if not p.stem.startswith("_")}
            out |= {p.name for p in d.iterdir()
                    if p.is_dir() and not p.name.startswith("_")}
        except Exception:
            continue
    return frozenset(out)


def mlx_can_run(cfg: dict) -> bool:
    """Quantized repos always load; others need an implemented architecture."""
    if not cfg:
        return False
    if "quantization" in cfg or "quantization_config" in cfg:
        return True
    fams = mlx_families()
    if not fams:
        return True
    mt = (cfg.get("model_type") or "").lower()
    return bool(mt) and (mt in fams or f"{mt}_text" in fams)


def est_memory(info) -> float:
    st = getattr(info, "safetensors", None)
    if not st or not st.parameters:
        return 0.0
    return sum(DTYPE_BYTES.get(k, 2) * v for k, v in st.parameters.items())


def fit_mark(size: float) -> str:
    if not size:
        return dim("     ")
    if size <= mem_comfy():
        return green(" fits")
    if size <= mem_tight():
        return yellow("tight")
    return red("  big")


def size_colour(size: float) -> str:
    text = f"{gb(size):>8}"
    if size <= mem_comfy():
        return green(text)
    return yellow(text) if size <= mem_tight() else red(text)


def parse_browse_args(args: list[str]) -> dict:
    """Turn browse arguments into search parameters."""
    kind = sort = None
    include_all = False
    terms = []
    for tok in args:
        t = tok.lower()
        if t in ("vision", "vlm", "multimodal"):
            kind = "vision"
        elif t in ("text", "lm"):
            kind = "text"
        elif t in ("trending", "hot"):
            sort = "trendingScore"
        elif t in ("popular", "downloads", "top"):
            sort = "downloads"
        elif t in ("new", "recent", "latest"):
            sort = "lastModified"
        elif t == "all":
            include_all = True
        else:
            terms.append(tok)
    return {"terms": terms, "kind": kind, "include_all": include_all,
            "sort": sort or ("downloads" if terms else "trendingScore")}


def hub_list(terms: list[str] | None = None, kind: str | None = None,
             sort: str = "trendingScore", author: str | None = None,
             limit: int | None = None, include_all: bool = False) -> list[dict]:
    from huggingface_hub import HfApi

    author = author if author is not None else setting("org")
    limit = limit or setting("browse_limit")
    query = " ".join(terms or "") or None
    if query and "/" in query:
        author, query = query.split("/", 1)
        author, query = author or None, query or None
    pipeline = {"vision": "image-text-to-text", "text": "text-generation"}.get(kind)

    kw = {"search": query, "author": author, "pipeline_tag": pipeline,
          "sort": sort, "filter": "mlx", "limit": limit * 3,
          "expand": ["downloads", "trendingScore", "tags", "safetensors",
                     "pipeline_tag", "lastModified"]}
    api = HfApi()
    try:
        raw = list(api.list_models(**kw))
    except TypeError:
        kw.pop("expand", None)
        raw = list(api.list_models(**kw))

    ceiling = mem_tight()
    rows = []
    for m in raw:
        tag = m.pipeline_tag or ""
        tags = set(m.tags or [])
        if not include_all and tag not in TEXT_TAGS | VISION_TAGS:
            continue
        size = est_memory(m)
        if not include_all and size > ceiling:
            continue
        rows.append({
            "repo": m.id,
            "vision": tag in VISION_TAGS or bool(tags & VISION_TAGS),
            "size": size,
            "downloads": m.downloads or 0,
            "trending": getattr(m, "trending_score", 0) or 0,
            "updated": m.last_modified,
            "have": is_downloaded(m.id),
        })
        if len(rows) >= limit:
            break
    return rows


# -------------------------------------------------------------- server control


def python_bin() -> str:
    return sys.executable


def port_open(port: int, host: str | None = None) -> bool:
    with socket.socket() as s:
        s.settimeout(0.4)
        return s.connect_ex((host or setting("host"), port)) == 0


def pid_alive(pid: int) -> bool:
    if not pid or pid < 1:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def pid_command(pid: int) -> str:
    try:
        return subprocess.run(["ps", "-o", "command=", "-p", str(pid)],
                              capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return ""


# Either "python -m mlx_lm server", "python -m mlx_vlm.server", or the
# console scripts mlx-lm installs. Matching "mlx_lm" and "server" anywhere in
# the command line would also match an editor open on a file of that name.
SERVER_CMD = re.compile(r"-m\s+mlx_(lm|vlm)(\.server)?\b"
                        r"|(^|/)mlx_(lm|vlm)\.server\b")


def is_our_server(pid: int) -> bool:
    """True if the pid is an mlx server, not something that reused the number."""
    if not pid_alive(pid):
        return False
    cmd = pid_command(pid)
    return bool(SERVER_CMD.search(cmd)) and "server" in cmd


def state_path(port: int) -> Path:
    return STATE_DIR / f"{port}.json"


def write_state(st: dict):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    state_path(st["port"]).write_text(json.dumps(st, indent=2) + "\n")


def migrate_state():
    """Move the old single server.json into the per-port directory."""
    if not STATE.exists():
        return
    try:
        st = json.loads(STATE.read_text())
        if st.get("port"):
            write_state(st)
    except Exception:
        pass
    STATE.unlink(missing_ok=True)


def read_state(port: int | None = None) -> dict | None:
    """Recorded server for one port, or None if it is gone."""
    migrate_state()
    port = port or setting("port")
    p = state_path(port)
    if not p.exists():
        return None
    try:
        st = json.loads(p.read_text())
    except Exception:
        p.unlink(missing_ok=True)
        return None
    if is_our_server(st.get("pid", -1)):
        return st
    p.unlink(missing_ok=True)
    return None


def list_servers(scan_port: bool = True) -> list[dict]:
    """Every mlxsh server still running, plus any found on the current port.

    scan_port=False skips the lsof lookup, for callers that run on a timer.
    """
    migrate_state()
    out = []
    if STATE_DIR.is_dir():
        for f in sorted(STATE_DIR.glob("*.json")):
            try:
                st = json.loads(f.read_text())
            except Exception:
                f.unlink(missing_ok=True)
                continue
            if is_our_server(st.get("pid", -1)):
                out.append(st)
            else:
                f.unlink(missing_ok=True)
    port = setting("port")
    if scan_port and not any(s.get("port") == port for s in out):
        found = adopt_on_port(port)
        if found:
            out.append(found)
    return sorted(out, key=lambda s: s.get("port", 0))


def adopt_on_port(port: int) -> dict | None:
    """An mlx server we did not start, listening on this port."""
    pid = port_owner(port)
    if not pid or not is_our_server(pid):
        return None
    cmd = pid_command(pid)
    parts = cmd.split()
    model = parts[parts.index("--model") + 1] if "--model" in parts else ""
    engine = "mlx_vlm" if "mlx_vlm" in cmd else "mlx_lm"
    return {"pid": pid, "mode": "vision" if engine == "mlx_vlm" else "lm",
            "model": model, "port": port, "host": setting("host"),
            "engine": engine, "adopted": True}


def proc_stats(pid: int) -> tuple[int, str]:
    return proc_stats_many([pid]).get(pid, (0, "?"))


def proc_stats_many(pids: list[int]) -> dict[int, tuple[int, str]]:
    """Resident size and uptime for several pids in one ps call."""
    pids = [p for p in pids if p]
    if not pids:
        return {}
    try:
        out = subprocess.run(
            ["ps", "-o", "pid=,rss=,etime=", "-p", ",".join(str(p) for p in pids)],
            capture_output=True, text=True, check=True).stdout
    except Exception:
        return {}
    stats = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 3:
            stats[int(parts[0])] = (int(parts[1]) * 1024, parts[2])
    return stats


def port_owner(port: int) -> int | None:
    try:
        out = subprocess.run(["lsof", "-nP", f"-tiTCP:{port}", "-sTCP:LISTEN"],
                             capture_output=True, text=True).stdout.split()
        return int(out[0]) if out else None
    except Exception:
        return None


def find_server(port: int) -> dict | None:
    """Recorded server on this port, or one running there that we did not start."""
    return read_state(port) or adopt_on_port(port)


def free_port(base: int | None = None) -> int:
    """First port at or above base with nothing on it."""
    base = base or setting("port")
    taken = {s.get("port") for s in list_servers()}
    for p in range(base, base + 100):
        if p not in taken and not port_open(p):
            return p
    return base


def match_servers(target: str | None) -> list[dict]:
    """Servers named by a port, a model substring, or "all"."""
    servers = list_servers()
    if not target:
        return servers
    if target == "all":
        return servers
    if target.isdigit():
        return [s for s in servers if s.get("port") == int(target)]
    low = target.lower()
    return [s for s in servers
            if low in (s.get("model") or "").lower()
            or low == (s.get("mode") or "")]


def stop_one(st: dict) -> int:
    pid = st["pid"]
    freed, _ = proc_stats(pid)
    print(f"  stopping {st['mode']} {short(st.get('model') or '') or pid} "
          f"on :{st['port']} ...", end="", flush=True)
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(os.getpgid(pid), sig)
        except OSError:
            try:
                os.kill(pid, sig)
            except OSError:
                break
        for _ in range(60):
            if not pid_alive(pid):
                break
            time.sleep(0.25)
        if not pid_alive(pid):
            break
    state_path(st["port"]).unlink(missing_ok=True)
    drop_cache_view(st["port"])
    print(green(" stopped") + (dim(f"   {gb(freed)} freed") if freed else ""))
    return freed


def stop_server(target: str | None = None, quiet: bool = False,
                choose=None) -> bool:
    """Stop one server, or all of them. target: port, model, mode, or "all"."""
    servers = match_servers(target)
    if not servers:
        port = setting("port")
        busy = port_owner(port)
        if target and list_servers():
            warn(f"no running server matches {target!r}")
            print_servers(list_servers())
        elif busy:
            warn(f"port {port} is held by pid {busy}, which is not an mlx server")
            note(dim(f"  {pid_command(busy)[:100]}"))
        elif not quiet:
            print(dim("  no server running"))
        return False

    if len(servers) > 1 and target != "all":
        chosen = choose(servers) if choose else None
        if chosen:
            servers = [chosen]
        elif choose and supports_tui():
            return False  # the picker was cancelled
        else:
            warn(f"{len(servers)} servers running, name one or use: stop all")
            print_servers(servers)
            return False

    freed = sum(stop_one(s) for s in servers)
    if len(servers) > 1 and freed:
        print(dim(f"  {gb(freed)} freed in total"))
    return True


def warmup(host: str, port: int, repo: str) -> bool:
    """Send one request so weights are loaded before we report ready."""
    body = json.dumps({"model": repo, "max_tokens": 1, "temperature": 0.0,
                       "messages": [{"role": "user", "content": "hi"}]}).encode()
    req = urllib.request.Request(f"http://{host}:{port}/v1/chat/completions",
                                 data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=setting("start_timeout")) as r:
            r.read()
        return True
    except Exception:
        return False


def log_path(port: int | str | None = None) -> Path:
    """One log per server, so several models do not interleave."""
    return (HOME / "logs" / f"{port}.log") if port else LOG


def cache_view(port: int, repo: str) -> Path | None:
    """A cache directory holding only this model.

    mlx_lm builds /v1/models by scanning the Hugging Face cache, so a server
    started against the real cache advertises every model on the machine and a
    client cannot tell which one is loaded. Pointing HF_HUB_CACHE at a
    directory with a single symlink narrows that list to the model actually
    being served, and keeps a stray request from swapping it out.
    """
    real = hub_cache() / ("models--" + repo.replace("/", "--"))
    if not real.exists():
        return None
    view = HOME / "views" / str(port)
    shutil.rmtree(view, ignore_errors=True)
    view.mkdir(parents=True, exist_ok=True)
    try:
        (view / real.name).symlink_to(real, target_is_directory=True)
    except OSError:
        return None
    return view


def drop_cache_view(port: int):
    shutil.rmtree(HOME / "views" / str(port), ignore_errors=True)


def rotate_log(path: Path | None = None):
    path = path or LOG
    try:
        if path.exists() and path.stat().st_size > LOG_MAX:
            path.replace(path.with_suffix(path.suffix + ".1"))
    except OSError:
        pass


def model_footprint(repo: str, pid: int = 0, sizes: dict | None = None) -> int:
    """What a loaded model costs. Resident size drops when macOS pages idle
    weights out, so the weights on disk are the more honest number."""
    sizes = cache_sizes() if sizes is None else sizes
    rss = proc_stats(pid)[0] if pid else 0
    return max(rss, sizes.get(repo, 0))


def memory_check(repo: str, port: int, yes: bool = False) -> bool:
    """Warn before another model pushes past what this machine can hold."""
    others = [s for s in list_servers() if s.get("port") != port]
    if not others:
        return True
    sizes = cache_sizes()
    used = sum(model_footprint(s.get("model") or "", s["pid"], sizes)
               for s in others)
    want = model_footprint(repo, sizes=sizes)
    if used + want <= mem_tight():
        print(dim(f"  {len(others)} other model(s) loaded, about {gb(used)}"))
        return True
    print(yellow(f"  {len(others)} other model(s) hold about {gb(used)}, "
                 f"{short(repo)} needs {gb(want)} more, "
                 f"this machine has {gb(MEM_TOTAL)}"))
    return yes or confirm("  start it anyway?", False)


def choose_port(repo: str, policy: str | None = None) -> int | None:
    """Which port to serve on. None means the user backed out.

    A port named on the command line or in the environment is taken as given.
    Otherwise the configured port is used when free, and when_busy decides what
    happens if another model already has it.
    """
    base = load_registry().get("port") or DEFAULTS["port"]
    if AUTO_PORT:
        return free_port(base)
    value, source = setting_with_source("port")
    if source in ("flag", "env"):
        return int(value)

    # the same model in the other mode is an engine swap, not a second copy
    same = next((s for s in list_servers() if s.get("model") == repo), None)
    if same:
        return same["port"]

    st = find_server(base)
    if not st:
        return base
    busy = short(st.get("model") or "") or f"pid {st['pid']}"

    policy = policy or setting("when_busy")
    if policy == "ask" and supports_tui():
        nxt = free_port(base)
        opts = [("new", (f"start on :{nxt}, keep "
                         f"{short(st.get('model') or '')} on :{base}")),
                ("replace", f"replace {short(st.get('model') or '')} on :{base}")]
        got = pick(opts, title=f":{base} is busy", render=lambda o: o[1],
                   key=lambda o: o[0])
        if not got:
            return None
        policy = got[0]
    if policy == "replace":
        return base
    nxt = free_port(base)
    print(dim(f"  :{base} has {busy}, starting on :{nxt}"))
    return nxt


def serve(mode: str, repo: str | None = None, extra: list[str] | None = None,
          foreground: bool = False, yes: bool = False,
          policy: str | None = None):
    reg = load_registry()
    host = setting("host")
    if repo:
        entry = find_model(reg, repo)
        repo = entry["repo"] if entry else repo
    else:
        repo = reg["defaults"].get(mode)
    if not repo:
        warn(f"no default model for {mode} mode, run browse then use <model> {mode}")
        return

    if mode == "vision":
        if not have_module("mlx_vlm"):
            warn("vision mode needs mlx-vlm")
            note(dim("  run: mlxsh setup"))
            return
        cfg = local_config(repo)
        if cfg is not None and not is_vision_config(cfg):
            warn(f"{short(repo)} has no vision tower, run it in lm mode")
            return
    elif not have_module("mlx_lm"):
        warn("lm mode needs mlx-lm")
        note(dim("  run: mlxsh setup"))
        return

    if not is_downloaded(repo):
        print(yellow(f"  {repo} is not downloaded"))
        if not pull(repo):
            return

    port = choose_port(repo, policy)
    if port is None:
        return

    # Only the server on the chosen port is replaced. Everything on other
    # ports keeps running.
    st = find_server(port)
    if st:
        if st["mode"] == mode and st["model"] == repo:
            print(green("  already running: ") + f"{mode} {short(repo)} on :{port}")
            return
        if not memory_check(repo, port, yes):
            return
        stop_one(st)
    elif port_open(port, host):
        pid = port_owner(port)
        if pid is None:
            warn(f"port {port} is in use by another process")
            print(dim("  pick another with --port N, or --port auto"))
            return
        print(yellow(f"  port {port} is used by pid {pid}"))
        print(dim(f"  {pid_command(pid)[:100]}"))
        if not confirm("  stop it?", False):
            return
        os.kill(pid, signal.SIGTERM)
        time.sleep(1.5)
    elif not memory_check(repo, port, yes):
        return

    py = python_bin()
    args = list(extra or reg["serve_args"].get(mode, []))
    module = ["-m", "mlx_lm", "server"] if mode == "lm" else ["-m", "mlx_vlm.server"]
    cmd = [py, *module, "--model", repo, "--host", host, "--port", str(port),
           "--log-level", "INFO", *args]

    engine = "mlx_lm" if mode == "lm" else "mlx_vlm"
    how = "text only, no vision tower" if mode == "lm" else "vision tower loaded"
    print(bold(f"  {mode} mode") + dim(f"   {engine}, {how}"))
    print(dim(f"  {repo}"))

    if foreground:
        BAR.stop()
        os.execv(py, cmd)

    ensure_home()
    log = log_path(port)
    log.parent.mkdir(parents=True, exist_ok=True)
    rotate_log(log)
    env = dict(os.environ)
    view = cache_view(port, repo) if setting("pin_model") else None
    if view:
        # only this model is visible to the server, and it cannot reach the
        # Hub to fetch another one
        env["HF_HUB_CACHE"] = str(view)
        env["HF_HUB_OFFLINE"] = "1"

    with log.open("a") as fh:
        fh.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} "
                 f"start {mode} {repo} ===\n")
        fh.flush()
        proc = subprocess.Popen(cmd, stdout=fh, stderr=fh, cwd=str(HOME),
                                start_new_session=True, env=env)

    write_state({"pid": proc.pid, "mode": mode, "model": repo, "host": host,
                 "port": port, "engine": engine, "log": str(log),
                 "started": time.time()})

    spin, t0 = "|/-\\", time.time()
    print(dim("  starting "), end="", flush=True)
    while time.time() - t0 < setting("start_timeout"):
        if proc.poll() is not None:
            print(red("\r  server exited during startup:") + " " * 20)
            print(dim(tail_log(15, log)))
            state_path(port).unlink(missing_ok=True)
            return
        if port_open(port, host):
            print(dim(f"\r  loading weights ({time.time() - t0:.0f}s) "),
                  end="", flush=True)
            ok = warmup(host, port, repo)
            rss, _ = proc_stats(proc.pid)
            if ok:
                print(green(f"\r  ready in {time.time() - t0:.0f}s") + " " * 24)
            else:
                print(yellow("\r  listening, warm-up request failed") + " " * 16)
            print(dim(f"  http://{host}:{port}/v1   {gb(rss)} resident"))
            return
        print(spin[int((time.time() - t0) * 4) % 4], end="\b", flush=True)
        time.sleep(0.25)
    print(yellow("\r  not listening yet, check the log"))


def tail_log(n: int = 40, path: Path | None = None) -> str:
    p = path or LOG
    if not p.exists():
        return "  no log yet"
    size = p.stat().st_size
    block = min(size, max(8192, n * 200))
    with p.open("rb") as fh:
        fh.seek(size - block)
        data = fh.read()
    lines = data.decode(errors="replace").splitlines()
    if block < size and lines:
        lines = lines[1:]
    return "\n".join(lines[-n:])


def server_line(st: dict, compact: bool = False,
                stats: dict | None = None) -> str:
    rss, up = (stats or {}).get(st["pid"]) or proc_stats(st["pid"])
    host = st.get("host", setting("host"))
    tag = green("lm    ") if st["mode"] == "lm" else blue("vision")
    line = (f"  {tag}  {bold(short(st.get('model') or '') or '?'):<34}"
            + dim(f"{gb(rss):>9}   up {up:<9} http://{host}:{st['port']}/v1"))
    if not compact and st.get("adopted"):
        line += dim("   started outside this shell")
    return line


def print_servers(servers: list[dict], compact: bool = False):
    stats = proc_stats_many([s["pid"] for s in servers])
    for st in servers:
        print(server_line(st, compact, stats))
    if len(servers) > 1:
        sizes = cache_sizes()
        rss = sum(r for r, _ in stats.values())
        weights = sum(max(stats.get(s["pid"], (0, ""))[0],
                          sizes.get(s.get("model") or "", 0))
                      for s in servers)
        print(dim(f"  {len(servers)} servers, {gb(rss)} resident now, about "
                  f"{gb(weights)} of weights, machine has {gb(MEM_TOTAL)}"))


def remote_json() -> dict:
    """The gateway and tunnel, for scripts."""
    gw, tun = gateway_state(), tunnel_state()
    return {
        "gateway": None if not gw else {
            "endpoint": f"http://{gw['host']}:{gw['port']}/v1",
            "port": gw["port"], "pid": gw["pid"]},
        "tunnel": None if not tun else {
            "endpoint": f"{tun.get('url') or 'https://' + tun['hostname']}/v1",
            "hostname": tun.get("hostname", ""), "pid": tun["pid"],
            "quick": bool(tun.get("quick"))},
    }


def servers_json() -> str:
    """Machine-readable server list. The model field is the full repo id,
    which is what the endpoint expects."""
    servers = list_servers()
    stats = proc_stats_many([s["pid"] for s in servers])
    rows = []
    for st in servers:
        rss, up = stats.get(st["pid"], (0, "?"))
        host = st.get("host", setting("host"))
        # mlx_lm maps this alias to whatever the server was started with, so
        # a client can pin to a port without naming a repo id. mlx_vlm has no
        # such alias and rejects it.
        alias = "default_model" if st.get("engine") == "mlx_lm" else None
        rows.append({"model": st.get("model") or "", "mode": st["mode"],
                     "alias": alias,
                     "engine": st.get("engine", ""), "host": host,
                     "port": st["port"], "pid": st["pid"],
                     "endpoint": f"http://{host}:{st['port']}/v1",
                     "resident_bytes": rss, "uptime": up,
                     "adopted": bool(st.get("adopted"))})
    return json.dumps(rows, indent=2)


def models_json(reg: dict) -> str:
    cached = cache_sizes()
    rows = [{"repo": m["repo"], "label": m.get("label") or short(m["repo"]),
             "vision": bool(m.get("vision")),
             "downloaded": is_downloaded(m["repo"]),
             "size_bytes": cached.get(m["repo"], 0),
             "speed": m.get("speed") or {}}
            for m in reg["models"]]
    return json.dumps({"models": rows, "defaults": reg["defaults"],
                       "endpoint": f"http://{setting('host')}:{setting('port')}/v1"},
                      indent=2)


def status(compact: bool = False):
    tun = tunnel_state()
    if tun:
        print(f"  {green('tunnel'):<9} {bold('public'):<32}"
              + dim(f"{tun.get('url') or 'https://' + tun['hostname']}/v1"))
    gw = gateway_state()
    if gw:
        print(f"  {green('gateway'):<9} {bold('authenticated'):<32}"
              + dim(f"http://{gw['host']}:{gw['port']}/v1"))
    servers = list_servers()
    if not servers:
        port = setting("port")
        busy = port_owner(port)
        if busy:
            print(yellow("  port ") + f"{port} is held by pid {busy}, "
                                      f"not an mlx server")
        else:
            print(dim(f"  stopped, port {port} free"))
        return
    print_servers(servers, compact)


def bench(tokens: int = 128, target: str | None = None, choose=None):
    servers = match_servers(target)
    if not servers:
        warn(f"no server matches {target!r}" if target else "no server running")
        return
    if len(servers) > 1:
        st = choose(servers) if choose else None
        if not st:
            if choose and supports_tui():
                return  # the picker was cancelled
            warn(f"{len(servers)} servers running, name one: bench <port|model>")
            print_servers(servers)
            return
    else:
        st = servers[0]
    host = st.get("host", setting("host"))
    body = json.dumps({
        "model": st["model"],
        "messages": [{"role": "user",
                      "content": "Write a short paragraph about unified memory."}],
        "max_tokens": tokens, "temperature": 0.7,
    }).encode()
    req = urllib.request.Request(f"http://{host}:{st['port']}/v1/chat/completions",
                                 data=body,
                                 headers={"Content-Type": "application/json"})
    print(dim(f"  benchmarking {st['mode']} {short(st['model'])}"))
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=setting("reply_timeout")) as r:
            out = json.loads(r.read())
    except (urllib.error.URLError, ConnectionError, TimeoutError,
            json.JSONDecodeError) as e:
        warn(f"request failed: {e}")
        return
    dt = time.time() - t0
    used = out.get("usage", {})
    n = used.get("completion_tokens") or tokens
    rate = n / dt
    print(f"  {bold(f'{rate:.1f} tok/s')}"
          + dim(f"   {n} tokens in {dt:.1f}s, prompt {used.get('prompt_tokens', '?')}"))

    reg = load_registry()
    entry = next((m for m in reg["models"] if m["repo"] == st["model"]), None)
    if entry is not None:
        entry.setdefault("speed", {})[st["mode"]] = round(rate, 1)
        save_registry(reg)


# ----------------------------------------------------------------- downloading


def pull(repo: str, yes: bool = False) -> bool:
    repo = repo.strip()
    if "/" not in repo:
        repo = f"{setting('org')}/{repo}"
    if is_downloaded(repo):
        print(green(f"  {repo} already downloaded"))
        register(repo)
        return True
    if not have_module("huggingface_hub"):
        warn("downloading needs huggingface_hub: pip install huggingface_hub")
        return False

    cfg = remote_config(repo)
    if cfg is None:
        warn(f"cannot read {repo}/config.json, check the id and your network")
        return False
    vision = is_vision_config(cfg)
    size = None
    try:
        from huggingface_hub import HfApi

        size = getattr(HfApi().model_info(repo, expand=["usedStorage"]),
                       "used_storage", None)
    except Exception:
        pass
    print(f"  {bold(repo)}")
    print(dim(f"  {cfg.get('model_type', '?')}, "
              f"{'vision' if vision else 'text only'}"
              + (f", {gb(size)} download" if size else "")))
    if not mlx_can_run(cfg):
        print(yellow(f"  warning: no MLX implementation for "
                     f"{cfg.get('model_type', '?')}"))
    if size and size > mem_tight():
        print(yellow(f"  warning: exceeds {gb(MEM_TOTAL)} of memory"))
    if not yes and not confirm("  download?", True):
        return False

    hf = shutil.which("hf") or shutil.which("huggingface-cli")
    if hf:
        cmd = [hf, "download", repo]
    else:
        cmd = [python_bin(), "-c",
               ("import sys;from huggingface_hub import snapshot_download;"
                "snapshot_download(sys.argv[1])"), repo]
    if subprocess.run(cmd).returncode != 0:
        warn("download failed")
        return False
    register(repo, vision)
    return True


def register(repo: str, vision: bool | None = None):
    reg = load_registry()
    if any(m["repo"] == repo for m in reg["models"]):
        return
    if vision is None:
        vision = is_vision_config(local_config(repo) or {})
    reg["models"].append({"repo": repo, "label": short(repo), "vision": vision,
                          "note": ""})
    save_registry(reg)
    print(green(f"  registered as {len(reg['models'])}"))
    before = dict(reg["defaults"])
    set_missing_defaults(reg)
    for mode in ("lm", "vision"):
        if reg["defaults"].get(mode) == repo and before.get(mode) != repo:
            print(green(f"  default {mode} is now {short(repo)}"))


def remove(needle: str, yes: bool = False):
    reg = load_registry()
    entry = resolve(reg, needle)
    repo = entry["repo"] if entry else needle
    sizes = cache_sizes()
    if repo not in sizes:
        warn(f"{repo} is not in the cache")
        return
    if not yes and not confirm(red(f"  delete {repo} ({gb(sizes[repo])})?")):
        return
    from huggingface_hub import scan_cache_dir

    info = scan_cache_dir()
    revs = [rev.commit_hash for r in info.repos if r.repo_id == repo
            for rev in r.revisions]
    info.delete_revisions(*revs).execute()
    reg["models"] = [m for m in reg["models"] if m["repo"] != repo]
    for mode, d in list(reg["defaults"].items()):
        if d == repo:
            reg["defaults"][mode] = ""
    save_registry(reg)
    set_missing_defaults(reg)
    print(green(f"  deleted {repo}"))


# -------------------------------------------------------------------- listings


def print_models(reg: dict):
    cached = cache_sizes()
    if not reg["models"]:
        print(dim("  no models registered, run browse"))
        return
    for i, m in enumerate(reg["models"], 1):
        repo = m["repo"]
        sz = cached.get(repo) if is_downloaded(repo) else None
        state = dim(f"{gb(sz)}") if sz else yellow("not downloaded")
        caps = cyan("vision") if m.get("vision") else dim("text  ")
        tags = " ".join(green(f"({md} default)") for md in ("lm", "vision")
                        if reg["defaults"].get(md) == repo)
        speed = "  ".join(f"{v:g} tok/s {k}"
                          for k, v in (m.get("speed") or {}).items())
        print(f"  {bold(f'{i:>2}')}  {caps}  {state:>16}  "
              f"{m.get('label') or short(repo)} {tags}".rstrip())
        print(dim(f"      {repo}") + (dim(f"   {speed}") if speed else ""))
        if m.get("note"):
            print(dim(f"      {m['note']}"))
    total = sum(cached.get(m["repo"], 0) for m in reg["models"])
    print(dim(f"\n  {len(reg['models'])} models, {gb(total)} on disk, defaults "
              f"lm={short(reg['defaults'].get('lm') or '-')} "
              f"vision={short(reg['defaults'].get('vision') or '-')}"))


def print_hub(rows: list[dict], heading: str, show_fit: bool = False):
    if not rows:
        print(dim("  nothing matched"))
        return
    print(bold(f"  {heading}"))
    for i, r in enumerate(rows, 1):
        have = r["have"] if "have" in r else is_downloaded(r["repo"])
        caps = cyan("vision") if r["vision"] else dim("text  ")
        age = r["updated"].strftime("%Y-%m") if r.get("updated") else "     "
        print(f"  {bold(f'{i:>2}')}{green(' v') if have else '  '} {caps} "
              f"{size_colour(r['size'])}"
              + (f" {fit_mark(r['size'])}" if show_fit else "")
              + f"  {r['repo'][:56]}" + dim(f"   {r['downloads']:,} downloads, {age}")
              + (green("  downloaded") if have else ""))
    print(dim("\n  sizes are estimated memory use, get <n> downloads"))


def render_model(m: dict, reg: dict, cached: dict,
                 have: dict | None = None) -> str:
    repo = m["repo"]
    downloaded = have.get(repo) if have is not None else is_downloaded(repo)
    sz = cached.get(repo) if downloaded else None
    caps = cyan("vision") if m.get("vision") else dim("text  ")
    size = dim(f"{gb(sz):>9}") if sz else yellow("  no files")
    tags = "".join(green(f" ({md} default)") for md in ("lm", "vision")
                   if reg["defaults"].get(md) == repo)
    speed = "  ".join(f"{v:g} {k}" for k, v in (m.get("speed") or {}).items())
    return (f"{caps} {size}  {(m.get('label') or short(repo))[:44]:44}{tags}"
            + (dim(f"  {speed} tok/s") if speed else ""))


def render_hub(r: dict) -> str:
    have = r["have"] if "have" in r else is_downloaded(r["repo"])
    caps = cyan("vision") if r["vision"] else dim("text  ")
    age = r["updated"].strftime("%Y-%m") if r.get("updated") else ""
    return (f"{caps} {size_colour(r['size'])}  {r['repo'][:52]:52}"
            + dim(f"  {r['downloads']:,}, {age}")
            + (green("  downloaded") if have else ""))


def choose_model(reg: dict, mode: str | None = None, title: str = "",
                 multi: bool = False):
    rows = [m for m in reg["models"]
            if not (mode == "vision" and not m.get("vision"))]
    if not rows:
        warn("no models registered, run browse")
        return None
    cached = cache_sizes()
    have = {m["repo"]: is_downloaded(m["repo"]) for m in rows}
    default = reg["defaults"].get(mode or "lm")
    idx = next((i for i, m in enumerate(rows) if m["repo"] == default), 0)
    return pick(rows, title=title or f"model for {mode or 'lm'} mode",
                render=lambda m: render_model(m, reg, cached, have),
                key=lambda m: f"{m['repo']} {m.get('label', '')} {m.get('note', '')}",
                index=idx, multi=multi)


def choose_mode(title: str = "mode") -> str | None:
    opts = [("lm", f"{green('lm')}      {dim('mlx_lm, text only')}"),
            ("vision", f"{blue('vision')}  {dim('mlx_vlm, multimodal')}")]
    got = pick(opts, title=title, render=lambda o: o[1], key=lambda o: o[0])
    return got[0] if got else None


# ------------------------------------------------------------------ chat, ask


def have_module(name: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def target_server(target: str | None = None, choose=None,
                  verb: str = "use") -> dict | None:
    """The server a command should talk to, out of everything running."""
    servers = match_servers(target)
    if not servers:
        if target:
            warn(f"no running server matches {target!r}")
        return None
    if len(servers) == 1:
        return servers[0]
    st = choose(servers) if choose else None
    if st:
        return st
    if choose and supports_tui():
        return None  # the picker was cancelled
    warn(f"{len(servers)} servers running, name one: {verb} --on <port|model>")
    print_servers(servers)
    return None


def arg_value(args: list[str], flag: str, fallback: int) -> int:
    if flag in args:
        try:
            return int(args[args.index(flag) + 1])
        except (IndexError, ValueError):
            pass
    return fallback


def image_payload(path: str) -> dict:
    import base64
    import mimetypes

    data = Path(path).expanduser().read_bytes()
    mime = mimetypes.guess_type(path)[0] or "image/png"
    return {"type": "image_url",
            "image_url": {"url": f"data:{mime};base64,"
                                 + base64.b64encode(data).decode()}}


def api_stream(st: dict, messages: list[dict], max_tokens: int = 2048,
               temperature: float = 0.7) -> str | None:
    """Stream a completion from a running server, printing as it arrives."""
    host = st.get("host", setting("host"))
    body = json.dumps({"model": st.get("model") or "", "messages": messages,
                       "max_tokens": max_tokens, "temperature": temperature,
                       "stream": True}).encode()
    req = urllib.request.Request(f"http://{host}:{st['port']}/v1/chat/completions",
                                 data=body,
                                 headers={"Content-Type": "application/json"})
    out: list[str] = []
    thinking = False
    try:
        with urllib.request.urlopen(req, timeout=setting("reply_timeout")) as r:
            for raw in r:
                line = raw.decode(errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                delta = (chunk.get("choices") or [{}])[0].get("delta", {})
                # Reasoning models send their chain of thought with no content
                # at all for a while. Show it on a terminal so the wait is
                # visible, but keep it out of piped output and out of the
                # history a chat builds up.
                if delta.get("reasoning"):
                    thinking = True
                    if TTY:
                        print(dim(delta["reasoning"]), end="", flush=True)
                if delta.get("content"):
                    if thinking:
                        print()
                        thinking = False
                    out.append(delta["content"])
                    print(delta["content"], end="", flush=True)
    except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
        warn(f"request failed: {e}")
        return None
    except KeyboardInterrupt:
        print(dim("\n  stopped"))
    print()
    if not out and thinking:
        warn("the model spent its whole budget reasoning, no answer came back")
        note(dim("  raise it with: config ask_args, or ask a simpler question"))
    return "".join(out)


def ask_server(st: dict, prompt: str, images: list[str]) -> bool:
    if images and st.get("mode") != "vision":
        warn(f"the server on :{st['port']} is lm mode, images need a vision one")
        note(dim("  start one with: vision <model>, or use --load"))
        return False
    content = prompt if not images else \
        [{"type": "text", "text": prompt}, *[image_payload(i) for i in images]]
    if TTY:  # piped output is the answer and nothing else
        print(dim(f"  {short(st.get('model') or '?')} on :{st['port']}"))
    return api_stream(st, [{"role": "user", "content": content}],
                      arg_value(setting_list("ask_args"), "--max-tokens",
                                2048)) is not None


def chat_server(st: dict):
    """A chat loop against a running server, so no second copy is loaded."""
    print(dim(f"  {short(st.get('model') or '?')} on :{st['port']}, "
              f"/reset clears the history, /exit leaves"))
    limit = arg_value(setting_list("chat_args"), "--max-tokens", 4096)
    messages: list[dict] = []
    while True:
        try:
            line = input(cyan("you> ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        if line in ("/exit", "/quit", "/bye"):
            return
        if line == "/reset":
            messages.clear()
            print(dim("  history cleared"))
            continue
        messages.append({"role": "user", "content": line})
        reply = api_stream(st, messages, limit)
        if reply is None:
            return
        messages.append({"role": "assistant", "content": reply})


def extract_on(args: list[str]) -> tuple[list[str], str | None, bool]:
    """Pull --on <server> and --load out of an argument list."""
    rest, target, load, i = [], None, False, 0
    while i < len(args):
        a = args[i]
        if a == "--on" and i + 1 < len(args):
            target = args[i + 1]
            i += 2
            continue
        if a.startswith("--on="):
            target = a.split("=", 1)[1]
            i += 1
            continue
        if a in ("--load", "--own"):
            load = True
            i += 1
            continue
        rest.append(a)
        i += 1
    return rest, target, load


def run_chat(mode: str, repo: str | None = None, replace: bool = True):
    reg = load_registry()
    if repo:
        e = find_model(reg, repo)
        repo = e["repo"] if e else repo
    else:
        repo = reg["defaults"].get(mode)
    if not repo:
        warn("no model selected")
        return
    py = python_bin()
    module = ["-m", "mlx_lm", "chat"] if mode == "lm" else ["-m", "mlx_vlm.chat"]
    cmd = [py, *module, "--model", repo, *setting_list("chat_args")]
    print(dim(f"  {'mlx_lm' if mode == 'lm' else 'mlx_vlm'} {short(repo)}, "
              f"loaded separately from the server"))
    if replace:
        BAR.stop()
        os.execv(py, cmd)
    subprocess.call(cmd)


def run_ask(prompt: str, images: list[str], mode: str | None = None,
            replace: bool = True):
    reg = load_registry()
    mode = mode or ("vision" if images else "lm")
    repo = reg["defaults"].get(mode)
    if not repo:
        warn(f"no default model for {mode} mode")
        return
    py = python_bin()
    module = ["-m", "mlx_lm", "generate"] if mode == "lm" \
        else ["-m", "mlx_vlm.generate"]
    print(dim(f"  loading {short(repo)}, no server to ask"))
    cmd = [py, *module, "--model", repo, "--prompt", prompt,
           *setting_list("ask_args")]
    for img in images:
        cmd += ["--image", img]
    if replace:
        BAR.stop()
        os.execv(py, cmd)
    subprocess.call(cmd)


def setting_list(key: str) -> list[str]:
    return list(load_registry().get(key) or DEFAULTS[key])


# --------------------------------------------------------------------- gateway
#
# The model servers have no authentication, which is why they only ever listen
# on localhost. The gateway is the authenticated front door: it checks a bearer
# key, then forwards to whichever loaded model the request asked for. Anything
# reaching this machine from outside goes through here, never straight to a
# model server.

MAX_BODY = 32 * 1024 * 1024
HOP_BY_HOP = {"connection", "keep-alive", "transfer-encoding", "upgrade",
              "proxy-authenticate", "proxy-authorization", "te", "trailer"}


def key_path() -> Path:
    return HOME / "api.key"


def api_key(new: bool = False) -> str:
    """The bearer key, created on first use."""
    import secrets

    from_env = os.environ.get("MLXSH_API_KEY")
    if from_env and not new:
        return from_env
    ensure_home()
    path = key_path()
    if new or not path.exists():
        path.write_text("mlxsh-" + secrets.token_urlsafe(32) + "\n")
        path.chmod(0o600)
    return path.read_text().strip()


def key_matches(header: str) -> bool:
    import hmac

    sent = header[7:].strip() if header[:7].lower() == "bearer " else ""
    return bool(sent) and hmac.compare_digest(sent, api_key())


def gateway_target(body: bytes) -> tuple[dict | None, bytes]:
    """The server a request is for, and the body with its model corrected."""
    servers = list_servers()
    if not servers:
        return None, body
    try:
        payload = json.loads(body or b"{}")
    except json.JSONDecodeError:
        return (servers[0] if len(servers) == 1 else None), body
    if not isinstance(payload, dict):
        return None, body

    wanted = str(payload.get("model") or "").strip()
    if not wanted or wanted == "default_model":
        chosen = servers[0]
    else:
        low = wanted.lower()
        chosen = (next((s for s in servers if s.get("model") == wanted), None)
                  or next((s for s in servers
                           if low in (s.get("model") or "").lower()
                           or low == s.get("mode")), None))
    if not chosen:
        return None, body
    # upstream wants its own repo id, whatever name the client used
    payload["model"] = chosen.get("model") or wanted
    return chosen, json.dumps(payload).encode()


def gateway_models() -> dict:
    return {"object": "list", "data": [
        {"id": s.get("model") or "", "object": "model",
         "owned_by": s.get("engine", "mlx"),
         "created": int(s.get("started", 0))} for s in list_servers()]}


def gateway_handler():
    from http.server import BaseHTTPRequestHandler

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = f"mlxsh/{__version__}"

        def log_message(self, fmt, *args):
            sys.stderr.write(f"{self.address_string()} {fmt % args}\n")

        def reply(self, code: int, payload: dict):
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def refuse(self, code: int, message: str):
            self.reply(code, {"error": {"message": message,
                                        "type": "invalid_request_error"}})

        def do_GET(self):
            self.route("GET")

        def do_POST(self):
            self.route("POST")

        def route(self, method: str):
            if self.path.rstrip("/").endswith(("/healthz", "/health")):
                return self.reply(200, {"status": "ok"})
            if not key_matches(self.headers.get("Authorization", "")):
                return self.refuse(401, "missing or invalid api key")
            if self.path.rstrip("/").endswith("/models"):
                return self.reply(200, gateway_models())

            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                return self.refuse(413, "request too large")
            body = self.rfile.read(length) if length else b""
            target, body = gateway_target(body)
            if not target:
                loaded = [s.get("model") for s in list_servers()]
                return self.refuse(404, "no model is loaded" if not loaded else
                                   f"no loaded model matches, try one of {loaded}")
            self.forward(method, target, body)

        def forward(self, method: str, target: dict, body: bytes):
            host = target.get("host", setting("host"))
            url = f"http://{host}:{target['port']}{self.path}"
            headers = {k: v for k, v in self.headers.items()
                       if k.lower() not in HOP_BY_HOP
                       and k.lower() not in ("authorization", "host",
                                             "content-length")}
            req = urllib.request.Request(url, data=body or None,
                                         headers=headers, method=method)
            try:
                upstream = urllib.request.urlopen(
                    req, timeout=setting("reply_timeout"))
            except urllib.error.HTTPError as e:
                return self.refuse(e.code,
                                   e.read().decode(errors="replace")[:400])
            except Exception as e:
                return self.refuse(502, f"{target.get('model')} did not answer: {e}")

            with upstream:
                size = upstream.headers.get("Content-Length")
                self.send_response(upstream.status)
                self.send_header("Content-Type",
                                 upstream.headers.get("Content-Type",
                                                      "application/json"))
                if size:
                    self.send_header("Content-Length", size)
                else:  # streaming: chunk it so tokens arrive as produced
                    self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                try:
                    while True:
                        chunk = upstream.read(4096)
                        if not chunk:
                            break
                        self.wfile.write(chunk if size else
                                         b"%x\r\n%s\r\n" % (len(chunk), chunk))
                        self.wfile.flush()
                    if not size:
                        self.wfile.write(b"0\r\n\r\n")
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass

    return Handler


def run_gateway(host: str, port: int):
    """Blocking. Started as a child process by the gateway command."""
    from http.server import ThreadingHTTPServer

    server = ThreadingHTTPServer((host, port), gateway_handler())
    server.daemon_threads = True
    server.serve_forever()


def gateway_file() -> Path:
    return HOME / "gateway.json"


def gateway_state() -> dict | None:
    path = gateway_file()
    if not path.exists():
        return None
    try:
        st = json.loads(path.read_text())
    except Exception:
        path.unlink(missing_ok=True)
        return None
    if pid_alive(st.get("pid", -1)) and "mlxsh" in pid_command(st["pid"]):
        return st
    path.unlink(missing_ok=True)
    return None


def start_gateway(host: str | None = None, port: int | None = None,
                  foreground: bool = False, expose: bool = False):
    running = gateway_state()
    if running:
        print(green("  already running: ")
              + f"http://{running['host']}:{running['port']}/v1")
        return
    host = host or setting("host")
    port = int(port or setting("gateway_port"))
    if host not in ("127.0.0.1", "localhost", "::1") and not expose:
        warn(f"binding {host} would put the endpoint on your network")
        note(dim("  the key would cross it in cleartext, since the gateway "
                 "has no TLS"))
        note(dim("  pass --expose if you meant it, or put a tunnel in front"))
        return
    if port_open(port, host):
        warn(f"port {port} is already in use")
        return

    key = api_key()
    if foreground:
        print(dim(f"  gateway on http://{host}:{port}/v1"))
        run_gateway(host, port)
        return

    ensure_home()
    log = log_path("gateway")
    log.parent.mkdir(parents=True, exist_ok=True)
    rotate_log(log)
    with log.open("a") as fh:
        fh.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} start gateway ===\n")
        fh.flush()
        proc = subprocess.Popen(
            [python_bin(), str(SELF), "--run-gateway", host, str(port)],
            stdout=fh, stderr=fh, cwd=str(HOME), start_new_session=True)
    gateway_file().write_text(json.dumps(
        {"pid": proc.pid, "host": host, "port": port,
         "started": time.time()}, indent=2) + "\n")

    for _ in range(40):
        if port_open(port, host):
            print(bold("  gateway") + dim(f"   http://{host}:{port}/v1"))
            print(dim(f"  key       {key}"))
            print(dim("  clients send: Authorization: Bearer <key>"))
            for line in tunnel_hint():
                print(dim(line))
            return
        if proc.poll() is not None:
            warn("the gateway exited at startup")
            note(dim(tail_log(10, log)))
            gateway_file().unlink(missing_ok=True)
            return
        time.sleep(0.25)
    warn("the gateway is not listening yet, check: log gateway")


def stop_gateway(quiet: bool = False):
    st = gateway_state()
    if not st:
        if not quiet:
            print(dim("  no gateway running"))
        return
    try:
        os.killpg(os.getpgid(st["pid"]), signal.SIGTERM)
    except OSError:
        try:
            os.kill(st["pid"], signal.SIGTERM)
        except OSError:
            pass
    gateway_file().unlink(missing_ok=True)
    print(green("  gateway stopped"))


# ---------------------------------------------------------------------- tunnel
#
# A named Cloudflare tunnel gives the machine a hostname that never changes,
# and every step of setting one up is a command, so mlxsh can run them. It
# supervises cloudflared the way it supervises a model server, and points it at
# the gateway rather than at a model, so nothing unauthenticated is ever
# exposed.


CLOUDFLARED_DOCS = ("https://developers.cloudflare.com/cloudflare-one/"
                    "connections/connect-networks/downloads/")


def cloudflared() -> str | None:
    return shutil.which("cloudflared")


def tunnel_hint() -> list[str]:
    """What to say next about reaching this machine from elsewhere."""
    if cloudflared():
        return ["  from another machine: mlxsh tunnel --quick",
                "  or a permanent address: mlxsh tunnel setup <hostname>"]
    return ["  to reach it from another machine you also need cloudflared",
            f"  {CLOUDFLARED_DOCS}",
            "  then: mlxsh tunnel --quick"]


def tunnel_file() -> Path:
    return HOME / "tunnel.json"


def tunnel_state() -> dict | None:
    path = tunnel_file()
    if not path.exists():
        return None
    try:
        st = json.loads(path.read_text())
    except Exception:
        path.unlink(missing_ok=True)
        return None
    if pid_alive(st.get("pid", -1)) and "cloudflared" in pid_command(st["pid"]):
        return st
    path.unlink(missing_ok=True)
    return None


def tunnel_commands(name: str, hostname: str, port: int,
                    logged_in: bool, exists: bool) -> list[list[str]]:
    """The setup steps still needed, in order. Empty when nothing is left."""
    exe = cloudflared() or "cloudflared"
    steps = []
    if not logged_in:
        steps.append([exe, "tunnel", "login"])
    if not exists:
        steps.append([exe, "tunnel", "create", name])
    steps.append([exe, "tunnel", "route", "dns", "--overwrite-dns",
                  name, hostname])
    return steps


def tunnel_run_command(name: str, port: int, quick: bool = False) -> list[str]:
    custom = load_registry().get("tunnel_cmd") or ""
    if custom and not quick:
        return shlex.split(custom.format(port=port))
    exe = cloudflared() or "cloudflared"
    if quick:  # no account, no domain, a new name every time
        return [exe, "tunnel", "--url", f"http://127.0.0.1:{port}"]
    return [exe, "tunnel", "run", "--url", f"http://127.0.0.1:{port}", name]


PUBLIC_URL = re.compile(r"https://[\w.-]+\.(?:trycloudflare\.com|ts\.net)\S*")


def url_from_log(text: str) -> str | None:
    """Quick tunnels and some other providers only announce their address in
    their own output."""
    found = PUBLIC_URL.findall(text)
    return found[-1] if found else None


def log_since(path: Path, offset: int) -> str:
    """Only what this run wrote. The log keeps earlier runs, and an old address
    in the tail would be read as the new one."""
    if not path.exists():
        return ""
    with path.open("rb") as fh:
        fh.seek(offset)
        return fh.read().decode(errors="replace")


def tunnel_exists(name: str) -> bool:
    exe = cloudflared()
    if not exe:
        return False
    try:
        out = subprocess.run([exe, "tunnel", "list"], capture_output=True,
                             text=True, timeout=30).stdout
    except Exception:
        return False
    return any(line.split()[1:2] == [name] for line in out.splitlines()[1:]
               if line.split())


def tunnel_setup(hostname: str):
    if not cloudflared():
        warn("cloudflared is not installed")
        note(dim(f"  {CLOUDFLARED_DOCS}"))
        return
    if not hostname or "." not in hostname:
        warn("give the hostname you want, for example: tunnel setup llm.example.com")
        return
    reg = load_registry()
    name = reg.get("tunnel_name") or DEFAULTS["tunnel_name"]
    logged_in = (Path.home() / ".cloudflared" / "cert.pem").exists()
    steps = tunnel_commands(name, hostname, setting("gateway_port"),
                            logged_in, tunnel_exists(name))
    for cmd in steps:
        print(dim("  " + " ".join(cmd)))
        result = subprocess.run(cmd)
        if result.returncode != 0:
            # an existing DNS record is not a failure worth stopping for
            if cmd[2:4] == ["route", "dns"]:
                note(dim("  the record may already point here, carrying on"))
                continue
            warn(f"failed: {' '.join(cmd)}")
            return
    reg = load_registry()
    reg["tunnel_hostname"] = hostname
    save_registry(reg)
    print(green(f"  {hostname} points at this machine"))
    print(dim("  start it with: mlxsh tunnel"))


def start_tunnel(foreground: bool = False, quick: bool = False):
    if tunnel_state():
        st = tunnel_state()
        print(green("  already running: ") + f"{st['url']}/v1")
        return
    reg = load_registry()
    hostname = "" if quick else setting("tunnel_hostname")
    if not hostname and not quick and not reg.get("tunnel_cmd"):
        warn("no hostname yet")
        note(dim("  run: mlxsh tunnel setup <hostname>"))
        note(dim("  or, for a throwaway address: mlxsh tunnel --quick"))
        return
    if not cloudflared() and not reg.get("tunnel_cmd"):
        warn("cloudflared is not installed")
        note(dim(f"  {CLOUDFLARED_DOCS}"))
        return

    if not gateway_state():
        print(dim("  starting the gateway first, so nothing is exposed "
                  "without a key"))
        start_gateway()
        if not gateway_state():
            return
    port = int(gateway_state()["port"])
    cmd = tunnel_run_command(reg.get("tunnel_name") or DEFAULTS["tunnel_name"],
                             port, quick)
    if foreground:
        os.execvp(cmd[0], cmd)

    ensure_home()
    log = log_path("tunnel")
    log.parent.mkdir(parents=True, exist_ok=True)
    rotate_log(log)
    with log.open("a") as fh:
        fh.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} start tunnel ===\n")
        fh.flush()
        start_at = fh.tell()
        proc = subprocess.Popen(cmd, stdout=fh, stderr=fh, cwd=str(HOME),
                                start_new_session=True)
    url = f"https://{hostname}" if hostname else ""
    for _ in range(40):   # a quick tunnel only announces its address in the log
        time.sleep(0.5)
        if proc.poll() is not None:
            warn("the tunnel exited at startup")
            note(dim(tail_log(15, log)))
            tunnel_file().unlink(missing_ok=True)
            return
        if url:
            break
        url = url_from_log(log_since(log, start_at)) or ""
        if url:
            break
    if not url:
        warn("the tunnel started but announced no address, check: log tunnel")
        url = "unknown"

    tunnel_file().write_text(json.dumps(
        {"pid": proc.pid, "hostname": hostname, "url": url, "port": port,
         "quick": quick, "started": time.time()}, indent=2) + "\n")
    print(bold("  tunnel") + dim(f"   {url}/v1"))
    print(dim(f"  key      {api_key()}"))
    print(yellow("  this endpoint is now reachable from the internet, "
                 "and the key is the only lock"))
    if quick:
        print(yellow("  Cloudflare does not support server-sent events on a "
                     "quick tunnel, so treat streaming as unreliable"))
        note(dim("  the address also changes every restart. For a permanent "
                 "one: tunnel setup <hostname>"))


def stop_tunnel(quiet: bool = False):
    st = tunnel_state()
    if not st:
        if not quiet:
            print(dim("  no tunnel running"))
        return
    try:
        os.killpg(os.getpgid(st["pid"]), signal.SIGTERM)
    except OSError:
        try:
            os.kill(st["pid"], signal.SIGTERM)
        except OSError:
            pass
    tunnel_file().unlink(missing_ok=True)
    tail = (f"   {st['hostname']} still points here" if st.get("hostname")
            else "")
    print(green("  tunnel stopped") + dim(tail))


# -------------------------------------------------------------------- commands


def help_text() -> str:
    return f"""
  {bold('mlxsh')} {dim(f'{__version__}, local MLX models on {machine_name()}')}

  {bold('usage')}
    mlxsh                    this help
    mlxsh shell              this help, then the interactive shell
    mlxsh <command> ...      run one command and exit
    mlxsh browse|models|get  open the picker, then stay in the shell

  {bold('serving')}
    lm [model]               serve text only (mlx_lm)
    vision [model]           serve multimodal (mlx_vlm)
    stop [port|model|all]    stop a server and free the memory
    status [--json]          what is running
    bench [port|model] [n]   measure tok/s, saved to the registry
    log [port|model] [n]     tail one server's log

    --new                    force the next free port
    --replace                force reuse of the configured port
    --port N, --host ADDR    override for one run
    --foreground             run attached to this terminal
    -y                       skip the memory warning when adding a server

  A model per port. Nothing running is stopped: a new model takes the next
  free port. "config when_busy replace" restores the old behaviour of reusing
  the configured port, "ask" prompts each time. Loading a model already
  running in the other mode swaps the engine in place.

  {bold('models')}
    models                   pick a model, then pick an action
    ls [--json]              the same list as plain text
    use [model] [mode]       set the default for lm or vision
    rm [model] [-y]          delete from disk
    edit                     open the registry in $EDITOR

  {bold('finding models')}
    browse [filters]         live list from hugging face, space marks downloads
                             filters: vision, text, trending, popular, new, all
                             any other word searches, "org/" scopes to an org
    get [n|repo] [-y]        download from the last browse list, or by repo id

  {bold('prompting')}
    ask <text>               single prompt, --image PATH sends a picture
    chat                     a chat loop
    --on <port|model>        which running model answers
    --load                   load a private copy instead of using a server

  Both talk to a running model when there is one, so nothing extra is loaded.
  With several running, name one with --on, or pick from the list.

  {bold('settings')}
    config                   list settings and where each value comes from
    config <key> <value>     change one, for example: config port 8080
    config reset <key>       back to the default
                             config status_bar off hides the live line at the
                             top of the shell
    doctor                   versions, paths, machine
    setup                    install the MLX packages where mlxsh runs

  {bold('reaching it from elsewhere')}
    needs cloudflared, see
    https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/

    gateway                  one authenticated endpoint in front of every
                             loaded model. Prints the bearer key
    gateway key [--new]      show or rotate the key
    tunnel setup <hostname>  point a hostname at this machine, once
    tunnel                   start the tunnel, and the gateway if needed
    tunnel --quick           a throwaway address, no account or domain, but
                             no streaming and a new name each time
    tunnel url               print the public address, for scripts
    gateway stop, tunnel stop

  {bold('the shell')}
    Every command works either way. The shell adds the things that only make
    sense while you sit there: pickers when you leave an argument off, a
    command list on an empty line, tab completion, history, a live line at the
    top showing what is loaded, and a chat loop. One-shot runs never open a
    picker, so scripts stay predictable.

    In a picker: up/down or j/k move, pgup pgdn home end jump, type to filter,
    enter selects, space marks in multi-select, esc clears the filter or cancels

  {bold('anywhere')}
    <command> -h             what that command does, with examples
    model can be a repo id, a number from ls, or a unique substring
    errors go to stderr and exit non-zero, output goes to stdout

  {dim('https://github.com/ansnadeem/mlxsh')}
"""


def doctor():
    import importlib.metadata as md

    print()
    print(bold("  machine   ") + machine_name())
    old = sys.version_info < MIN_ENGINE_PYTHON
    print(bold("  python    ") + f"{platform.python_version()}  {sys.executable}"
          + (yellow("   too old for mlx, needs 3.10+") if old else ""))
    for pkg, label in (("mlx", "mlx"), ("mlx-lm", "mlx-lm"),
                       ("mlx-vlm", "mlx-vlm"), ("huggingface-hub", "hf-hub")):
        try:
            print(bold(f"  {label:<10}") + md.version(pkg))
        except Exception:
            print(bold(f"  {label:<10}") + red("not installed"))
    reg = load_registry()
    print(bold("  memory    ") + f"{gb(MEM_TOTAL)} total, comfortable to "
          f"{gb(mem_comfy())}, limit {gb(mem_tight())}")
    print(bold("  registry  ") + f"{REGISTRY}  ({len(reg['models'])} models)")
    print(bold("  cache     ") + f"{hub_cache()}  ({gb(sum(cache_sizes().values()))})")
    print(bold("  logs      ") + str(HOME / "logs"))
    print(bold("  endpoint  ") + f"http://{setting('host')}:{setting('port')}/v1")
    print()
    status()
    print()


def fmt_setting(value) -> str:
    if isinstance(value, bool):
        return "on" if value else "off"
    return str(value)


def print_settings():
    print()
    for key, (_env, _kind, blurb) in SETTINGS.items():
        value, src = setting_with_source(key)
        print("  " + bold(f"{key:<14}") + f"{fmt_setting(value):<18}"
              + dim(f"{src:<10}  {blurb}"))
    print(dim(f"\n  set with: config <key> <value>, or {SETTINGS['port'][0]}"
              f"-style env vars\n"))


def set_override(key: str, raw: str):
    if key == "port" and raw in ("auto", "next", "new"):
        OVERRIDES.pop("port", None)
        AUTO_PORT.add(True)
        return
    try:
        OVERRIDES[key] = SETTINGS[key][1](raw)
    except ValueError:
        warn(f"{raw!r} is not a valid {key}")


def split_flags(args: list[str]) -> list[str]:
    """Pull --host, --port and --new out of an argument list into OVERRIDES."""
    rest, i = [], 0
    while i < len(args):
        a = args[i]
        if a in ("--new", "--also"):
            AUTO_PORT.add(True)
            i += 1
            continue
        if a in ("--replace", "--swap"):
            OVERRIDES["when_busy"] = "replace"
            i += 1
            continue
        if a in ("--port", "--host") and i + 1 < len(args):
            set_override(a[2:], args[i + 1])
            i += 2
            continue
        if a.startswith(("--port=", "--host=")):
            key, raw = a[2:].split("=", 1)
            set_override(key, raw)
            i += 1
            continue
        rest.append(a)
        i += 1
    return rest


class Ctl:
    def __init__(self, tui: bool = False):
        # The cache scan is deferred: commands that only touch a server should
        # not pay for walking a cache that can hold hundreds of repos.
        self.reg = load_registry()
        self.hub: list[dict] = []
        self.tui = tui and supports_tui()

    def sync(self) -> dict:
        """Reload the registry and pick up anything new in the cache."""
        self.reg = adopt_cached_models(load_registry(refresh=True))
        return self.reg

    def do_help(self, a):
        cmd = ALIASES.get(a[0].lower(), a[0].lower()) if a else None
        if cmd in COMMAND_HELP:
            command_help(cmd)
        elif cmd:
            warn(f"no command {a[0]!r}")
            note(dim("  run help for the list"))
        else:
            print(help_text())

    def do_status(self, a):
        print(servers_json()) if "--json" in a else status()
    def do_doctor(self, a): doctor()

    def do_gateway(self, a):
        a = split_flags(a)
        if a and a[0] in ("stop", "off"):
            return stop_gateway()
        if a and a[0] == "key":
            print(api_key(new="--new" in a))
            return
        start_gateway(OVERRIDES.get("host"), OVERRIDES.get("port"),
                      foreground="--foreground" in a,
                      expose="--expose" in a)

    def do_tunnel(self, a):
        a = split_flags(a)
        if a and a[0] in ("stop", "off"):
            return stop_tunnel()
        if a and a[0] == "url":
            st = tunnel_state()
            if not st:
                warn("no tunnel running")
                return
            print(st.get("url") or f"https://{st['hostname']}")
            return
        if a and a[0] == "setup":
            return tunnel_setup(a[1] if len(a) > 1
                                else setting("tunnel_hostname"))
        start_tunnel(foreground="--foreground" in a,
                     quick="--quick" in a or (a and a[0] == "quick"))

    def do_setup(self, a):
        setup(yes="-y" in a or "--yes" in a)
    def do_log(self, a):
        a, target, _ = extract_on(a)
        lines = next((int(x) for x in a if x.isdigit() and len(x) < 5), 40)
        rest = [x for x in a if not (x.isdigit() and len(x) < 5)]
        target = target or (rest[0] if rest else None)
        servers = list_servers()
        path = None
        if servers:
            st = target_server(target, lambda s: self.choose_server(
                s, "log of which server?"), "log")
            if not st:
                return
            path = Path(st["log"]) if st.get("log") else log_path(st["port"])
            print(dim(f"  {short(st.get('model') or '?')} on :{st['port']}"))
        elif target:
            path = log_path(int(target)) if target.isdigit() else None
        print(tail_log(lines, path))

    def choose_server(self, servers, title="which server?"):
        if not self.tui:
            return None
        return pick(servers, title=title, render=lambda s: server_line(s, True),
                    key=lambda s: f"{s['port']} {s.get('model', '')} {s['mode']}")

    def do_stop(self, a):
        target = a[0] if a else None
        stop_server(target, choose=lambda servers: self.choose_server(
            servers, "stop which server?"))

    def do_bench(self, a):
        ports = {str(s.get("port")) for s in list_servers()}
        target = next((x for x in a if x in ports or not x.isdigit()), None)
        tokens = next((int(x) for x in a if x.isdigit() and x not in ports), 128)
        bench(tokens, target, choose=lambda servers: self.choose_server(
            servers, "benchmark which server?"))

    def _serve(self, mode: str, a: list[str]):
        a = split_flags(a)
        model = a[0] if a and not a[0].startswith("-") else None
        if model is None and self.tui:
            self.reg = load_registry(refresh=True)
            entry = choose_model(self.reg, mode, f"{mode} mode, pick a model")
            if not entry:
                return
            model = entry["repo"]
        serve(mode, model, foreground="--foreground" in a,
              yes="-y" in a or "--yes" in a,
              policy=OVERRIDES.get("when_busy"))

    def do_lm(self, a): self._serve("lm", a)

    def do_vision(self, a): self._serve("vision", a)

    def do_serve(self, a):
        if a and a[0] in ("lm", "vision"):
            self._serve(a[0], a[1:])
            return
        if not a and self.tui:
            mode = choose_mode("which mode?")
            if mode:
                self._serve(mode, [])
            return
        warn("usage: serve lm|vision [model]")

    def do_ls(self, a):
        reg = self.sync()
        print(models_json(reg)) if "--json" in a else print_models(reg)

    def do_models(self, a):
        if not self.tui:
            return self.do_ls(a)
        entry = choose_model(self.sync(), None, "models")
        if not entry:
            return
        repo = entry["repo"]
        acts = [("lm", f"serve in {green('lm')} mode {dim('(text only)')}"),
                *([("vision", (f"serve in {blue('vision')} mode "
                               f"{dim('(multimodal)')}"))]
                  if entry.get("vision") else []),
                ("use lm", "make it the default lm model"),
                *([("use vision", "make it the default vision model")]
                  if entry.get("vision") else []),
                ("chat", "terminal chat with it"),
                ("rm", red("delete it from disk"))]
        got = pick(acts, title=short(repo), render=lambda x: x[1],
                   key=lambda x: x[1])
        if got:
            dispatch(self, f"{got[0]} {shlex.quote(repo)}", interactive=True)

    def do_use(self, a):
        self.sync()
        mode = next((x for x in a if x in ("lm", "vision")), None)
        rest = [x for x in a if x not in ("lm", "vision")]
        if rest:
            entry = resolve(self.reg, rest[0])
            if not entry:
                warn(f"no model matches {rest[0]!r}, run ls to see them")
                return
        elif self.tui:
            if mode is None:
                mode = choose_mode("default for which mode?")
                if not mode:
                    return
            entry = choose_model(self.reg, mode, f"default model for {mode} mode")
            if not entry:
                return
        else:
            warn("usage: use <model> [lm|vision]")
            return
        mode = mode or "lm"
        if mode == "vision" and not entry.get("vision"):
            warn(f"{short(entry['repo'])} has no vision tower")
            return
        self.reg["defaults"][mode] = entry["repo"]
        save_registry(self.reg)
        print(green(f"  default {mode} is now {short(entry['repo'])}"))

    def do_rm(self, a):
        targets = [x for x in a if x != "-y"]
        if not targets:
            if not self.tui:
                warn("usage: rm <model>")
                return
            self.reg = load_registry(refresh=True)
            chosen = choose_model(self.reg, None,
                                  "delete from disk, space marks, enter confirms",
                                  multi=True) or []
            targets = [m["repo"] for m in chosen]
        for t in targets:
            remove(t, yes="-y" in a)
        self.reg = load_registry(refresh=True)

    def do_edit(self, a):
        editor = os.environ.get("VISUAL") or os.environ.get("EDITOR") or "vi"
        BAR.stop()
        subprocess.call([editor, str(REGISTRY)])
        BAR.start()
        self.reg = load_registry(refresh=True)

    def do_config(self, a):
        if not a:
            if not self.tui:
                print_settings()
                return
            rows = list(SETTINGS)
            got = pick(rows, title="settings",
                       render=lambda k: (bold(f"{k:<14}")
                                         + f"{fmt_setting(setting(k)):<18}"
                                         + dim(f"{setting_with_source(k)[1]:<10}  "
                                               f"{SETTINGS[k][2]}")),
                       key=lambda k: k + " " + SETTINGS[k][2])
            if not got:
                return
            try:
                raw = input(f"  {got} = ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if raw:
                set_setting(got, raw)  # the typed line already shows the value
            return
        if a[0] in ("reset", "unset"):
            if len(a) < 2:
                warn("usage: config reset <key>")
                return
            if reset_setting(a[1]):
                print(green(f"  {a[1]} = {fmt_setting(setting(a[1]))} (default)"))
            return
        key = a[0]
        if key not in SETTINGS:
            warn(f"unknown setting {key!r}")
            note(dim("  run config to list them"))
            return
        if len(a) == 1:
            value, src = setting_with_source(key)
            print(f"  {key} = {fmt_setting(value)} ({src})")
            return
        if set_setting(key, a[1]):
            print(green(f"  {key} = {fmt_setting(setting(key))}"))

    def do_browse(self, a):
        opts = parse_browse_args(a)
        if not have_module("huggingface_hub"):
            warn("browsing needs huggingface_hub: pip install huggingface_hub")
            return
        print(dim("  searching..."), end="\r", flush=True)
        try:
            self.hub = hub_list(opts["terms"], opts["kind"], opts["sort"],
                                include_all=opts["include_all"])
        except Exception as e:
            print(" " * 30, end="\r")
            warn(f"hub request failed: {e}")
            return
        print(" " * 30, end="\r")
        what = " ".join(x for x in [opts["kind"] or "",
                                    opts["sort"].replace("Score", ""),
                                    " ".join(opts["terms"])] if x)
        heading = f"mlx models, {what}" + (", all sizes" if opts["include_all"] else "")
        if not self.tui:
            print_hub(self.hub, heading, show_fit=opts["include_all"])
            return
        self.download_from(self.hub, heading)

    def download_from(self, rows: list[dict], title: str):
        chosen = pick(rows, title=title, render=render_hub,
                      key=lambda r: r["repo"], multi=True) or []
        for r in chosen:
            pull(r["repo"])
            r["have"] = is_downloaded(r["repo"])
        self.reg = load_registry(refresh=True)

    def do_get(self, a):
        if not a and self.tui:
            if not self.hub:
                self.do_browse([])
                return
            self.download_from(self.hub, "download, space marks, enter confirms")
            return
        if not a:
            warn("usage: get <n from browse | repo id>")
            return
        yes = "-y" in a
        for tok in [x for x in a if x != "-y"]:
            if tok.isdigit():
                i = int(tok)
                if not 1 <= i <= len(self.hub):
                    warn(f"no row {i} in the last browse list")
                    continue
                pull(self.hub[i - 1]["repo"], yes)
            else:
                pull(tok, yes)
        self.reg = load_registry(refresh=True)

    def do_chat(self, a, replace=True):
        a, target, load = extract_on(a)
        mode = "lm"
        if a and a[0] in ("lm", "vision"):
            mode, a = a[0], a[1:]
        model = a[0] if a else None

        # talk to something already loaded rather than loading a second copy
        if not load and not model and (target or list_servers()):
            st = target_server(target, lambda s: self.choose_server(
                s, "chat with which server?"), "chat")
            if not st:
                return
            chat_server(st)
            return

        if model is None and self.tui:
            self.reg = load_registry(refresh=True)
            entry = choose_model(self.reg, mode, f"chat with, {mode} mode")
            if not entry:
                return
            model = entry["repo"]
        run_chat(mode, model, replace=replace)

    def do_ask(self, a, replace=True):
        a, target, load = extract_on(a)
        if not a and self.tui:
            try:
                a = shlex.split(input("  prompt: "))
            except (EOFError, KeyboardInterrupt):
                print()
                return
        if not a:
            warn("usage: ask <text> [--image PATH] [--on port|model] [--load]")
            return
        imgs = [a[i + 1] for i, t in enumerate(a) if t == "--image" and i + 1 < len(a)]
        skip: set[int] = set()
        for i, t in enumerate(a):
            if t == "--image":
                skip |= {i, i + 1}
        text = " ".join(t for i, t in enumerate(a) if i not in skip)

        if not load and (target or list_servers()):
            st = target_server(target, lambda s: self.choose_server(
                s, "ask which server?"), "ask")
            if st:
                ask_server(st, text, imgs)
            return
        run_ask(text, imgs, replace=replace)


# usage, one line about it, examples
COMMAND_HELP = {
    "lm": ("lm [model] [--port N] [--host ADDR] [--new] [--replace] "
           "[--foreground] [-y]",
           "Serve a model text-only with mlx_lm.",
           ["mlxsh lm", "mlxsh lm qwen3.6", "mlxsh lm 3 --new"]),
    "vision": ("vision [model] [--port N] [--host ADDR] [--new] [--replace] "
               "[--foreground] [-y]",
               "Serve a model with images and video, using mlx_vlm.",
               ["mlxsh vision", "mlxsh vision gemma-4-31b --new"]),
    "serve": ("serve lm|vision [model] [flags]",
              "The same as lm and vision, spelled out.",
              ["mlxsh serve vision gemma-4-31b"]),
    "stop": ("stop [port|model|mode|all]",
             "Stop a server and free its memory. With several running, name "
             "one.",
             ["mlxsh stop", "mlxsh stop 41278", "mlxsh stop gemma",
              "mlxsh stop all"]),
    "status": ("status [--json]",
               "Every server running, with memory, uptime and endpoint.",
               ["mlxsh status", "mlxsh status --json"]),
    "bench": ("bench [port|model] [tokens]",
              "Measure tok/s of a running server, saved to the registry.",
              ["mlxsh bench", "mlxsh bench gemma 256"]),
    "log": ("log [port|model] [lines]",
            "Tail one server's log.",
            ["mlxsh log", "mlxsh log 41278 100"]),
    "ls": ("ls [--json]",
           "Models this machine knows about, with sizes and measured speeds.",
           ["mlxsh ls", "mlxsh ls --json"]),
    "models": ("models",
               "Pick a model, then pick what to do with it. Needs a terminal.",
               ["mlxsh models"]),
    "use": ("use [model] [lm|vision]",
            "Set the default model for a mode.",
            ["mlxsh use qwen3.6 lm", "mlxsh use 2 vision"]),
    "rm": ("rm [model] [-y]",
           "Delete a model from disk. Asks first unless you pass -y.",
           ["mlxsh rm qwen3-0.6b", "mlxsh rm 5 -y"]),
    "browse": ("browse [filters and search terms]",
               "A live list from hugging face. Filters: vision, text, "
               "trending, popular, new, all. An org/ scopes the search.",
               ["mlxsh browse", "mlxsh browse vision new",
                "mlxsh browse qwen3.6"]),
    "get": ("get [row|repo] [-y]",
            "Download a model, by row from the last browse or by repo id.",
            ["mlxsh get 3", "mlxsh get mlx-community/Qwen3.6-27B-4bit -y"]),
    "ask": ("ask <text> [--on port|model] [--image PATH] [--load]",
            "One prompt. Uses a running model, or loads one if none is up.",
            ['mlxsh ask "explain unified memory"',
             'mlxsh ask --on gemma --image shot.png "what is this"']),
    "chat": ("chat [--on port|model] [--load]",
             "A chat loop against a running model.",
             ["mlxsh chat", "mlxsh chat --on qwen3.6"]),
    "config": ("config [key] [value] | config reset <key>",
               "Show settings and where each value comes from, or change one.",
               ["mlxsh config", "mlxsh config port 8080",
                "mlxsh config reset port"]),
    "gateway": ("gateway [stop] [key [--new]] [--port N] [--host ADDR] [--expose]",
                "An authenticated endpoint in front of every loaded model: one\n"
                "URL, a bearer key, routed by model name. It listens on this\n"
                "machine only. Reaching it from anywhere else needs\n"
                "cloudflared and mlxsh tunnel.",
                ["mlxsh gateway", "mlxsh gateway key", "mlxsh gateway stop"]),
    "tunnel": ("tunnel [setup <hostname>] [--quick] [url] [stop]",
               "A public address for the gateway, through cloudflared, which\n"
               "must be installed. setup routes a hostname you own and keeps\n"
               "it; --quick borrows a throwaway one with no account, but it\n"
               "changes on every restart.\n"
               f"  {CLOUDFLARED_DOCS}",
               ["mlxsh tunnel setup llm.example.com", "mlxsh tunnel",
                "mlxsh tunnel --quick", "mlxsh tunnel url",
                "mlxsh tunnel stop"]),
    "setup": ("setup [-y]",
              "Install mlx-lm, mlx-vlm and huggingface_hub where mlxsh runs.",
              ["mlxsh setup"]),
    "doctor": ("doctor", "Versions, paths, memory and endpoint.",
               ["mlxsh doctor"]),
    "edit": ("edit", "Open the registry in $EDITOR.", ["mlxsh edit"]),
    "shell": ("shell", "The command list, then the interactive shell.",
              ["mlxsh shell"]),
}


def command_help(cmd: str):
    usage, what, examples = COMMAND_HELP[cmd]
    print()
    print("  " + bold(usage))
    for line in what.split("\n"):
        print("  " + line)
    if examples:
        print()
        for e in examples:
            print(dim("    " + e))
    print()


ALIASES = {
    "?": "help", "h": "help", "--help": "help", "-h": "help",
    "st": "status", "s": "status", "vl": "vision", "v": "vision",
    "list": "ls", "m": "models", "search": "browse", "find": "browse",
    "hub": "browse", "pull": "get", "download": "get", "add": "get",
    "delete": "rm", "remove": "rm", "default": "use",
    "settings": "config", "cfg": "config", "set": "config",
    "kill": "stop", "quit": "exit", "q": "exit", "bye": "exit",
}
COMMANDS = ["help", "shell", "lm", "vision", "serve", "stop", "status", "bench",
            "log", "ls", "models", "use", "rm", "edit", "config", "browse",
            "get", "chat", "ask", "doctor", "setup", "gateway", "tunnel",
            "exit"]


def dispatch(ctl: Ctl, line: str, interactive: bool) -> bool:
    """Run one command line. Returns False to leave the shell."""
    OVERRIDES.clear()   # flags belong to the command that carried them
    AUTO_PORT.clear()
    FAILED.clear()
    try:
        parts = shlex.split(line)
    except ValueError:
        parts = line.split()
    if not parts:
        return True
    if parts[0] == "mlxsh" and len(parts) > 1:
        parts = parts[1:]   # habit: people type the program name in the shell
    cmd, args = parts[0].lower(), parts[1:]
    cmd = ALIASES.get(cmd, cmd)
    # -h anywhere means "explain yourself", never "do something"
    if any(a in ("-h", "--help") for a in args):
        if cmd in COMMAND_HELP:
            command_help(cmd)
        else:
            print(help_text())
        return True
    if cmd == "exit":
        return False
    if cmd == "shell":
        if interactive:
            note(dim("  already in the shell"))
        else:
            shell()
        return True
    if cmd in ("chat", "ask"):
        getattr(ctl, "do_" + cmd)(args, replace=not interactive)
        return True
    fn = getattr(ctl, "do_" + cmd, None)
    if fn is None:
        warn(f"unknown command {parts[0]!r}")
        near = difflib.get_close_matches(cmd, COMMANDS + list(ALIASES), 1, 0.6)
        note(dim(f"  did you mean {near[0]}?" if near else "  run help"))
        return True
    fn(args)
    return True


# ----------------------------------------------------------------------- shell


class StatusBar:
    """A line pinned to the top of the terminal showing what is loaded.

    The terminal's scroll region is shrunk to everything below row 1, so normal
    output never touches the bar, and the bar is repainted on a timer with the
    cursor saved and restored around it.
    """

    def __init__(self):
        self.thread = None
        self.quit = threading.Event()
        self.paused = False
        self.lock = threading.Lock()
        self.on = False

    def start(self):
        if self.on or not supports_tui() or not setting("status_bar"):
            return
        self.on = True
        rows = shutil.get_terminal_size((100, 30)).lines
        sys.stdout.write("\n")             # do not overwrite the last line
        sys.stdout.write(f"\033[2;{rows}r")  # scroll region below the bar
        sys.stdout.write(f"\033[{rows};1H")
        sys.stdout.flush()
        atexit.register(self.stop)
        try:
            signal.signal(signal.SIGWINCH, self._resized)
            # a killed shell must not leave the terminal with a shrunk region
            for sig in (signal.SIGTERM, signal.SIGHUP):
                signal.signal(sig, self._terminated)
        except (ValueError, OSError):
            pass
        self.paint()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        if not self.on:
            return
        self.on = False
        self.quit.set()
        with self.lock:
            sys.stdout.write("\0337\033[1;1H\033[2K\0338\033[r")
            sys.stdout.flush()

    def _terminated(self, signum, _frame):
        self.stop()
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    def _resized(self, *_):
        if not self.on:
            return
        rows = shutil.get_terminal_size((100, 30)).lines
        with self.lock:
            sys.stdout.write(f"\0337\033[2;{rows}r\0338")
            sys.stdout.flush()
        self.paint()

    def _loop(self):
        while not self.quit.wait(setting("bar_interval")):
            self.paint()

    def text(self) -> str:
        servers = list_servers(scan_port=False)
        if not servers:
            return "mlxsh   nothing loaded"
        stats = proc_stats_many([s["pid"] for s in servers])
        bits = []
        for s in servers:
            rss, up = stats.get(s["pid"], (0, "?"))
            bits.append(f"{s['mode']} {short(s.get('model') or '?')[:24]} "
                        f":{s['port']} {rss / 1e9:.1f}G {up}")
        total = sum(rss for rss, _ in stats.values())
        return ("mlxsh   " + "   ".join(bits)
                + f"   [{len(servers)} loaded, {total / 1e9:.1f}G]")

    def paint(self):
        if not self.on or self.paused:
            return
        cols = shutil.get_terminal_size((100, 30)).columns
        line = self.text()[:cols].ljust(cols)
        with self.lock:
            sys.stdout.write(f"\0337\033[1;1H\033[7m{line}\033[0m\0338")
            sys.stdout.flush()

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False
        self.paint()


BAR = StatusBar()


def setup_readline(ctl: Ctl):
    try:
        import readline
    except ImportError:
        return

    def words():
        return (COMMANDS + list(ALIASES) + list(SETTINGS)
                + ["vision", "text", "trending", "popular", "new", "all"]
                + [short(m["repo"]) for m in ctl.reg["models"]])

    def complete(text, state):
        line = readline.get_line_buffer()
        opts = ([w for w in COMMANDS + list(ALIASES) if w.startswith(text)]
                if line.strip() == text else
                [w for w in words() if w.lower().startswith(text.lower())])
        return opts[state] if state < len(opts) else None

    readline.set_completer(complete)
    readline.set_completer_delims(" \t\n")
    if "libedit" in (readline.__doc__ or ""):
        readline.parse_and_bind("bind ^I rl_complete")
    else:
        readline.parse_and_bind("tab: complete")
    try:
        readline.read_history_file(HISTORY)
    except Exception:
        pass
    readline.set_history_length(500)


def prompt_text() -> str:
    # What is running belongs on the status bar, not in the prompt: with
    # several models loaded there is no single name to show.
    return "mlxsh> "


PALETTE = [
    ("lm", "serve a text-only model, mlx_lm"),
    ("vision", "serve a multimodal model, mlx_vlm"),
    ("stop", "stop a server and free the memory"),
    ("status", "what is running"),
    ("bench", "measure tok/s of a running server"),
    ("models", "pick a model, then pick an action"),
    ("ls", "list models as plain text"),
    ("use", "set the default model for a mode"),
    ("browse", "live model list from hugging face"),
    ("get", "download from the last browse list"),
    ("chat", "terminal chat"),
    ("ask", "single prompt"),
    ("rm", "delete models from disk"),
    ("config", "ports, host, org, memory limits"),
    ("log", "tail the server log"),
    ("doctor", "versions, paths, machine"),
    ("gateway", "an authenticated endpoint for clients off this machine"),
    ("tunnel", "a permanent public address for the gateway"),
    ("edit", "edit the registry"),
    ("help", "full command list"),
    ("exit", "leave the shell"),
]


def shell(banner: bool = True):
    ctl = Ctl(tui=True)
    setup_readline(ctl)
    if banner:
        print(help_text())
        check_deps()
        print(bold("  mlxsh") + dim(f"  {machine_name()}"))
        status(compact=True)
        print()
    BAR.start()
    try:
        shell_loop(ctl)
    finally:
        BAR.stop()


def shell_loop(ctl: Ctl):
    while True:
        BAR.paint()
        try:
            line = input(prompt_text())
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print()
            continue
        if not line.strip() and ctl.tui:
            got = pick(PALETTE, title="commands",
                       render=lambda p: f"{bold(p[0]):<22}{dim(p[1])}",
                       key=lambda p: p[0] + " " + p[1])
            if not got:
                continue
            line = got[0]
        try:
            if not dispatch(ctl, line, interactive=True):
                break
        except KeyboardInterrupt:
            print(dim("\n  cancelled"))
        except Exception as e:
            warn(f"{type(e).__name__}: {e}")
    try:
        import readline

        ensure_home()
        readline.write_history_file(HISTORY)
    except Exception:
        pass


# ------------------------------------------------------------------------ main


def ensure_interpreter():
    """Re-exec under an interpreter that can import mlx_lm, if this one cannot."""
    if have_module("mlx_lm") or os.environ.get("MLXSH_REEXECED"):
        return
    for cand in [os.environ.get("MLXSH_PYTHON"),
                 SELF.parent / ".venv" / "bin" / "python",
                 HOME / ".venv" / "bin" / "python"]:
        if not cand:
            continue
        cand = Path(cand)
        if cand.exists() and str(cand) != sys.executable:
            os.environ["MLXSH_REEXECED"] = "1"
            try:
                os.execv(str(cand), [str(cand), str(SELF), *sys.argv[1:]])
            except OSError:
                del os.environ["MLXSH_REEXECED"]


MIN_ENGINE_PYTHON = (3, 10)


def setup_commands(venv: Path, uv: str | None,
                   into: str | None = None) -> list[list[str]]:
    """How to build, or top up, the environment mlxsh runs models from."""
    pkgs = ["mlx-lm", "mlx-vlm", "huggingface_hub"]
    if into:  # an environment that already works, missing a package
        if uv:
            return [[uv, "pip", "install", "--python", into, *pkgs]]
        return [[into, "-m", "pip", "install", "--upgrade", *pkgs]]
    if uv:  # uv supplies a python of its own, so nothing else is needed
        return [[uv, "venv", "--python", "3.12", str(venv)],
                [uv, "pip", "install", "--python", str(venv / "bin" / "python"),
                 *pkgs]]
    return [[sys.executable, "-m", "venv", str(venv)],
            [str(venv / "bin" / "pip"), "install", "--upgrade", *pkgs]]


def setup(yes: bool = False):
    """Install the MLX packages into the environment mlxsh runs models from."""
    ensure_home()
    venv = HOME / ".venv"
    uv = shutil.which("uv")
    if not uv and sys.version_info < MIN_ENGINE_PYTHON:
        warn(f"python {platform.python_version()} is too old for mlx, "
             f"and uv is not installed")
        print(dim("  install uv first: curl -LsSf https://astral.sh/uv/install.sh | sh"))
        return

    # If this interpreter already runs models, top it up rather than building a
    # second environment beside it: an install missing only mlx-vlm is the
    # common case.
    into = sys.executable if have_module("mlx_lm") else None
    where = Path(sys.executable).parent.parent if into else venv
    if into and not uv:
        into = sys.executable
    print(f"  installing mlx-lm, mlx-vlm and huggingface_hub into {where}")
    print(dim(f"  using {'uv' if uv else sys.executable}, a few hundred MB"))
    if not yes and not confirm("  go ahead?", True):
        return
    had_engines = have_module("mlx_lm")
    for cmd in setup_commands(venv, uv, into):
        if subprocess.run(cmd).returncode != 0:
            warn(f"failed: {' '.join(cmd)}")
            return
    print(green("  done"))
    if not had_engines:
        # this process is still running the interpreter that lacked them
        print(dim("  start mlxsh again to use it"))
    print(dim("  check it with: mlxsh doctor, then: mlxsh browse"))

    if not cloudflared():
        print()
        print(dim("  optional: cloudflared, only needed to reach this machine"))
        print(dim("  from elsewhere, through mlxsh gateway and mlxsh tunnel"))
        brew = shutil.which("brew")
        if brew and not yes and confirm("  install cloudflared now?", False):
            subprocess.run([brew, "install", "cloudflared"])
        else:
            print(dim(f"  {CLOUDFLARED_DOCS}"))


def check_deps():
    # mlxsh itself runs on older pythons, the engines do not, and macOS ships
    # 3.9. Saying so beats a pip error about an unsupported version.
    if sys.version_info < MIN_ENGINE_PYTHON:
        print(yellow(f"  this is python {platform.python_version()}, "
                     f"mlx needs 3.10 or newer"))
        print(dim("  run: mlxsh setup"))
        print(dim("  it needs uv: curl -LsSf https://astral.sh/uv/install.sh | sh"))
        print()
        return
    missing = [p for m, p in (("mlx_lm", "mlx-lm"),
                              ("huggingface_hub", "huggingface_hub"))
               if not have_module(m)]
    if missing:
        print(yellow("  missing: ") + ", ".join(missing))
        print(dim("  run: mlxsh setup"))
        print()


# Help and version answer from whatever interpreter is running, so a broken
# virtualenv cannot stop you reading the help.
NO_REEXEC = {"help", "--help", "-h", "?", "h", "version", "--version", "-V"}


def main(argv: list[str]):
    if argv[:1] == ["--run-gateway"]:   # the child started by the gateway
        run_gateway(argv[1], int(argv[2]))
        return
    if not argv:
        ensure_interpreter()
        print(help_text())
        check_deps()
        if not load_registry()["models"]:
            print(dim("  no models yet. Run: mlxsh browse\n"))
        return
    if argv[0].lower() not in NO_REEXEC:
        ensure_interpreter()
    if argv[0] in ("-V", "--version", "version"):
        print(f"mlxsh {__version__}")
        return

    cmd = ALIASES.get(argv[0].lower(), argv[0].lower())
    line = " ".join(shlex.quote(a) for a in argv)
    if cmd in ("browse", "models", "get") and supports_tui():
        ctl = Ctl(tui=True)
        setup_readline(ctl)
        dispatch(ctl, line, interactive=True)
        BAR.start()
        try:
            shell_loop(ctl)
        finally:
            BAR.stop()
        return
    dispatch(Ctl(tui=False), line, interactive=False)
    if FAILED:
        sys.exit(1)


def cli():
    try:
        main(sys.argv[1:])
    except KeyboardInterrupt:
        print()
    except Exception as e:
        if os.environ.get("MLXSH_DEBUG"):
            raise
        warn(f"{type(e).__name__}: {e}")
        print(dim("  set MLXSH_DEBUG=1 for a traceback, and please report it"))
        sys.exit(1)


if __name__ == "__main__":
    cli()

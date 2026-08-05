"""
Tests for mlxsh. Standard library only, no network, no models.

    python3 -m unittest discover -s tests -v
"""

import contextlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import types
import unittest
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("mlxsh", ROOT / "mlxsh.py")
mlxsh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mlxsh)


@contextlib.contextmanager
def captured():
    """Both streams: output goes to stdout, diagnostics to stderr."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        yield out, err


class TempHome(unittest.TestCase):
    """Point the registry and state at a scratch directory."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._saved = (mlxsh.HOME, mlxsh.REGISTRY, mlxsh.STATE, mlxsh.STATE_DIR,
                       mlxsh.LOG, mlxsh._registry, dict(mlxsh.OVERRIDES),
                       set(mlxsh.AUTO_PORT))
        mlxsh.HOME = Path(self._tmp.name)
        mlxsh.REGISTRY = mlxsh.HOME / "models.json"
        mlxsh.STATE = mlxsh.HOME / "server.json"
        mlxsh.STATE_DIR = mlxsh.HOME / "servers"
        mlxsh.LOG = mlxsh.HOME / "server.log"
        mlxsh._registry = None
        mlxsh.OVERRIDES.clear()
        mlxsh.AUTO_PORT.clear()

    def tearDown(self):
        (mlxsh.HOME, mlxsh.REGISTRY, mlxsh.STATE, mlxsh.STATE_DIR, mlxsh.LOG,
         mlxsh._registry, overrides, auto) = self._saved
        mlxsh.OVERRIDES.clear()
        mlxsh.OVERRIDES.update(overrides)
        mlxsh.AUTO_PORT.clear()
        mlxsh.AUTO_PORT.update(auto)
        for var in list(os.environ):
            if var.startswith("MLXSH_") and var != "MLXSH_REEXECED":
                del os.environ[var]
        self._tmp.cleanup()


class MemoryEstimate(unittest.TestCase):
    """4-bit MLX weights are packed into U32 words, so a parameter count alone
    is off by 8x."""

    def est(self, params):
        info = type("Info", (), {"safetensors": type("St", (), {"parameters": params})})
        return mlxsh.est_memory(info)

    def test_packed_4bit(self):
        self.assertAlmostEqual(
            self.est({"BF16": 1_084_267_648, "U32": 3_357_540_352}) / 1e9,
            15.6, places=1)

    def test_bf16(self):
        self.assertAlmostEqual(self.est({"BF16": 1_000_000_000}) / 1e9, 2.0, places=6)

    def test_missing_metadata(self):
        self.assertEqual(mlxsh.est_memory(type("Info", (), {"safetensors": None})), 0.0)
        self.assertEqual(mlxsh.est_memory(object()), 0.0)

    def test_unknown_dtype_assumed_two_bytes(self):
        self.assertEqual(self.est({"WHAT": 10}), 20)


class VisionSniffing(unittest.TestCase):
    def test_vision_config_key(self):
        self.assertTrue(mlxsh.is_vision_config({"vision_config": {}}))
        self.assertTrue(mlxsh.is_vision_config({"vision_tower": "x"}))
        self.assertTrue(mlxsh.is_vision_config({"image_token_index": 7}))

    def test_architecture_names(self):
        self.assertTrue(mlxsh.is_vision_config(
            {"architectures": ["Gemma4ForConditionalGeneration"]}))
        self.assertTrue(mlxsh.is_vision_config({"architectures": ["Qwen2VLModel"]}))

    def test_text_only(self):
        self.assertFalse(mlxsh.is_vision_config(
            {"model_type": "gpt_oss", "architectures": ["GptOssForCausalLM"]}))
        self.assertFalse(mlxsh.is_vision_config({}))


class MlxCanRun(unittest.TestCase):
    def setUp(self):
        self.original = mlxsh.mlx_families

    def tearDown(self):
        mlxsh.mlx_families = self.original

    def test_quantized_repo_always_runnable(self):
        self.assertTrue(mlxsh.mlx_can_run({"quantization": {"bits": 4}}))
        self.assertTrue(mlxsh.mlx_can_run({"quantization_config": {"bits": 8}}))

    def test_empty_config(self):
        self.assertFalse(mlxsh.mlx_can_run({}))

    def test_falls_back_to_yes_when_mlx_absent(self):
        mlxsh.mlx_families = lambda: frozenset()
        self.assertTrue(mlxsh.mlx_can_run({"model_type": "anything"}))

    def test_unsupported_family_rejected(self):
        mlxsh.mlx_families = lambda: frozenset({"llama", "qwen3"})
        self.assertFalse(mlxsh.mlx_can_run({"model_type": "some_torch_only_arch"}))
        self.assertTrue(mlxsh.mlx_can_run({"model_type": "llama"}))


class DownloadDetection(unittest.TestCase):
    """A cached config.json is not a usable model: we fetch those to sniff a
    repo before downloading it."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["HF_HUB_CACHE"] = self.tmp.name
        self.snap = Path(self.tmp.name) / "models--acme--thing" / "snapshots" / "abc"
        self.snap.mkdir(parents=True)

    def tearDown(self):
        os.environ.pop("HF_HUB_CACHE", None)
        self.tmp.cleanup()

    def test_config_only_is_not_downloaded(self):
        (self.snap / "config.json").write_text("{}")
        self.assertFalse(mlxsh.is_downloaded("acme/thing"))

    def test_weights_present(self):
        (self.snap / "model.safetensors").write_text("x")
        self.assertTrue(mlxsh.is_downloaded("acme/thing"))

    def test_absent_repo(self):
        self.assertFalse(mlxsh.is_downloaded("acme/missing"))

    def test_local_config_read(self):
        (self.snap / "config.json").write_text(json.dumps({"model_type": "gemma4"}))
        self.assertEqual(mlxsh.local_config("acme/thing")["model_type"], "gemma4")


class Resolve(unittest.TestCase):
    reg: dict = {"models": [{"repo": "mlx-community/Qwen3.6-35B-A3B-4bit",
                       "label": "Qwen3.6 35B"},
                      {"repo": "mlx-community/gemma-4-31b-it-4bit",
                       "label": "Gemma 4 31B"},
                      {"repo": "mlx-community/gemma-4-26B-A4B-it-qat-4bit",
                       "label": "Gemma 4 26B"}]}

    def test_by_index(self):
        self.assertEqual(mlxsh.resolve(self.reg, "2")["repo"],
                         "mlx-community/gemma-4-31b-it-4bit")
        self.assertIsNone(mlxsh.resolve(self.reg, "9"))

    def test_exact_repo(self):
        self.assertEqual(
            mlxsh.resolve(self.reg, "mlx-community/gemma-4-31b-it-4bit")["label"],
            "Gemma 4 31B")

    def test_unique_substring(self):
        self.assertEqual(mlxsh.resolve(self.reg, "qwen3.6")["label"], "Qwen3.6 35B")
        self.assertEqual(mlxsh.resolve(self.reg, "31b")["label"], "Gemma 4 31B")

    def test_ambiguous_returns_none(self):
        with captured() as (_out, err):
            self.assertIsNone(mlxsh.resolve(self.reg, "gemma"))
        self.assertIn("matches 2", err.getvalue())

    def test_empty(self):
        self.assertIsNone(mlxsh.resolve(self.reg, ""))


class BrowseArgs(unittest.TestCase):
    def test_defaults_to_trending(self):
        self.assertEqual(mlxsh.parse_browse_args([]),
                         {"terms": [], "kind": None, "include_all": False,
                          "sort": "trendingScore"})

    def test_search_terms_sort_by_downloads(self):
        got = mlxsh.parse_browse_args(["qwen3.6"])
        self.assertEqual(got["sort"], "downloads")
        self.assertEqual(got["terms"], ["qwen3.6"])

    def test_filters(self):
        got = mlxsh.parse_browse_args(["vision", "new", "all", "ocr"])
        self.assertEqual(got["kind"], "vision")
        self.assertEqual(got["sort"], "lastModified")
        self.assertTrue(got["include_all"])
        self.assertEqual(got["terms"], ["ocr"])


class FakeApi:
    def __init__(self, models):
        self.models = models
        self.kwargs = None

    def list_models(self, **kw):
        self.kwargs = kw
        return self.models


def hub_model(repo, params, pipeline="text-generation", downloads=5):
    return type("M", (), {
        "id": repo,
        "safetensors": type("St", (), {"parameters": params}),
        "pipeline_tag": pipeline,
        "tags": [],
        "downloads": downloads,
        "trending_score": 1,
        "last_modified": datetime(2026, 5, 1, tzinfo=timezone.utc),
    })


class HubList(TempHome):
    def run_hub(self, models, **kw):
        api = FakeApi(models)
        module = types.ModuleType("huggingface_hub")
        module.HfApi = lambda: api
        saved = sys.modules.get("huggingface_hub")
        sys.modules["huggingface_hub"] = module
        try:
            return mlxsh.hub_list(**kw), api
        finally:
            if saved is None:
                del sys.modules["huggingface_hub"]
            else:
                sys.modules["huggingface_hub"] = saved

    def test_drops_models_too_big_for_this_machine(self):
        big = int(mlxsh.MEM_TOTAL / 2)  # BF16 is 2 bytes per parameter
        rows, _ = self.run_hub([hub_model("a/small", {"BF16": 1_000_000}),
                                hub_model("a/huge", {"BF16": big})])
        self.assertEqual([r["repo"] for r in rows], ["a/small"])

    def test_include_all_keeps_them(self):
        big = int(mlxsh.MEM_TOTAL / 2)
        rows, _ = self.run_hub([hub_model("a/huge", {"BF16": big})],
                               include_all=True)
        self.assertEqual(len(rows), 1)

    def test_drops_non_llm_pipelines(self):
        rows, _ = self.run_hub([
            hub_model("a/speech", {"F32": 10},
                      pipeline="automatic-speech-recognition"),
            hub_model("a/text", {"F32": 10}),
        ])
        self.assertEqual([r["repo"] for r in rows], ["a/text"])

    def test_marks_vision_and_download_state(self):
        rows, _ = self.run_hub([hub_model("a/vl", {"F32": 10},
                                          pipeline="image-text-to-text")])
        self.assertTrue(rows[0]["vision"])
        self.assertIn("have", rows[0])

    def test_always_filters_to_mlx_and_uses_the_configured_org(self):
        mlxsh.set_setting("org", "someone-else")
        _, api = self.run_hub([hub_model("a/x", {"F32": 10})])
        self.assertEqual(api.kwargs["filter"], "mlx")
        self.assertEqual(api.kwargs["author"], "someone-else")

    def test_org_slash_query_scopes_the_search(self):
        _, api = self.run_hub([hub_model("a/x", {"F32": 10})],
                              terms=["lmstudio-community/gemma"])
        self.assertEqual(api.kwargs["author"], "lmstudio-community")
        self.assertEqual(api.kwargs["search"], "gemma")


class Settings(TempHome):
    def test_default_when_nothing_set(self):
        self.assertEqual(mlxsh.setting_with_source("port"), (41277, "default"))

    def test_registry_beats_default(self):
        mlxsh.set_setting("port", "8080")
        self.assertEqual(mlxsh.setting_with_source("port"), (8080, "registry"))

    def test_env_beats_registry(self):
        mlxsh.set_setting("port", "8080")
        os.environ["MLXSH_PORT"] = "9090"
        self.assertEqual(mlxsh.setting_with_source("port"), (9090, "env"))

    def test_flag_beats_env(self):
        os.environ["MLXSH_PORT"] = "9090"
        mlxsh.split_flags(["--port", "7070"])
        self.assertEqual(mlxsh.setting_with_source("port"), (7070, "flag"))

    def test_reset_returns_to_default(self):
        mlxsh.set_setting("port", "8080")
        mlxsh.reset_setting("port")
        self.assertEqual(mlxsh.setting_with_source("port"), (41277, "default"))

    def test_rejects_bad_values(self):
        with captured():
            self.assertFalse(mlxsh.set_setting("port", "not-a-number"))
            self.assertFalse(mlxsh.set_setting("port", "70000"))
            self.assertFalse(mlxsh.set_setting("mem_comfy", "2"))
            self.assertFalse(mlxsh.set_setting("nonsense", "1"))
        self.assertEqual(mlxsh.setting("port"), 41277)

    def test_flags_are_removed_from_the_argument_list(self):
        rest = mlxsh.split_flags(["qwen3.6", "--port", "8080", "--foreground"])
        self.assertEqual(rest, ["qwen3.6", "--foreground"])
        self.assertEqual(mlxsh.setting("port"), 8080)

    def test_equals_form(self):
        self.assertEqual(mlxsh.split_flags(["--host=0.0.0.0"]), [])
        self.assertEqual(mlxsh.setting("host"), "0.0.0.0")

    def test_memory_thresholds_follow_the_setting(self):
        mlxsh.set_setting("mem_tight", "0.5")
        self.assertAlmostEqual(mlxsh.mem_tight(), mlxsh.MEM_TOTAL * 0.5)

    def test_every_setting_has_an_env_var_and_a_default(self):
        for key, (env_var, _kind, blurb) in mlxsh.SETTINGS.items():
            self.assertTrue(env_var.startswith("MLXSH_"), key)
            self.assertIn(key, mlxsh.DEFAULTS, key)
            self.assertTrue(blurb, key)


class Registry(TempHome):
    def test_seeds_then_round_trips(self):
        reg = mlxsh.load_registry()
        self.assertEqual(reg["models"], [])
        self.assertTrue(mlxsh.REGISTRY.exists())
        reg["models"].append({"repo": "a/b", "label": "b", "vision": False})
        mlxsh.save_registry(reg)
        self.assertEqual(mlxsh.load_registry(refresh=True)["models"][0]["repo"],
                         "a/b")

    def test_old_file_gains_new_keys(self):
        mlxsh.REGISTRY.parent.mkdir(parents=True, exist_ok=True)
        mlxsh.REGISTRY.write_text(json.dumps({"models": [], "port": 1234}))
        reg = mlxsh.load_registry(refresh=True)
        for key in ("version", "host", "org", "chat_args", "browse_limit"):
            self.assertIn(key, reg)
        self.assertEqual(reg["port"], 1234)


class ServerIdentity(TempHome):
    """Pids get reused, and other programs use ports. Killing whatever holds
    either one is not acceptable."""

    def setUp(self):
        super().setUp()
        self._pid_alive, self._pid_command = mlxsh.pid_alive, mlxsh.pid_command

    def tearDown(self):
        mlxsh.pid_alive, mlxsh.pid_command = self._pid_alive, self._pid_command
        super().tearDown()

    def test_our_server_recognised(self):
        mlxsh.pid_alive = lambda pid: True
        mlxsh.pid_command = lambda pid: (
            "/usr/bin/python -m mlx_lm server --model mlx-community/x --port 41277")
        self.assertTrue(mlxsh.is_our_server(999))

    def test_vision_server_recognised(self):
        mlxsh.pid_alive = lambda pid: True
        mlxsh.pid_command = lambda pid: "python -m mlx_vlm.server --model a/b"
        self.assertTrue(mlxsh.is_our_server(999))

    def test_unrelated_process_rejected(self):
        mlxsh.pid_alive = lambda pid: True
        mlxsh.pid_command = lambda pid: "/Applications/Some.app/Contents/MacOS/Some"
        self.assertFalse(mlxsh.is_our_server(999))

    def test_dead_pid_rejected(self):
        mlxsh.pid_alive = lambda pid: False
        self.assertFalse(mlxsh.is_our_server(999))

    def write_state(self):
        mlxsh.write_state({"pid": 999, "mode": "lm", "model": "a/b",
                           "port": 41277})

    def test_state_ignored_when_the_pid_is_not_ours(self):
        self.write_state()
        mlxsh.pid_alive = lambda pid: True
        mlxsh.pid_command = lambda pid: "vim notes.txt"
        self.assertIsNone(mlxsh.read_state())

    def test_state_kept_when_the_pid_is_ours(self):
        self.write_state()
        mlxsh.pid_alive = lambda pid: True
        mlxsh.pid_command = lambda pid: "python -m mlx_lm server --model a/b"
        self.assertEqual(mlxsh.read_state()["model"], "a/b")

    def test_stop_refuses_to_kill_a_stranger_on_the_port(self):
        killed = []
        saved_owner, saved_kill = mlxsh.port_owner, os.kill
        mlxsh.port_owner = lambda port: 4242
        mlxsh.pid_alive = lambda pid: True
        mlxsh.pid_command = lambda pid: "python -m http.server 41277"
        os.kill = lambda pid, sig: killed.append((pid, sig))
        try:
            with captured() as (_out, err):
                self.assertFalse(mlxsh.stop_server())
        finally:
            mlxsh.port_owner, os.kill = saved_owner, saved_kill
        self.assertEqual(killed, [])
        self.assertIn("not an mlx server", err.getvalue())

    def test_adopts_a_server_started_outside_the_shell(self):
        saved_owner = mlxsh.port_owner
        mlxsh.port_owner = lambda port: 4242
        mlxsh.pid_alive = lambda pid: True
        mlxsh.pid_command = lambda pid: (
            "python -m mlx_lm server --model mlx-community/x --port 41277")
        try:
            found = mlxsh.find_server(41277)
        finally:
            mlxsh.port_owner = saved_owner
        self.assertEqual(found["model"], "mlx-community/x")
        self.assertTrue(found["adopted"])


class MultipleServers(TempHome):
    """One model per port. Serving on another port leaves the first alone."""

    def setUp(self):
        super().setUp()
        self._saved_fns = (mlxsh.pid_alive, mlxsh.pid_command, mlxsh.port_owner,
                           mlxsh.proc_stats, mlxsh.port_open)
        mlxsh.pid_alive = lambda pid: True
        mlxsh.pid_command = lambda pid: "python -m mlx_lm server --model a/b"
        mlxsh.port_owner = lambda port: None
        mlxsh.proc_stats = lambda pid: (1_000_000_000, "00:10")
        mlxsh.port_open = lambda port, host=None: False

    def tearDown(self):
        (mlxsh.pid_alive, mlxsh.pid_command, mlxsh.port_owner,
         mlxsh.proc_stats, mlxsh.port_open) = self._saved_fns
        super().tearDown()

    def add(self, port, model, mode="lm", pid=None):
        mlxsh.write_state({"pid": pid or 1000 + port, "mode": mode,
                           "model": model, "port": port, "host": "127.0.0.1",
                           "engine": "mlx_lm"})

    def test_each_port_has_its_own_state_file(self):
        self.add(41277, "a/one")
        self.add(41300, "a/two")
        self.assertEqual([s["port"] for s in mlxsh.list_servers()],
                         [41277, 41300])
        self.assertEqual(mlxsh.read_state(41300)["model"], "a/two")

    def test_dead_servers_are_forgotten(self):
        self.add(41277, "a/one")
        self.add(41300, "a/two")
        mlxsh.pid_alive = lambda pid: pid != 41300 + 1000
        self.assertEqual([s["port"] for s in mlxsh.list_servers()], [41277])
        self.assertFalse(mlxsh.state_path(41300).exists())

    def test_legacy_single_state_file_is_migrated(self):
        mlxsh.HOME.mkdir(parents=True, exist_ok=True)
        mlxsh.STATE.write_text(json.dumps({"pid": 42, "mode": "vision",
                                           "model": "a/old", "port": 41277}))
        servers = mlxsh.list_servers()
        self.assertEqual([s["model"] for s in servers], ["a/old"])
        self.assertFalse(mlxsh.STATE.exists())
        self.assertTrue(mlxsh.state_path(41277).exists())

    def test_match_by_port_model_mode_and_all(self):
        self.add(41277, "mlx-community/gemma-4-31b")
        self.add(41300, "mlx-community/qwen3.6", mode="vision")
        self.assertEqual(len(mlxsh.match_servers("all")), 2)
        self.assertEqual(mlxsh.match_servers("41300")[0]["model"],
                         "mlx-community/qwen3.6")
        self.assertEqual(mlxsh.match_servers("gemma")[0]["port"], 41277)
        self.assertEqual(mlxsh.match_servers("vision")[0]["port"], 41300)
        self.assertEqual(mlxsh.match_servers("nothing"), [])

    def test_free_port_skips_used_ones(self):
        self.add(41277, "a/one")
        self.add(41278, "a/two")
        self.assertEqual(mlxsh.free_port(41277), 41279)

    def test_stop_without_a_target_needs_one_when_several_run(self):
        self.add(41277, "a/one")
        self.add(41300, "a/two")
        with captured() as (_out, err):
            self.assertFalse(mlxsh.stop_server())
        self.assertIn("name one", err.getvalue())

    def test_stop_uses_the_chooser_when_several_run(self):
        self.add(41277, "a/one")
        self.add(41300, "a/two")
        killed = []
        saved = mlxsh.stop_one
        mlxsh.stop_one = lambda st: killed.append(st["port"]) or 0
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                mlxsh.stop_server(choose=lambda servers: servers[1])
        finally:
            mlxsh.stop_one = saved
        self.assertEqual(killed, [41300])

    def test_stop_all(self):
        self.add(41277, "a/one")
        self.add(41300, "a/two")
        killed = []
        saved = mlxsh.stop_one
        mlxsh.stop_one = lambda st: killed.append(st["port"]) or 0
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                mlxsh.stop_server("all")
        finally:
            mlxsh.stop_one = saved
        self.assertEqual(killed, [41277, 41300])

    def test_memory_check_passes_when_there_is_room(self):
        self.add(41277, "a/one")
        saved = mlxsh.cache_sizes
        mlxsh.cache_sizes = lambda: {"a/two": 1_000_000_000}
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertTrue(mlxsh.memory_check("a/two", 41300))
        finally:
            mlxsh.cache_sizes = saved

    def test_memory_check_asks_when_it_would_not_fit(self):
        self.add(41277, "a/one")
        saved_cache, saved_confirm = mlxsh.cache_sizes, mlxsh.confirm
        mlxsh.cache_sizes = lambda: {"a/two": mlxsh.MEM_TOTAL}
        asked = []
        mlxsh.confirm = lambda prompt, default=False: asked.append(prompt) or False
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertFalse(mlxsh.memory_check("a/two", 41300))
                self.assertTrue(mlxsh.memory_check("a/two", 41300, yes=True))
        finally:
            mlxsh.cache_sizes, mlxsh.confirm = saved_cache, saved_confirm
        self.assertEqual(len(asked), 1)

    def test_free_port_when_nothing_runs(self):
        self.assertEqual(mlxsh.choose_port("a/one"), 41277)

    def test_busy_port_defaults_to_a_new_one(self):
        self.add(41277, "a/one")
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(mlxsh.choose_port("a/two"), 41278)
        self.assertIn("starting on :41278", out.getvalue())

    def test_replace_policy_reuses_the_port(self):
        self.add(41277, "a/one")
        self.assertEqual(mlxsh.choose_port("a/two", policy="replace"), 41277)

    def test_replace_can_be_configured(self):
        self.add(41277, "a/one")
        mlxsh.set_setting("when_busy", "replace")
        self.assertEqual(mlxsh.choose_port("a/two"), 41277)

    def test_ask_falls_back_to_new_without_a_terminal(self):
        self.add(41277, "a/one")
        mlxsh.set_setting("when_busy", "ask")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(mlxsh.choose_port("a/two"), 41278)

    def test_when_busy_only_accepts_known_values(self):
        with captured():
            self.assertFalse(mlxsh.set_setting("when_busy", "explode"))
        self.assertEqual(mlxsh.setting("when_busy"), "new")

    def test_an_explicit_port_wins_even_if_busy(self):
        self.add(41277, "a/one")
        mlxsh.split_flags(["--port", "41277"])
        self.assertEqual(mlxsh.choose_port("a/two"), 41277)

    def test_same_model_in_another_mode_reuses_its_port(self):
        self.add(41300, "a/one", mode="lm")
        self.assertEqual(mlxsh.choose_port("a/one"), 41300)

    def test_new_flag_asks_for_the_next_port(self):
        self.add(41277, "a/one")
        self.assertEqual(mlxsh.split_flags(["qwen", "--new"]), ["qwen"])
        self.assertEqual(mlxsh.setting("port"), 41277)  # the setting is untouched
        self.assertEqual(mlxsh.choose_port("a/two"), 41278)

    def test_port_auto_is_accepted(self):
        self.add(41277, "a/one")
        mlxsh.split_flags(["--port", "auto"])
        self.assertEqual(mlxsh.choose_port("a/two"), 41278)


class Targeting(TempHome):
    """ask, chat, bench and log all have to work out which model answers."""

    def setUp(self):
        super().setUp()
        self._fns = (mlxsh.pid_alive, mlxsh.pid_command, mlxsh.port_owner)
        mlxsh.pid_alive = lambda pid: True
        mlxsh.pid_command = lambda pid: "python -m mlx_lm server --model a/b"
        mlxsh.port_owner = lambda port: None

    def tearDown(self):
        (mlxsh.pid_alive, mlxsh.pid_command, mlxsh.port_owner) = self._fns
        super().tearDown()

    def add(self, port, model, mode="lm"):
        mlxsh.write_state({"pid": 1000 + port, "mode": mode, "model": model,
                           "port": port, "host": "127.0.0.1"})

    def test_extract_on_and_load(self):
        rest, target, load = mlxsh.extract_on(
            ["say", "hi", "--on", "41300", "--load"])
        self.assertEqual(rest, ["say", "hi"])
        self.assertEqual(target, "41300")
        self.assertTrue(load)

    def test_extract_on_equals_form(self):
        rest, target, load = mlxsh.extract_on(["--on=gemma", "hello"])
        self.assertEqual((rest, target, load), (["hello"], "gemma", False))

    def test_no_flags(self):
        self.assertEqual(mlxsh.extract_on(["plain", "words"]),
                         (["plain", "words"], None, False))

    def test_single_server_needs_no_target(self):
        self.add(41277, "a/only")
        self.assertEqual(mlxsh.target_server()["model"], "a/only")

    def test_target_by_port_and_by_name(self):
        self.add(41277, "org/gemma")
        self.add(41278, "org/qwen")
        self.assertEqual(mlxsh.target_server("41278")["model"], "org/qwen")
        self.assertEqual(mlxsh.target_server("gemma")["port"], 41277)

    def test_several_servers_without_a_target_explain_themselves(self):
        self.add(41277, "org/gemma")
        self.add(41278, "org/qwen")
        with captured() as (_out, err):
            self.assertIsNone(mlxsh.target_server(verb="ask"))
        self.assertIn("ask --on", err.getvalue())

    def test_unknown_target(self):
        self.add(41277, "org/gemma")
        with captured() as (_out, err):
            self.assertIsNone(mlxsh.target_server("nope"))
        self.assertIn("no running server matches", err.getvalue())

    def test_chooser_used_when_several_run(self):
        self.add(41277, "org/gemma")
        self.add(41278, "org/qwen")
        self.assertEqual(
            mlxsh.target_server(choose=lambda servers: servers[1])["port"], 41278)

    def test_log_path_per_port(self):
        self.assertEqual(mlxsh.log_path(41300).name, "41300.log")
        self.assertEqual(mlxsh.log_path(), mlxsh.LOG)

    def test_max_tokens_comes_from_the_setting(self):
        self.assertEqual(mlxsh.arg_value(["--max-tokens", "512"],
                                         "--max-tokens", 2048), 512)
        self.assertEqual(mlxsh.arg_value([], "--max-tokens", 2048), 2048)
        self.assertEqual(mlxsh.arg_value(["--max-tokens"], "--max-tokens", 7), 7)


class ApiStreaming(TempHome):
    """ask and chat read server-sent events rather than loading a model."""

    def serve_sse(self, chunks):
        import http.server
        import threading

        body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.write(body.encode())

            def log_message(self, *_):
                pass

        httpd = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=httpd.handle_request, daemon=True).start()
        return httpd.server_address[1], httpd

    def test_deltas_are_joined(self):
        chunks = [{"choices": [{"delta": {"content": "Hel"}}]},
                  {"choices": [{"delta": {"content": "lo"}}]},
                  {"choices": [{"delta": {}}]}]
        port, httpd = self.serve_sse(chunks)
        try:
            with contextlib.redirect_stdout(io.StringIO()) as out:
                got = mlxsh.api_stream({"port": port, "host": "127.0.0.1",
                                        "model": "a/b"},
                                       [{"role": "user", "content": "hi"}])
        finally:
            httpd.server_close()
        self.assertEqual(got, "Hello")
        self.assertIn("Hello", out.getvalue())

    def test_unreachable_server_reports_the_failure(self):
        with captured() as (_out, err):
            got = mlxsh.api_stream({"port": 9, "host": "127.0.0.1", "model": "a/b"},
                                   [{"role": "user", "content": "hi"}])
        self.assertIsNone(got)
        self.assertIn("request failed", err.getvalue())

    def test_images_need_a_vision_server(self):
        with captured() as (_out, err):
            self.assertFalse(mlxsh.ask_server({"port": 1, "mode": "lm",
                                               "model": "a/b"}, "hi", ["x.png"]))
        self.assertIn("images need a vision one", err.getvalue())

    def test_image_payload_is_a_data_url(self):
        png = Path(self._tmp.name) / "x.png"
        png.write_bytes(b"\x89PNG\r\n")
        payload = mlxsh.image_payload(str(png))
        self.assertEqual(payload["type"], "image_url")
        self.assertTrue(payload["image_url"]["url"].startswith("data:image/png;base64,"))


class StatusBarText(TempHome):
    """The bar replaces the model name that used to sit in the prompt."""

    def setUp(self):
        super().setUp()
        self._fns = (mlxsh.pid_alive, mlxsh.pid_command, mlxsh.proc_stats_many)
        mlxsh.pid_alive = lambda pid: True
        mlxsh.pid_command = lambda pid: "python -m mlx_lm server --model a/b"
        mlxsh.proc_stats_many = lambda pids: dict.fromkeys(
            pids, (2_000_000_000, "01:00"))

    def tearDown(self):
        (mlxsh.pid_alive, mlxsh.pid_command, mlxsh.proc_stats_many) = self._fns
        super().tearDown()

    def test_prompt_stays_plain(self):
        mlxsh.write_state({"pid": 1, "mode": "lm", "model": "a/b", "port": 41277})
        self.assertEqual(mlxsh.prompt_text(), "mlxsh> ")

    def test_nothing_loaded(self):
        self.assertIn("nothing loaded", mlxsh.StatusBar().text())

    def test_lists_every_server(self):
        mlxsh.write_state({"pid": 1, "mode": "lm", "model": "org/one",
                           "port": 41277})
        mlxsh.write_state({"pid": 2, "mode": "vision", "model": "org/two",
                           "port": 41278})
        text = mlxsh.StatusBar().text()
        self.assertIn("lm one :41277", text)
        self.assertIn("vision two :41278", text)
        self.assertIn("2 loaded", text)
        self.assertIn("4.0G", text)  # both, added up


class BooleanSetting(TempHome):
    def test_accepts_the_usual_spellings(self):
        for on in ("on", "true", "yes", "1", True):
            self.assertTrue(mlxsh.boolean(on))
        for off in ("off", "false", "no", "0", False):
            self.assertFalse(mlxsh.boolean(off))

    def test_rejects_nonsense(self):
        with self.assertRaises(ValueError):
            mlxsh.boolean("maybe")

    def test_config_can_turn_the_bar_off(self):
        self.assertTrue(mlxsh.setting("status_bar"))
        self.assertTrue(mlxsh.set_setting("status_bar", "off"))
        self.assertFalse(mlxsh.setting("status_bar"))
        self.assertEqual(mlxsh.setting_with_source("status_bar")[1], "registry")

    def test_env_can_turn_it_off(self):
        os.environ["MLXSH_STATUS_BAR"] = "off"
        self.assertFalse(mlxsh.setting("status_bar"))


class ProcStats(unittest.TestCase):
    def test_reads_several_pids_at_once(self):
        stats = mlxsh.proc_stats_many([os.getpid()])
        self.assertIn(os.getpid(), stats)
        rss, up = stats[os.getpid()]
        self.assertGreater(rss, 0)
        self.assertTrue(up)

    def test_no_pids(self):
        self.assertEqual(mlxsh.proc_stats_many([]), {})

    def test_single_pid_helper_agrees(self):
        rss, _ = mlxsh.proc_stats(os.getpid())
        self.assertGreater(rss, 0)


class FlagsAreScopedToOneCommand(TempHome):
    """A --port on one command must not follow you around the shell."""

    def test_dispatch_clears_run_flags(self):
        ctl = mlxsh.Ctl(tui=False)
        mlxsh.split_flags(["--port", "9999"])
        mlxsh.AUTO_PORT.add(True)
        with contextlib.redirect_stdout(io.StringIO()):
            mlxsh.dispatch(ctl, "status", interactive=True)
        self.assertEqual(mlxsh.OVERRIDES, {})
        self.assertEqual(mlxsh.AUTO_PORT, set())
        self.assertEqual(mlxsh.setting_with_source("port"), (41277, "default"))


class ExitCodes(TempHome):
    """Scripts and agents need a failed command to be a failed command."""

    def test_warning_marks_the_run_as_failed(self):
        mlxsh.FAILED.clear()
        with captured():
            mlxsh.warn("something went wrong")
        self.assertTrue(mlxsh.FAILED)

    def test_dispatch_starts_each_command_clean(self):
        ctl = mlxsh.Ctl(tui=False)
        mlxsh.FAILED.append("stale")
        with captured():
            mlxsh.dispatch(ctl, "status", interactive=True)
        self.assertEqual(mlxsh.FAILED, [])

    def test_an_unknown_command_fails(self):
        ctl = mlxsh.Ctl(tui=False)
        with contextlib.redirect_stdout(io.StringIO()):
            mlxsh.dispatch(ctl, "nonsense", interactive=False)
        self.assertTrue(mlxsh.FAILED)


class Bootstrap(TempHome):
    """setup builds the environment that mlxsh re-execs into."""

    def test_uv_path_supplies_its_own_python(self):
        cmds = mlxsh.setup_commands(Path("/tmp/x/.venv"), "/opt/bin/uv")
        self.assertEqual(cmds[0][:3], ["/opt/bin/uv", "venv", "--python"])
        for pkg in ("mlx-lm", "mlx-vlm", "huggingface_hub"):
            self.assertIn(pkg, cmds[1])

    def test_without_uv_it_uses_this_interpreter(self):
        cmds = mlxsh.setup_commands(Path("/tmp/x/.venv"), None)
        self.assertEqual(cmds[0][:3], [sys.executable, "-m", "venv"])
        self.assertTrue(cmds[1][0].endswith("/.venv/bin/pip"))

    def test_topping_up_an_environment_that_already_works(self):
        cmds = mlxsh.setup_commands(Path("/tmp/x/.venv"), "/opt/bin/uv",
                                    into="/opt/tools/mlxsh/bin/python")
        self.assertEqual(len(cmds), 1)  # no venv is created
        self.assertEqual(cmds[0][:4],
                         ["/opt/bin/uv", "pip", "install", "--python"])
        self.assertIn("mlx-vlm", cmds[0])

    def test_topping_up_without_uv(self):
        cmds = mlxsh.setup_commands(Path("/tmp/x/.venv"), None,
                                    into="/opt/env/bin/python")
        self.assertEqual(cmds[0][:3], ["/opt/env/bin/python", "-m", "pip"])

    def test_the_venv_is_where_the_re_exec_looks(self):
        venv = mlxsh.HOME / ".venv"
        cmds = mlxsh.setup_commands(venv, None)
        self.assertIn(str(venv), cmds[0])

    def test_too_old_without_uv_is_refused(self):
        saved = mlxsh.shutil.which
        mlxsh.shutil.which = lambda name: None
        version = sys.version_info
        try:
            mlxsh.sys.version_info = (3, 9, 6)
            with captured() as (_out, err):
                mlxsh.setup(yes=True)
        finally:
            mlxsh.sys.version_info = version
            mlxsh.shutil.which = saved
        self.assertIn("too old", err.getvalue())
        self.assertFalse((mlxsh.HOME / ".venv").exists())


class CacheView(TempHome):
    """A server should advertise its own model, not the whole cache."""

    def setUp(self):
        super().setUp()
        self.cache = tempfile.TemporaryDirectory()
        os.environ["HF_HUB_CACHE"] = self.cache.name
        self.repo_dir = Path(self.cache.name) / "models--acme--thing"
        (self.repo_dir / "snapshots" / "abc").mkdir(parents=True)

    def tearDown(self):
        os.environ.pop("HF_HUB_CACHE", None)
        self.cache.cleanup()
        super().tearDown()

    def test_view_holds_only_that_model(self):
        view = mlxsh.cache_view(41277, "acme/thing")
        self.assertIsNotNone(view)
        entries = list(view.iterdir())
        self.assertEqual([e.name for e in entries], ["models--acme--thing"])
        self.assertTrue(entries[0].is_symlink())
        self.assertEqual(entries[0].resolve(), self.repo_dir.resolve())

    def test_one_view_per_port(self):
        a = mlxsh.cache_view(41277, "acme/thing")
        b = mlxsh.cache_view(41278, "acme/thing")
        self.assertNotEqual(a, b)
        self.assertTrue(a.exists() and b.exists())

    def test_rebuilding_a_view_does_not_pile_up(self):
        mlxsh.cache_view(41277, "acme/thing")
        view = mlxsh.cache_view(41277, "acme/thing")
        self.assertEqual(len(list(view.iterdir())), 1)

    def test_missing_repo_gets_no_view(self):
        self.assertIsNone(mlxsh.cache_view(41277, "acme/absent"))

    def test_dropping_a_view(self):
        view = mlxsh.cache_view(41277, "acme/thing")
        mlxsh.drop_cache_view(41277)
        self.assertFalse(view.exists())
        mlxsh.drop_cache_view(41277)  # twice is fine

    def test_the_setting_is_on_by_default(self):
        self.assertTrue(mlxsh.setting("pin_model"))
        mlxsh.set_setting("pin_model", "off")
        self.assertFalse(mlxsh.setting("pin_model"))


class ModelAlias(TempHome):
    """default_model is an mlx_lm feature; mlx_vlm answers 400 to it."""

    def setUp(self):
        super().setUp()
        self._fns = (mlxsh.pid_alive, mlxsh.pid_command, mlxsh.proc_stats_many,
                     mlxsh.port_owner)
        mlxsh.pid_alive = lambda pid: True
        mlxsh.pid_command = lambda pid: "python -m mlx_lm server --model a/b"
        mlxsh.proc_stats_many = lambda pids: dict.fromkeys(pids, (1, "00:01"))
        mlxsh.port_owner = lambda port: None

    def tearDown(self):
        (mlxsh.pid_alive, mlxsh.pid_command, mlxsh.proc_stats_many,
         mlxsh.port_owner) = self._fns
        super().tearDown()

    def test_offered_for_lm_servers(self):
        mlxsh.write_state({"pid": 1, "mode": "lm", "model": "a/b",
                           "port": 41277, "engine": "mlx_lm"})
        self.assertEqual(json.loads(mlxsh.servers_json())[0]["alias"],
                         "default_model")

    def test_not_offered_for_vision_servers(self):
        mlxsh.write_state({"pid": 1, "mode": "vision", "model": "a/b",
                           "port": 41277, "engine": "mlx_vlm"})
        self.assertIsNone(json.loads(mlxsh.servers_json())[0]["alias"])


class GatewayKey(TempHome):
    def test_created_once_and_reused(self):
        first = mlxsh.api_key()
        self.assertTrue(first.startswith("mlxsh-"))
        self.assertEqual(mlxsh.api_key(), first)
        self.assertEqual(oct(mlxsh.key_path().stat().st_mode)[-3:], "600")

    def test_rotating_replaces_it(self):
        first = mlxsh.api_key()
        self.assertNotEqual(mlxsh.api_key(new=True), first)

    def test_environment_wins(self):
        mlxsh.api_key()
        os.environ["MLXSH_API_KEY"] = "mlxsh-from-the-environment"
        self.assertEqual(mlxsh.api_key(), "mlxsh-from-the-environment")

    def test_header_matching(self):
        key = mlxsh.api_key()
        self.assertTrue(mlxsh.key_matches(f"Bearer {key}"))
        self.assertTrue(mlxsh.key_matches(f"bearer {key}"))
        for wrong in ("", "Bearer ", "Bearer wrong", key, f"Basic {key}",
                      f"Bearer {key}x"):
            self.assertFalse(mlxsh.key_matches(wrong), wrong)


class GatewayRouting(TempHome):
    """One endpoint in front of several ports, chosen by model name."""

    def setUp(self):
        super().setUp()
        self._fns = (mlxsh.pid_alive, mlxsh.pid_command, mlxsh.port_owner)
        mlxsh.pid_alive = lambda pid: True
        mlxsh.pid_command = lambda pid: "python -m mlx_lm server --model a/b"
        mlxsh.port_owner = lambda port: None
        for port, model, mode in ((41277, "org/gemma-4-26B", "lm"),
                                  (41278, "org/qwen3.6", "vision")):
            mlxsh.write_state({"pid": 1000 + port, "mode": mode, "model": model,
                               "port": port, "host": "127.0.0.1",
                               "engine": "mlx_lm", "started": 1})

    def tearDown(self):
        mlxsh.pid_alive, mlxsh.pid_command, mlxsh.port_owner = self._fns
        super().tearDown()

    def target(self, model):
        body = json.dumps({"model": model, "messages": []}).encode()
        server, rewritten = mlxsh.gateway_target(body)
        return server, json.loads(rewritten)

    def test_exact_repo_id(self):
        server, body = self.target("org/qwen3.6")
        self.assertEqual(server["port"], 41278)
        self.assertEqual(body["model"], "org/qwen3.6")

    def test_substring(self):
        server, _ = self.target("gemma")
        self.assertEqual(server["port"], 41277)

    def test_mode(self):
        server, _ = self.target("vision")
        self.assertEqual(server["port"], 41278)

    def test_alias_and_empty_take_the_first(self):
        for name in ("default_model", ""):
            server, _ = self.target(name)
            self.assertEqual(server["port"], 41277, name)

    def test_the_upstream_always_gets_its_own_repo_id(self):
        _, body = self.target("vision")
        self.assertEqual(body["model"], "org/qwen3.6")

    def test_unknown_model_has_no_target(self):
        server, _ = self.target("something/else")
        self.assertIsNone(server)

    def test_nothing_loaded(self):
        for port in (41277, 41278):
            mlxsh.state_path(port).unlink()
        server, _ = mlxsh.gateway_target(b'{"model": "anything"}')
        self.assertIsNone(server)

    def test_malformed_body_with_one_server(self):
        mlxsh.state_path(41278).unlink()
        server, _ = mlxsh.gateway_target(b"not json at all")
        self.assertEqual(server["port"], 41277)

    def test_models_listing_covers_every_loaded_model(self):
        listing = mlxsh.gateway_models()
        self.assertEqual([m["id"] for m in listing["data"]],
                         ["org/gemma-4-26B", "org/qwen3.6"])

    def test_refuses_a_public_bind_without_expose(self):
        with captured() as (_out, err):
            mlxsh.start_gateway(host="0.0.0.0")
        self.assertIn("would put the endpoint on your network", err.getvalue())
        self.assertIsNone(mlxsh.gateway_state())


class TunnelCommands(TempHome):
    """Setting up a named tunnel is a sequence of cloudflared calls."""

    def test_full_sequence_when_nothing_exists(self):
        steps = mlxsh.tunnel_commands("mlxsh", "llm.example.com", 41377,
                                      logged_in=False, exists=False)
        self.assertEqual([s[1:3] for s in steps],
                         [["tunnel", "login"], ["tunnel", "create"],
                          ["tunnel", "route"]])

    def test_login_skipped_when_already_logged_in(self):
        steps = mlxsh.tunnel_commands("mlxsh", "llm.example.com", 41377,
                                      logged_in=True, exists=False)
        self.assertNotIn("login", [s[2] for s in steps])

    def test_create_skipped_when_the_tunnel_exists(self):
        steps = mlxsh.tunnel_commands("mlxsh", "llm.example.com", 41377,
                                      logged_in=True, exists=True)
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0][1:4], ["tunnel", "route", "dns"])
        self.assertIn("llm.example.com", steps[0])

    def test_run_points_at_the_gateway(self):
        cmd = mlxsh.tunnel_run_command("mlxsh", 41377)
        self.assertIn("http://127.0.0.1:41377", cmd)
        self.assertEqual(cmd[1:3], ["tunnel", "run"])

    def test_a_custom_command_replaces_cloudflared(self):
        reg = mlxsh.load_registry()
        reg["tunnel_cmd"] = "tailscale funnel --bg --https=443 localhost:{port}"
        mlxsh.save_registry(reg)
        self.assertEqual(mlxsh.tunnel_run_command("mlxsh", 41377),
                         ["tailscale", "funnel", "--bg", "--https=443",
                          "localhost:41377"])

    def test_the_help_says_what_it_depends_on(self):
        for cmd in ("gateway", "tunnel"):
            _usage, what, _examples = mlxsh.COMMAND_HELP[cmd]
            self.assertIn("cloudflared", what, cmd)
            self.assertNotIn("brew", what, cmd)
        self.assertIn(mlxsh.CLOUDFLARED_DOCS, mlxsh.COMMAND_HELP["tunnel"][1])

    def test_the_hint_names_the_missing_piece(self):
        saved = mlxsh.cloudflared
        try:
            mlxsh.cloudflared = lambda: None
            missing = " ".join(mlxsh.tunnel_hint())
            self.assertIn("cloudflared", missing)
            self.assertIn(mlxsh.CLOUDFLARED_DOCS, missing)
            self.assertNotIn("brew", missing)   # not everyone has homebrew
            mlxsh.cloudflared = lambda: "/opt/homebrew/bin/cloudflared"
            present = " ".join(mlxsh.tunnel_hint())
            self.assertIn("tunnel --quick", present)
            self.assertNotIn(mlxsh.CLOUDFLARED_DOCS, present)
        finally:
            mlxsh.cloudflared = saved

    def test_quick_needs_no_hostname_or_account(self):
        cmd = mlxsh.tunnel_run_command("mlxsh", 41377, quick=True)
        self.assertEqual(cmd[1:], ["tunnel", "--url", "http://127.0.0.1:41377"])
        self.assertNotIn("run", cmd)

    def test_the_address_is_read_out_of_the_log(self):
        log = ("2026-08-04 INF Thank you for trying Cloudflare Tunnel.\n"
               "2026-08-04 INF |  https://loud-quiet-mango-tree.trycloudflare.com"
               "  |\n2026-08-04 INF Registered tunnel connection\n")
        self.assertEqual(mlxsh.url_from_log(log),
                         "https://loud-quiet-mango-tree.trycloudflare.com")

    def test_a_tailscale_address_is_read_too(self):
        self.assertEqual(
            mlxsh.url_from_log("Available within your tailnet:\n"
                               "https://ans-mac.tail1234.ts.net/\n"),
            "https://ans-mac.tail1234.ts.net/")

    def test_no_address_in_the_log(self):
        self.assertIsNone(mlxsh.url_from_log("starting up\nconnected\n"))

    def test_starting_without_a_hostname_explains_itself(self):
        with captured() as (_out, err):
            mlxsh.start_tunnel()
        self.assertIn("no hostname yet", err.getvalue())

    def test_setup_needs_cloudflared(self):
        saved = mlxsh.cloudflared
        mlxsh.cloudflared = lambda: None
        try:
            with captured() as (_out, err):
                mlxsh.tunnel_setup("llm.example.com")
        finally:
            mlxsh.cloudflared = saved
        self.assertIn("cloudflared is not installed", err.getvalue())

    def test_setup_rejects_something_that_is_not_a_hostname(self):
        saved = mlxsh.cloudflared
        mlxsh.cloudflared = lambda: "/usr/local/bin/cloudflared"
        try:
            with captured() as (_out, err):
                mlxsh.tunnel_setup("notahostname")
        finally:
            mlxsh.cloudflared = saved
        self.assertIn("give the hostname", err.getvalue())


class GatewayOverHttp(TempHome):
    """The whole path: a real request through the gateway to a fake upstream."""

    def setUp(self):
        super().setUp()
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        self.seen = []
        seen = self.seen

        class Upstream(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                seen.append(json.loads(body))
                if json.loads(body).get("stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    for word in ("one ", "two ", "three"):
                        piece = ("data: " + json.dumps(
                            {"choices": [{"delta": {"content": word}}]}) + "\n\n")
                        chunk = piece.encode()
                        self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                        self.wfile.flush()
                    done = b"data: [DONE]\n\n"
                    self.wfile.write(b"%x\r\n%s\r\n0\r\n\r\n" % (len(done), done))
                    return
                payload = json.dumps({"choices": [{"message":
                                     {"content": "hello"}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        self.upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
        self.upstream.daemon_threads = True
        threading.Thread(target=self.upstream.serve_forever, daemon=True).start()

        self._fns = (mlxsh.pid_alive, mlxsh.pid_command, mlxsh.port_owner)
        mlxsh.pid_alive = lambda pid: True
        mlxsh.pid_command = lambda pid: "python -m mlx_lm server --model a/b"
        mlxsh.port_owner = lambda port: None
        mlxsh.write_state({"pid": 999, "mode": "lm", "model": "org/the-model",
                           "port": self.upstream.server_address[1],
                           "host": "127.0.0.1", "engine": "mlx_lm",
                           "started": 1})

        self.gateway = ThreadingHTTPServer(("127.0.0.1", 0),
                                           mlxsh.gateway_handler())
        self.gateway.daemon_threads = True
        threading.Thread(target=self.gateway.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.gateway.server_address[1]}"
        self.key = mlxsh.api_key()

    def tearDown(self):
        self.gateway.shutdown()
        self.upstream.shutdown()
        mlxsh.pid_alive, mlxsh.pid_command, mlxsh.port_owner = self._fns
        super().tearDown()

    def call(self, path, key=None, payload=None, stream=False):
        import urllib.error
        import urllib.request
        data = json.dumps(payload).encode() if payload is not None else None
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        req = urllib.request.Request(self.base + path, data=data,
                                     headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                # read inside the block: the response closes on exit
                return r.status, (list(r) if stream else r.read())
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    def test_no_key_is_refused(self):
        code, body = self.call("/v1/models")
        self.assertEqual(code, 401)
        self.assertIn("api key", json.loads(body)["error"]["message"])

    def test_wrong_key_is_refused(self):
        self.assertEqual(self.call("/v1/models", "mlxsh-nope")[0], 401)

    def test_health_needs_no_key(self):
        code, body = self.call("/healthz")
        self.assertEqual(code, 200)
        self.assertEqual(json.loads(body)["status"], "ok")

    def test_models_with_a_key(self):
        code, body = self.call("/v1/models", self.key)
        self.assertEqual(code, 200)
        self.assertEqual([m["id"] for m in json.loads(body)["data"]],
                         ["org/the-model"])

    def test_a_request_reaches_the_upstream(self):
        code, body = self.call("/v1/chat/completions", self.key,
                               {"model": "the-model", "messages": []})
        self.assertEqual(code, 200)
        self.assertEqual(json.loads(body)["choices"][0]["message"]["content"],
                         "hello")
        self.assertEqual(self.seen[-1]["model"], "org/the-model")

    def test_an_unknown_model_is_not_forwarded(self):
        code, body = self.call("/v1/chat/completions", self.key,
                               {"model": "not/loaded", "messages": []})
        self.assertEqual(code, 404)
        self.assertIn("no loaded model matches",
                      json.loads(body)["error"]["message"])
        self.assertEqual(self.seen, [])

    def test_streaming_is_passed_through_in_pieces(self):
        code, lines = self.call("/v1/chat/completions", self.key,
                                {"model": "the-model", "messages": [],
                                 "stream": True}, stream=True)
        self.assertEqual(code, 200)
        chunks = [line for line in lines if line.startswith(b"data: ")]
        self.assertEqual(len(chunks), 4)  # three words and [DONE]
        self.assertIn(b"three", chunks[2])


class HelpAndStreams(TempHome):
    """Conventions a command line is expected to follow."""

    def test_asking_for_help_never_acts(self):
        acted = []
        ctl = mlxsh.Ctl(tui=False)
        saved = mlxsh.serve
        mlxsh.serve = lambda *a, **k: acted.append(a)
        try:
            for line in ("lm --help", "lm -h", "vision --help"):
                with captured() as (out, _err):
                    mlxsh.dispatch(ctl, line, interactive=False)
                self.assertIn("Serve a model", out.getvalue())
        finally:
            mlxsh.serve = saved
        self.assertEqual(acted, [])

    def test_help_for_a_destructive_command_deletes_nothing(self):
        removed = []
        ctl = mlxsh.Ctl(tui=False)
        saved = mlxsh.remove
        mlxsh.remove = lambda *a, **k: removed.append(a)
        try:
            with captured() as (out, _err):
                mlxsh.dispatch(ctl, "rm --help", interactive=False)
        finally:
            mlxsh.remove = saved
        self.assertEqual(removed, [])
        self.assertIn("Delete a model", out.getvalue())

    def test_every_command_documents_itself(self):
        for cmd in mlxsh.COMMANDS:
            if cmd in ("help", "exit"):
                continue
            self.assertIn(cmd, mlxsh.COMMAND_HELP, cmd)
            usage, what, examples = mlxsh.COMMAND_HELP[cmd]
            self.assertTrue(usage.startswith(cmd), cmd)
            # a full stop, or a link on the last line
            self.assertTrue(what.endswith((".", "/")), cmd)
            self.assertTrue(all(e.startswith("mlxsh ") for e in examples), cmd)

    def test_diagnostics_go_to_stderr(self):
        with captured() as (out, err):
            mlxsh.warn("something broke")
            mlxsh.note("  a hint")
        self.assertEqual(out.getvalue(), "")
        self.assertIn("something broke", err.getvalue())
        self.assertIn("a hint", err.getvalue())

    def test_a_typo_suggests_the_command(self):
        ctl = mlxsh.Ctl(tui=False)
        with captured() as (_out, err):
            mlxsh.dispatch(ctl, "statsu", interactive=False)
        self.assertIn("did you mean status", err.getvalue())


class ServerCommandMatching(unittest.TestCase):
    """What counts as an mlx server, since stop signals a process group."""

    def setUp(self):
        self._alive, self._cmd = mlxsh.pid_alive, mlxsh.pid_command
        mlxsh.pid_alive = lambda pid: True

    def tearDown(self):
        mlxsh.pid_alive, mlxsh.pid_command = self._alive, self._cmd

    def check(self, command):
        mlxsh.pid_command = lambda pid: command
        return mlxsh.is_our_server(1)

    def test_the_forms_we_launch(self):
        self.assertTrue(self.check("/x/python -m mlx_lm server --model a/b"))
        self.assertTrue(self.check("/x/python -m mlx_vlm.server --model a/b"))

    def test_console_scripts_started_by_hand(self):
        self.assertTrue(self.check("/usr/local/bin/mlx_lm.server --model a/b"))
        self.assertTrue(self.check("/opt/venv/bin/mlx_vlm.server --port 8080"))

    def test_other_mlx_commands_are_not_servers(self):
        self.assertFalse(self.check("/x/python -m mlx_lm chat --model a/b"))
        self.assertFalse(self.check("/x/python -m mlx_lm generate --prompt hi"))

    def test_unrelated_processes_that_merely_mention_it(self):
        self.assertFalse(self.check("vim notes-on-mlx_lm-server.md"))
        self.assertFalse(self.check("grep -r 'mlx_lm server' ~/src"))
        self.assertFalse(self.check("/Applications/Mail.app/Contents/MacOS/Mail"))


class PickerDoesNotTouchTheRegistry(TempHome):
    """Transient display state used to be written into models.json."""

    def test_choose_model_leaves_entries_alone(self):
        reg = mlxsh.load_registry()
        reg["models"] = [{"repo": "a/b", "label": "b", "vision": False}]
        mlxsh.save_registry(reg)
        mlxsh.choose_model(reg)  # no tty: returns None, must not mutate
        mlxsh.save_registry(reg)
        stored = json.loads(mlxsh.REGISTRY.read_text())["models"][0]
        self.assertEqual(set(stored), {"repo", "label", "vision"})


class RegistryTypes(TempHome):
    """The registry is a text file people edit by hand."""

    def test_a_string_port_is_coerced(self):
        reg = mlxsh.load_registry()
        reg["port"] = "8080"
        mlxsh.save_registry(reg)
        self.assertEqual(mlxsh.setting("port"), 8080)

    def test_nonsense_falls_back_to_the_default(self):
        reg = mlxsh.load_registry()
        reg["port"] = "eighty eighty"
        mlxsh.save_registry(reg)
        with captured() as (_out, err):
            self.assertEqual(mlxsh.setting("port"), 41277)
        self.assertIn("not a valid", err.getvalue())


class LogTail(TempHome):
    def write_log(self, lines):
        mlxsh.LOG.parent.mkdir(parents=True, exist_ok=True)
        mlxsh.LOG.write_text("\n".join(lines) + "\n")

    def test_last_lines_of_a_large_file(self):
        self.write_log([f"line {i} " + "x" * 200 for i in range(20_000)])
        out = mlxsh.tail_log(5).splitlines()
        self.assertEqual(len(out), 5)
        self.assertTrue(out[-1].startswith("line 19999"))

    def test_small_file(self):
        self.write_log(["one", "two"])
        self.assertEqual(mlxsh.tail_log(10), "one\ntwo")

    def test_missing_file(self):
        self.assertIn("no log", mlxsh.tail_log())

    def test_rotation(self):
        mlxsh.LOG.parent.mkdir(parents=True, exist_ok=True)
        mlxsh.LOG.write_bytes(b"y" * (mlxsh.LOG_MAX + 10))
        mlxsh.rotate_log()
        self.assertFalse(mlxsh.LOG.exists())
        self.assertTrue(mlxsh.LOG.with_suffix(".log.1").exists())

    def test_no_rotation_below_the_cap(self):
        self.write_log(["small"])
        mlxsh.rotate_log()
        self.assertTrue(mlxsh.LOG.exists())


class Dispatch(unittest.TestCase):
    def test_every_command_has_a_handler(self):
        for cmd in mlxsh.COMMANDS:
            if cmd in ("exit", "shell"):
                continue
            self.assertTrue(hasattr(mlxsh.Ctl, "do_" + cmd), cmd)

    def test_every_alias_points_at_a_command(self):
        for alias, target in mlxsh.ALIASES.items():
            self.assertIn(target, mlxsh.COMMANDS, f"{alias} -> {target}")

    def test_palette_entries_are_commands(self):
        for name, _ in mlxsh.PALETTE:
            self.assertIn(name, mlxsh.COMMANDS, name)


class TerminalHelpers(unittest.TestCase):
    def test_visible_len_ignores_colour(self):
        self.assertEqual(mlxsh.visible_len("\033[1mhello\033[0m"), 5)

    def test_truncate_keeps_codes(self):
        out = mlxsh.truncate("\033[1mhello world\033[0m", 6)
        self.assertIn("\033[1m", out)
        self.assertLessEqual(mlxsh.visible_len(out), 6)

    def test_truncate_leaves_short_strings(self):
        self.assertEqual(mlxsh.truncate("short", 40), "short")


class KeyReading(unittest.TestCase):
    """Arrow keys arrive as escape sequences. Reading them through a buffered
    text stream loses the tail and types "[A" into the filter."""

    def key_from(self, raw: bytes) -> str:
        r, w = os.pipe()
        os.write(w, raw)
        os.close(w)
        try:
            return mlxsh.read_key(r)
        finally:
            os.close(r)

    def test_arrows(self):
        self.assertEqual(self.key_from(b"\x1b[A"), mlxsh.UP)
        self.assertEqual(self.key_from(b"\x1b[B"), mlxsh.DOWN)

    def test_application_cursor_mode(self):
        self.assertEqual(self.key_from(b"\x1bOA"), mlxsh.UP)
        self.assertEqual(self.key_from(b"\x1bOB"), mlxsh.DOWN)

    def test_paging_and_jumps(self):
        self.assertEqual(self.key_from(b"\x1b[5~"), mlxsh.PGUP)
        self.assertEqual(self.key_from(b"\x1b[6~"), mlxsh.PGDN)
        self.assertEqual(self.key_from(b"\x1b[H"), mlxsh.HOME_KEY)
        self.assertEqual(self.key_from(b"\x1b[F"), mlxsh.END_KEY)

    def test_plain_characters(self):
        self.assertEqual(self.key_from(b"q"), "q")
        self.assertEqual(self.key_from(b"\r"), "\r")
        self.assertEqual(self.key_from(b" "), " ")

    def test_bare_escape(self):
        self.assertEqual(self.key_from(b"\x1b"), "\x1b")

    def test_utf8(self):
        self.assertEqual(self.key_from("é".encode()), "é")

    def test_eof_reads_as_interrupt(self):
        self.assertEqual(self.key_from(b""), "\x03")


class Formatting(unittest.TestCase):
    def test_gb(self):
        self.assertEqual(mlxsh.gb(15_600_000_000), "15.6 GB")
        self.assertEqual(mlxsh.gb(0), "?")
        self.assertEqual(mlxsh.gb(None), "?")

    def test_short(self):
        self.assertEqual(mlxsh.short("mlx-community/gemma-4-31b"), "gemma-4-31b")
        self.assertEqual(mlxsh.short("gemma-4-31b"), "gemma-4-31b")

    def test_fit_marks_track_this_machine(self):
        self.assertIn("fits", mlxsh.fit_mark(mlxsh.mem_comfy() - 1))
        self.assertIn("tight", mlxsh.fit_mark(mlxsh.mem_comfy() + 1))
        self.assertIn("big", mlxsh.fit_mark(mlxsh.mem_tight() + 1))


class Hardware(unittest.TestCase):
    def test_memory_is_plausible(self):
        self.assertGreater(mlxsh.total_memory(), 1e9)
        self.assertLess(mlxsh.mem_comfy(), mlxsh.mem_tight())
        self.assertLess(mlxsh.mem_tight(), mlxsh.MEM_TOTAL)

    def test_machine_name_has_a_size(self):
        self.assertRegex(mlxsh.machine_name(), r"\d+ GB$")


class Rendering(unittest.TestCase):
    def test_hub_row(self):
        row = {"repo": "mlx-community/x", "vision": True, "size": 1e10,
               "downloads": 12, "updated": datetime(2026, 1, 2, tzinfo=timezone.utc), "have": True}
        out = mlxsh.render_hub(row)
        self.assertIn("mlx-community/x", out)
        self.assertIn("downloaded", out)

    def test_hub_row_without_date_or_flag(self):
        row = {"repo": "a/b", "vision": False, "size": 0, "downloads": 0,
               "updated": None}
        self.assertIn("a/b", mlxsh.render_hub(row))

    def test_model_row_minimal(self):
        self.assertIn("thing", mlxsh.render_model({"repo": "a/thing"},
                                                  {"defaults": {}}, {}))


class Voice(unittest.TestCase):
    def test_output_stays_plain(self):
        text = (ROOT / "mlxsh.py").read_text()
        for ch in ("—", "·", "★", "▸", "●", "○"):
            self.assertNotIn(ch, text, f"{ch!r} in mlxsh.py")


if __name__ == "__main__":
    unittest.main()

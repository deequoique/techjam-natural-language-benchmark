"""Explicit adapter boundary for loading an external TechJam Agent."""

from __future__ import annotations

import importlib
import inspect
import json
import os
from pathlib import Path
import selectors
import sys
import subprocess
import tempfile
from types import ModuleType
from typing import Any


class AgentLoadError(RuntimeError):
    pass


class AgentProcessError(RuntimeError):
    pass


def _import_agent_module(agent_repo: Path) -> ModuleType:
    if not agent_repo.is_dir():
        raise AgentLoadError(f"agent repository does not exist: {agent_repo}")
    # A CLI process normally imports one Agent.  Clearing an existing starter
    # module avoids accidentally loading a sibling checkout that happened to
    # be imported by a caller earlier in the process.
    old_modules = [name for name in sys.modules if name == "starter" or name.startswith("starter.")]
    for name in old_modules:
        sys.modules.pop(name, None)
    repository = str(agent_repo)
    sys.path.insert(0, repository)
    try:
        return importlib.import_module("starter.agent")
    except Exception as exc:
        raise AgentLoadError(f"could not import starter.agent from {agent_repo}: {exc}") from exc
    finally:
        try:
            sys.path.remove(repository)
        except ValueError:
            pass


def load_agent_class(agent_repo: str | Path) -> type:
    module = _import_agent_module(Path(agent_repo).expanduser().resolve())
    agent_class = getattr(module, "Agent", None)
    if not inspect.isclass(agent_class):
        raise AgentLoadError("starter.agent does not export a class named Agent")
    for method_name in ("reset", "respond"):
        method = getattr(agent_class, method_name, None)
        if not callable(method):
            raise AgentLoadError(f"Agent is missing callable {method_name}()")
    return agent_class


def load_agent(agent_repo: str | Path, catalog_path: str | Path) -> Any:
    """Instantiate the external Agent without writing into its checkout."""

    if os.environ.get("NL_BENCHMARK_WORKER") != "1":
        raise AgentLoadError("external Agent loading is restricted to the IPC worker process")
    agent_class = load_agent_class(agent_repo)
    catalog = str(Path(catalog_path).expanduser().resolve())
    try:
        signature = inspect.signature(agent_class)
    except (TypeError, ValueError):
        signature = None
    try:
        if signature is not None and "catalog_path" in signature.parameters:
            return agent_class(catalog_path=catalog)
        # The official Agent accepts a positional catalog path.  A no-arg
        # fallback keeps small deterministic test doubles convenient.
        if signature is not None and len(signature.parameters) == 0:
            return agent_class()
        return agent_class(catalog)
    except Exception as exc:
        raise AgentLoadError(f"could not instantiate Agent: {exc}") from exc


class SubprocessAgent:
    """JSONL IPC client for an untrusted external Agent.

    The benchmark parent never imports the external ``starter`` package.  A
    short-lived worker process receives only reset/respond protocol messages;
    target IDs and evaluator state remain in the parent process.  The worker
    runs with ``-B``/``PYTHONDONTWRITEBYTECODE`` and a disposable cwd so normal
    imports cannot create bytecode in the submitted Agent checkout.
    """

    def __init__(
        self,
        agent_repo: str | Path,
        catalog_path: str | Path,
        *,
        timeout: float = 120.0,
        force_intent_model: bool = False,
        intent_confidence: float | None = None,
    ):
        self.agent_repo = Path(agent_repo).expanduser().resolve()
        self.catalog_path = Path(catalog_path).expanduser().resolve()
        if not self.agent_repo.is_dir():
            raise AgentProcessError(f"agent repository does not exist: {self.agent_repo}")
        if not self.catalog_path.is_file():
            raise AgentProcessError(f"catalog does not exist: {self.catalog_path}")
        self.timeout = max(float(timeout), 1.0)
        self.last_diagnostics: dict[str, Any] = {}
        self.force_intent_model = bool(force_intent_model)
        self.intent_confidence = None if intent_confidence is None else float(intent_confidence)
        if self.intent_confidence is not None and not 0.0 <= self.intent_confidence <= 1.0:
            raise AgentProcessError("intent_confidence must be between 0.0 and 1.0")
        self._tempdir = tempfile.TemporaryDirectory(prefix="nl-benchmark-agent-")
        package_root = Path(__file__).resolve().parents[1]
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONNOUSERSITE"] = "1"
        env["NL_BENCHMARK_WORKER"] = "1"
        current_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(package_root), current_pythonpath)))
        try:
            command = [
                    sys.executable,
                    "-B",
                    "-m",
                    "nl_benchmark.worker",
                    "--agent-repo",
                    str(self.agent_repo),
                    "--catalog",
                    str(self.catalog_path),
                ]
            if self.force_intent_model:
                command.append("--force-intent-model")
            if self.intent_confidence is not None:
                command.extend(("--intent-confidence", str(self.intent_confidence)))
            self._process = subprocess.Popen(
                command,
                cwd=self._tempdir.name,
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
        except OSError:
            self._tempdir.cleanup()
            raise

    def _request(self, payload: dict[str, Any]) -> Any:
        process = self._process
        if process.poll() is not None or process.stdin is None or process.stdout is None:
            raise AgentProcessError("Agent worker is not running")
        try:
            process.stdin.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
            process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise AgentProcessError(f"could not send request to Agent worker: {exc}") from exc
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        try:
            events = selector.select(self.timeout)
        finally:
            selector.close()
        if not events:
            self.close()
            raise AgentProcessError(f"Agent worker timed out after {self.timeout:.1f}s")
        line = process.stdout.readline()
        if not line:
            raise AgentProcessError("Agent worker exited without a response")
        try:
            response = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AgentProcessError(f"Agent worker returned invalid JSON: {line[:500]!r}") from exc
        if not isinstance(response, dict):
            raise AgentProcessError("Agent worker response is not an object")
        if not response.get("ok", False):
            raise AgentProcessError(str(response.get("error") or "Agent worker request failed"))
        diagnostics = response.get("diagnostics")
        self.last_diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
        return response.get("response")

    def reset(self, session_id: str, user_profile: dict[str, Any]) -> None:
        self._request({"op": "reset", "session_id": session_id, "user_profile": user_profile})

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> Any:
        return self._request({"op": "respond", "session_id": session_id, "user_message": user_message, "turn": int(turn), "top_k": int(top_k)})

    def close(self) -> None:
        process = getattr(self, "_process", None)
        if process is not None and process.poll() is None:
            try:
                if process.stdin:
                    process.stdin.close()
            except OSError:
                pass
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        if process is not None:
            for stream in (process.stdin, process.stdout, process.stderr):
                try:
                    if stream is not None:
                        stream.close()
                except OSError:
                    pass
        tempdir = getattr(self, "_tempdir", None)
        if tempdir is not None:
            tempdir.cleanup()

    def __enter__(self) -> "SubprocessAgent":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

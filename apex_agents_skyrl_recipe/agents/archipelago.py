"""
Archipelago agent — MCP tool-calling agent for Archipelago sandbox environments.

Connects to the Archipelago sandbox's MCP gateway via Modal tunnel,
loads available tools, and runs an LLM + tool-calling loop.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import shlex
import time
from pathlib import Path
from typing import Any

import httpx
import litellm
from anyio import BrokenResourceError, ClosedResourceError
from fastmcp import Client as FastMCPClient
from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.llms.base import ContextLengthExceededError
from harbor.models.agent.context import AgentContext
from litellm.experimental_mcp_client import call_openai_tool, load_mcp_tools
from mcp.shared.exceptions import McpError
from mcp.types import TextContent

from apex_agents_skyrl_recipe.agents.llm import (
    generate_response,
    generate_response_tito,
    with_llm_retry,
)
from apex_agents_skyrl_recipe.agents.tool_result import truncate_tool_result_messages
from apex_agents_skyrl_recipe.agents.utils import (
    content_blocks_to_messages,
    flatten_envelope_tools,
    get_error_tool_message,
    is_fatal_mcp_error,
    rewrap_envelope_arguments,
    save_trajectory_to_disk,
)

# ── Default system prompt ─────────────────────────────────────────────

DEFAULT_SYSTEM_PROMPT = """\
You are an AI assistant that completes tasks by reasoning and using tools.

## Think Before Acting

Before making tool calls, briefly explain your reasoning in 1-3 sentences:
- What you learned from the previous step
- What you're doing next and why

Don't over-explain. Be concise but show your thinking.

## Tools

You have access to various tools for file management, code execution, reading \
documents, spreadsheets, and more. Use the tools provided to complete the task.

## Workflow

1. Analyze the task and identify what information or actions are needed
2. Use available tools to gather information, read files, and execute actions
3. When done, provide your final answer as a text response (no tool call)

## Rules

- Show your work for calculations
- When you are done, respond with your final answer as text without any tool calls
"""

# Appended to the most recent tool observation once the TITO context window
# nears its limit, nudging the agent to wrap up. "{pct}" is replaced with the
# integer percent of the window projected free at the next turn's start (see
# the injection site); custom text may include it too, or omit it (no-op).
DEFAULT_BUDGET_WARNING = (
    "[SYSTEM NOTICE] You are almost out of context budget (~{pct}% of the "
    "context window remains). Limit the number of tool calls and provide your "
    "final answer soon."
)

# Python packages the `code_execution` tool commonly needs.
DEFAULT_PREINSTALL_PACKAGES = [
    "pandas",
    "openpyxl",
    "numpy",
    "python-docx",
    "PyPDF2",
    "xlrd",
    "pdfplumber",
    "pymupdf",
    "pypdf",
    "python-pptx",
    "pillow",
    "pytesseract",
    "xlsxwriter",
]


class ArchipelagoAgent(BaseAgent):
    """
    MCP tool-calling agent for Mercor sandbox environments.

    Connects to the sandbox's MCP gateway via Modal tunnel to discover
    and execute tools.
    """

    SUPPORTS_ATIF = False  # We populate metadata manually

    def __init__(
        self,
        logs_dir: Path,
        model_name: str,
        # MCP configuration
        api_base: str | None = None,
        session_id: str | None = None,
        session_affinity_backend: str = "skyrl",
        # Agent loop configuration
        max_steps: int = 30,
        tool_call_timeout: int = 60,
        # LLM configuration
        # LLM retry configuration
        max_timeout_retries: int = 3,
        max_rate_limit_retries: int = 5,
        max_transient_retries: int = 3,
        # Mercor-specific
        system_prompt: str | None = None,
        model_info: dict | None = None,
        llm_kwargs: dict | None = None,
        store_reasoning_in_messages: bool = True,
        gui_enabled: bool = True,
        # Hints configuration
        use_hints: bool = False,
        # Extra prompt appended verbatim to the task instruction (e.g. a
        # text-only / no-vision note). Independent of use_hints.
        extra_prompt: str | None = None,
        # Salt -- so we don't share prefix cache across model versions in fully async.
        cache_salt: str | None = None,
        # TITO configuration
        use_tito: bool = False,
        tito_tokenizer_name: str | None = None,
        tito_tool_call_parser: str = "glm47",
        tito_reasoning_parser: str = "glm45",
        tito_dump_transitions: bool = True,
        # Budget warning (TITO only): once the fraction of the context window
        # still free drops to <= this ratio, inject a one-shot note telling the
        # agent to stop calling tools and finalize. None disables the feature.
        tito_budget_warning_ratio: float | None = None,
        tito_budget_warning_text: str | None = None,
        # Root-install these Python packages into the sandbox's system
        # interpreter at setup() so the `code_execution` tool can import them.
        # None -> DEFAULT_PREINSTALL_PACKAGES; [] -> disable (skip entirely).
        preinstall_packages: list[str] | None = None,
        # TITO sampling backend (used when use_tito=True):
        #   "openai_completions" — litellm /completions (local vLLM OR Fireworks)
        #   "tinker"             — Tinker SDK SamplingClient.sample_async
        tito_backend: str = "openai_completions",
        # API key for hosted openai_completions backends (e.g. Fireworks). If
        # api_key_env is given, the value is read from that env var at init.
        api_key: str | None = None,
        api_key_env: str | None = None,
        # Tinker backend config (tito_backend="tinker"): sample from a base model
        # or a trained checkpoint (model_path), against tinker_base_url.
        tinker_base_model: str | None = None,
        tinker_model_path: str | None = None,
        tinker_base_url: str | None = None,
        # Tool result truncation
        tool_result_max_chars: int | None = None,
        # When the model emits a <tool_call> block that the vLLM tool parser cannot
        # parse, tool_calls is empty and the trajectory would silently terminate at
        # reward 0. Instead, inject a corrective observation (as a role=tool message
        # so it survives TITO normalization) and let the model retry, up to
        # tito_max_repair_attempts times per trajectory.
        tito_repair_unparsed_tool_calls: bool = True,
        tito_max_repair_attempts: int = 2,
        **kwargs,
    ):
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)

        self._api_base = api_base
        self._session_id = session_id
        self._max_steps = max_steps

        if session_affinity_backend not in ("skyrl", "fireworks"):
            raise ValueError(
                "session_affinity_backend must be 'skyrl' or 'fireworks', got " f"{session_affinity_backend!r}"
            )

        # Keep every turn in one agent trajectory on the same rollout replica.
        # SkyRL's local router consumes request-body session/cache fields and
        # X-Session-ID. Fireworks uses two standard rollout headers instead;
        # sending the local-only body fields to its OpenAI API would make the
        # provider reject an otherwise valid /completions request.
        if session_id is not None:
            llm_kwargs = llm_kwargs or {}
            extra_headers = llm_kwargs.setdefault("extra_headers", {})
            if session_affinity_backend == "fireworks":
                extra_headers["x-multi-turn-session-id"] = str(session_id)
                extra_headers["x-session-affinity"] = str(session_id)
            else:
                extra_body = llm_kwargs.setdefault("extra_body", {})
                extra_body["session_id"] = session_id
                if cache_salt is not None:
                    extra_body["cache_salt"] = cache_salt
                    self.logger.info(f"Using cache salt: {cache_salt}")
                extra_headers["X-Session-ID"] = str(session_id)
        self._tool_call_timeout = tool_call_timeout
        self._system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self._gui_enabled = gui_enabled
        self._use_hints = use_hints
        self._extra_prompt = extra_prompt
        # Envelope handling: {tool_name: "input"/"request"} for the tools whose advertised schema we
        # flatten; re-wrapped on dispatch.
        self._wrapper_map: dict[str, str] = {}
        # can include timeout, reasoning_effort, sampling_params, etc.
        self._llm_kwargs = llm_kwargs or {}
        assert (
            "num_retries" not in self._llm_kwargs and "max_retries" not in self._llm_kwargs
        ), "num_retries and max_retries are not allowed in llm_kwargs. We set them to zero internally."
        self._store_reasoning_in_messages = store_reasoning_in_messages
        self._use_tito = use_tito
        self._tito_tokenizer = None
        self._tito_tool_call_parser = tito_tool_call_parser
        self._tito_reasoning_parser = tito_reasoning_parser
        self._tito_dump_transitions = tito_dump_transitions
        self._tito_budget_warning_ratio = tito_budget_warning_ratio
        self._tito_budget_warning_text = tito_budget_warning_text

        self._preinstall_packages = (
            list(DEFAULT_PREINSTALL_PACKAGES) if preinstall_packages is None else list(preinstall_packages)
        )

        # max_context_len for TITO: /v1/completions defaults to max_tokens=16,
        # so we compute max_tokens = max_context_len - prompt_len dynamically.
        if self._use_tito:
            assert model_info["max_input_tokens"] is not None, "max_input_tokens is required for TITO"
            self._max_context_len = model_info["max_input_tokens"]

        # Load the HF tokenizer for building token-id prompts + feeding the
        # vLLM parsers. For tito_backend="tinker" with no explicit tokenizer we
        # defer to the sampling client's own tokenizer (_init_tinker_sampling_client),
        # so a Tinker run needs no HF tokenizer at all.
        if use_tito and (tito_tokenizer_name or tito_backend != "tinker"):
            from transformers import AutoTokenizer

            tokenizer_name = tito_tokenizer_name or model_name.replace("text-completion-openai/", "")
            # local_files_only=True forces the tokenizer to read from the
            # local HF cache instead of hitting the Hub. Avoids rate-limiting
            # the HF token across many concurrent trials. Note: we only
            # constrain the tokenizer load — we don't set HF_HUB_OFFLINE
            # globally because the trainer in a sibling process still needs
            # to fetch model weights from the Hub on first launch.
            self._tito_tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name,
                trust_remote_code=True,
                local_files_only=True,
            )

        self._tito_repair_unparsed_tool_calls = tito_repair_unparsed_tool_calls
        self._tito_max_repair_attempts = tito_max_repair_attempts
        self._tool_result_max_chars = tool_result_max_chars

        # TITO backend + credentials.
        if tito_backend not in ("openai_completions", "tinker"):
            raise ValueError(f"Unknown tito_backend {tito_backend!r}; expected 'openai_completions' or 'tinker'.")
        self._tito_backend = tito_backend
        self._api_key = api_key or (os.environ.get(api_key_env) if api_key_env else None)
        self._tinker_base_model = tinker_base_model
        self._tinker_model_path = tinker_model_path
        self._tinker_base_url = tinker_base_url
        self._tinker_sampling_client = None

        # Wrap generate_response with instance-specific retry config.
        retry_decorator = with_llm_retry(
            logger=self.logger,
            max_timeout_retries=max_timeout_retries,
            max_rate_limit_retries=max_rate_limit_retries,
            max_transient_retries=max_transient_retries,
        )
        self._generate_response = retry_decorator(generate_response)
        self._generate_response_tito = retry_decorator(generate_response_tito)

        self._mcp_gateway_url: str | None = None
        self._task_slug: str | None = None
        self._env_vars: dict[str, str] = {}
        self._environment: BaseEnvironment | None = None
        self._expected_mcp_servers: list[str] = []

    @staticmethod
    def name() -> str:
        return "archipelago"

    def version(self) -> str | None:
        return "0.1.0"

    #####################################
    # setup() and its helper methods
    #####################################

    async def setup(self, environment: BaseEnvironment) -> None:
        """
        Initialize the Mercor sandbox:
        1. Read archipelago.json from the task directory to get task_slug
        2. Run start.sh <task-slug> to boot MCP servers
        3. Get tunnel URL for external MCP access
        4. Wait for MCP gateway readiness
        5. Capture initial snapshot for grading
        """
        self._environment = environment
        try:
            await self._setup_impl(environment)
        finally:
            # Download start.log as a separate artifact (best-effort).
            # Runs on both success and failure so the log is always available.
            try:
                dest = self.logs_dir / "start.log"
                await environment.download_file("/tmp/start.log", dest)
                self.logger.info(f"Saved start.log to {dest}")
            except Exception:
                self.logger.debug("Could not download start.log (file may not exist)")

    async def _setup_impl(self, environment: BaseEnvironment) -> None:
        # Bring up the Tinker sampling client before the agent loop needs it.
        if self._use_tito and self._tito_backend == "tinker" and self._tinker_sampling_client is None:
            self._init_tinker_sampling_client()

        task_dir = self._find_task_dir(environment)
        self._task_dir = task_dir
        archipelago_json = task_dir / "archipelago.json"
        if not archipelago_json.exists():
            raise FileNotFoundError(
                f"archipelago.json not found in {task_dir}. "
                "Was the task generated by the Archipelago-Harbor adapter?"
            )

        archipelago_meta = json.loads(archipelago_json.read_text())
        self._task_slug = archipelago_meta["task_slug"]
        self.logger.info(
            f"Loaded mercor metadata: task_slug={self._task_slug}, " f"ecr_image={archipelago_meta.get('ecr_image')}"
        )

        # Build env vars for the sandbox (required API keys).
        # If a key isn't in the environment, inject a dummy value so start.sh's
        # validate_secrets() doesn't abort.
        for key in archipelago_meta.get("required_env_keys", []):
            val = os.environ.get(key)
            self._env_vars[key] = val if val else "DUMMY_NOT_SET"

        # Determine per-server offline mode based on which API keys are available.
        # Each server has its own offline flag — only force offline for servers
        # whose specific key is missing. EDGAR needs no key (always online).
        _KEY_TO_OFFLINE_SED = {
            "FMP_API_KEY": "-e \"s/FMP_OFFLINE_MODE='false'/FMP_OFFLINE_MODE='true'/\"",
            "TERRAPIN_API_KEY": "-e \"s/TERRAPIN_OFFLINE='0'/TERRAPIN_OFFLINE='1'/\"",
        }
        missing_keys = [k for k, v in self._env_vars.items() if v == "DUMMY_NOT_SET"]
        online_keys = [k for k, v in self._env_vars.items() if v != "DUMMY_NOT_SET"]

        if online_keys:
            self.logger.info(f"API keys found (ONLINE): {online_keys}")
        if missing_keys:
            offline_servers = [k for k in missing_keys if k in _KEY_TO_OFFLINE_SED]
            passthrough_keys = [k for k in missing_keys if k not in _KEY_TO_OFFLINE_SED]
            if offline_servers:
                self.logger.warning(
                    f"API keys missing (OFFLINE): {offline_servers} — "
                    f"those servers will use cached/local data. "
                    f"Set these env vars or pass via --ae to enable online mode."
                )
            if passthrough_keys:
                # These are keys like SEARCH_MCP_GOOGLE_API_KEY and
                # SEARCH_MCP_GOOGLE_CSE_ID that have no offline mode sed
                # pattern. Dummy values are passed so start.sh validation
                # doesn't abort; the server will start but API calls will fail.
                # `SEARCH_MCP_GOOGLE_API_KEY` will be set as "DUMMY_NOT_SET" since it
                # is in archipelago.json. Only listed in world 223 in eval set, but the
                # actual environment does not need it.
                self.logger.warning(
                    f"API keys missing (no offline fallback): {passthrough_keys} — "
                    f"dummy values will be passed; server may fail at runtime."
                )

        # Build sed command: per-server offline patches + optionally disable GUI_ENABLED
        sed_parts = [_KEY_TO_OFFLINE_SED[k] for k in missing_keys if k in _KEY_TO_OFFLINE_SED]
        if not self._gui_enabled:
            # Use meta-tools (GUI_ENABLED=false) instead of individual tools.
            # Production start.sh sets GUI_ENABLED='true' for UI display, but LLM agents
            # work better with fewer, consolidated meta-tools.
            sed_parts.append("-e \"s/GUI_ENABLED='true'/GUI_ENABLED='false'/\"")
        else:
            self.logger.info("GUI_ENABLED=true: using full individual tool set")

        offline_sed = f"sed -i {' '.join(sed_parts)} /app/tools/start.sh 2>/dev/null; " if sed_parts else ""

        # Patch the PowerPoint slides server BEFORE start.sh boots it. The
        # dev-1928 (`dev-061726-*`) images ship a slides server that flattens
        # tool *output* schemas with optional strings marked `nullable:true`,
        # which fastmcp's output validation rejects as "None is not of type
        # 'string'" on every successful call. The fix script GATES on that code
        # path, so it is a verified no-op on the older `dev-*` / `eval-*` images.
        slides_fix_cmd = self._build_slides_fix_cmd()

        # Run start.sh in the background to boot the MCP servers.
        # Env vars are inlined so background processes inherit them.
        # If this fails, raise immediately — the trial-level retry in
        # _setup_environment_and_agent() will tear down the broken container
        # and recreate it from scratch.
        self.logger.info(f"Running start.sh {self._task_slug} (background)...")
        try:
            env_exports = " ".join(f"{k}={v}" for k, v in self._env_vars.items())
            cmd = (
                f"{offline_sed}"
                f"{slides_fix_cmd}"
                f"{env_exports + ' ' if env_exports else ''}"
                f"nohup /app/tools/start.sh {self._task_slug} "
                f"> /tmp/start.log 2>&1 &"
            )
            await environment.exec(cmd, timeout_sec=30)
            self.logger.info("start.sh launched in background")
        except Exception as e:
            raise RuntimeError(f"start.sh failed to launch: {e}") from e

        # Pre-install common Python packages into the code sandbox's system
        # interpreter (some images ship without them; see DEFAULT_PREINSTALL_PACKAGES). Runs as root
        # via environment.exec (the code_execution service user cannot write site-packages, and its
        # pip user-base is the locked /.apps_data). Idempotent + non-fatal.
        await self._preinstall_python_packages(environment)

        # Log environment config from the ECR image for debugging.
        # This captures the state *after* sed patching so we see actual values.
        try:
            result = await environment.exec(
                "cat /app/tools/mcp.json 2>/dev/null || echo '{}'; "
                "echo '---SECRETS---'; "
                "cat /app/tools/required_secrets.txt 2>/dev/null; "
                "echo '---FLAGS---'; "
                "grep -oE '(GUI_ENABLED|FMP_OFFLINE_MODE|TERRAPIN_OFFLINE|EDGAR_OFFLINE_MODE)=[^ ]*' /app/tools/start.sh 2>/dev/null",
                timeout_sec=15,
            )
            output = result.stdout or ""
            parts = output.split("---SECRETS---")
            mcp_raw = parts[0].strip() if len(parts) > 1 else ""
            rest = parts[1] if len(parts) > 1 else ""
            parts2 = rest.split("---FLAGS---")
            secrets_raw = parts2[0].strip() if len(parts2) > 1 else ""
            flags_raw = parts2[1].strip() if len(parts2) > 1 else ""

            try:
                mcp_data = json.loads(mcp_raw)
                servers = list(mcp_data.get("mcpServers", {}).keys())
            except json.JSONDecodeError:
                servers = []
            self._expected_mcp_servers = servers
            secrets = [s for s in secrets_raw.splitlines() if s.strip()]
            flags = dict(line.split("=", 1) for line in flags_raw.splitlines() if "=" in line)
            # Strip quotes from flag values
            flags = {k: v.strip("'\"") for k, v in flags.items()}

            self.logger.info(f"[env-config] MCP_SERVERS={json.dumps(servers)}")
            self.logger.info(f"[env-config] REQUIRED_SECRETS={json.dumps(secrets)}")
            self.logger.info(f"[env-config] ENV_FLAGS={json.dumps(flags)}")
        except Exception as e:
            self.logger.warning(f"Failed to read environment config: {e}")

        # Get tunnel URL for MCP gateway
        url = await environment.get_tunnel_url(8000)
        self._mcp_gateway_url = url + "/mcp/"
        self.logger.info(f"MCP gateway URL: {self._mcp_gateway_url[:80]}...")

        # Wait for MCP gateway to be ready
        await self._wait_for_mcp_ready(url, environment=environment)

        # Fix /filesystem file permissions so MCP servers can write.
        # ECR images often have files owned by root with mode 644, but MCP
        # servers run as service users (svc_excel, svc_word, etc.) that lack
        # write access.  This varies by image, so we fix unconditionally.
        try:
            # chmod -R a+rwX /filesystem:
            #   -R     recursive (all files and subdirectories)
            #   a+rw   add read+write for all users (owner, group, others)
            #   X      add execute only on directories (for traversal), not files
            await environment.exec(
                "chmod -R a+rwX /filesystem 2>/dev/null || true",
                timeout_sec=60,
            )
            self.logger.info("chmod -R a+rwX /filesystem — MCP servers can now write to all files")
        except Exception:
            self.logger.debug("Could not chmod /filesystem (directory may not exist)")

        # Capture initial snapshot for grading (stored inside sandbox).
        # Skip if all verifiers are "Final Answer Only" — no file diff needed.
        if archipelago_meta.get("needs_snapshot", True):
            await self._capture_initial_snapshot(environment)
        else:
            self.logger.info("Skipping initial snapshot (all verifiers are Final Answer Only)")

    def _find_task_dir(self, environment: BaseEnvironment) -> Path:
        """Find the task directory from environment context."""
        env_dir = getattr(environment, "environment_dir", None)
        if env_dir and env_dir.parent.exists():
            return env_dir.parent
        raise RuntimeError(
            "Could not determine task directory from environment. " "Expected environment_dir to be set."
        )

    async def _wait_for_mcp_ready(
        self,
        base_url: str,
        environment: BaseEnvironment,
        timeout: int = 900,
    ) -> None:
        """Wait for the MCP gateway to become ready.

        1. Wait for /health to return 200 (gateway process is alive)
        2. Wait for POST /mcp/ tools/list to return non-404 (MCP endpoint registered)

        After passing, captures a resource snapshot for the trial record.
        start.log is downloaded as a separate artifact in setup().
        """
        start = time.time()
        health_ok = False

        async with httpx.AsyncClient() as client:
            while time.time() - start < timeout:
                elapsed = int(time.time() - start)

                if not health_ok:
                    try:
                        resp = await client.get(
                            f"{base_url}/health",
                            timeout=5,
                        )
                        if resp.status_code == 200:
                            health_ok = True
                            self.logger.info(f"Runner healthy ({elapsed}s)")
                    except httpx.RequestError:
                        pass
                    await asyncio.sleep(3)
                    continue

                try:
                    resp = await client.post(
                        f"{base_url}/mcp/",
                        json={"jsonrpc": "2.0", "method": "tools/list", "id": 1},
                        timeout=10,
                    )
                    if resp.status_code != 404:
                        self.logger.info(f"MCP registered ({elapsed}s)")
                        break
                except httpx.RequestError:
                    pass
                await asyncio.sleep(5)
            else:
                self.logger.error(f"MCP gateway not ready after {timeout}s")
                raise TimeoutError(f"MCP gateway not ready after {timeout}s at {base_url}")

        # Resource snapshot after startup (start.log is downloaded separately in setup())
        # Samples CPU over 1s to compute instantaneous utilization %.
        try:
            result = await environment.exec(
                "T1=$(cat /sys/fs/cgroup/cpuacct/cpuacct.usage 2>/dev/null || echo 0) && "
                "sleep 1 && "
                "T2=$(cat /sys/fs/cgroup/cpuacct/cpuacct.usage 2>/dev/null || echo 0) && "
                "PCT=$(( (T2 - T1) / 10000000 )) && "
                "MEM_MB=$(( $(cat /sys/fs/cgroup/memory/memory.usage_in_bytes 2>/dev/null || echo 0) / 1048576 )) && "
                'echo "cpu: ${PCT}% (1s sample), nproc=$(nproc 2>/dev/null || echo n/a)" && '
                'echo "memory: ${MEM_MB} MB"',
                timeout_sec=10,
            )
            self.logger.info(f"Resource snapshot at startup:\n{(result.stdout or '').strip()}")
        except Exception:
            pass

    async def _capture_initial_snapshot(self, environment: BaseEnvironment) -> None:
        """Capture initial snapshot inside the sandbox via localhost.

        Calls the MCP gateway's /data/snapshot endpoint from within the
        sandbox itself (python urllib to localhost:8000), so the snapshot
        data never leaves the machine — no external HTTP transfer.
        """
        self.logger.info("Capturing initial snapshot (local)...")
        try:
            await environment.exec(
                'python3 -c "'
                "import urllib.request; "
                "req = urllib.request.Request('http://localhost:8000/data/snapshot', method='POST'); "
                "resp = urllib.request.urlopen(req, timeout=120); "
                "data = resp.read(); "
                "open('/logs/agent/initial_snapshot.tar.gz', 'wb').write(data); "
                "print(f'Snapshot: {len(data)} bytes')"
                '"',
                timeout_sec=120,
            )
            self.logger.info("Initial snapshot captured")
        except Exception as e:
            self.logger.warning(f"Failed to capture initial snapshot: {e}")

    def _maybe_add_hints(self, instruction: str) -> str:
        """Append per-verifier hints to instruction if available."""
        task_dir = getattr(self, "_task_dir", None)
        if not task_dir:
            return instruction

        hints_path = task_dir / "tests" / "hints.json"
        if not hints_path.exists():
            return instruction

        try:
            hints = json.loads(hints_path.read_text())
        except (json.JSONDecodeError, OSError):
            return instruction

        active_hints = [h["hint"] for h in hints if h.get("hint")]
        if not active_hints:
            return instruction

        self.logger.info(f"Injecting {len(active_hints)} hints for {task_dir.name}")
        lines = "\n".join(f"- {hint}" for hint in active_hints)
        return instruction + f"\n\n---\n\n## Key Requirements\n{lines}"

    def _init_tinker_sampling_client(self) -> None:
        """Create the Tinker SDK sampling client (tito_backend='tinker').

        Samples from a base model (``tinker_base_model``) or a trained checkpoint
        (``tinker_model_path``) against ``tinker_base_url`` (falls back to
        ``TINKER_BASE_URL`` env or the Tinker prod URL). Reads ``TINKER_API_KEY``
        from the environment (Tinker SDK convention).
        """
        import tinker

        base_url = (
            self._tinker_base_url
            or os.environ.get("TINKER_BASE_URL")
            or "https://tinker.thinkingmachines.dev/services/tinker-prod"
        )
        service_client = tinker.ServiceClient(base_url=base_url)
        if self._tinker_model_path:
            self._tinker_sampling_client = service_client.create_sampling_client(model_path=self._tinker_model_path)
        elif self._tinker_base_model:
            self._tinker_sampling_client = service_client.create_sampling_client(base_model=self._tinker_base_model)
        else:
            raise ValueError("tito_backend='tinker' requires either tinker_model_path or tinker_base_model.")
        # If no HF tokenizer was loaded (tinker with no tito_tokenizer_name),
        # use the sampling client's own tokenizer — guaranteed to match the model.
        if self._tito_tokenizer is None:
            self._tito_tokenizer = self._tinker_sampling_client.get_tokenizer()
        self.logger.info(
            f"[tinker] sampling client ready "
            f"(base_model={self._tinker_base_model}, model_path={self._tinker_model_path})"
        )

    def _build_slides_fix_cmd(self) -> str:
        """Shell snippet that ships + runs the PowerPoint output-validation fix.

        Reads the sibling ``slides_output_validation_fix.py`` module, base64-encodes
        it, and emits a ``base64 -d > /tmp/slides_fix.py && python3 …`` command to be
        prepended to the start.sh launch. The script gates on the broken code path,
        so it is a no-op on images that don't have the bug (older dev-*/eval-*).
        Returns ``""`` (skip) if the script can't be read for any reason.
        """
        try:
            script = (Path(__file__).parent / "slides_output_validation_fix.py").read_text()
        except OSError as e:
            self.logger.warning(f"Could not read slides_output_validation_fix.py: {e}")
            return ""
        b64 = base64.b64encode(script.encode()).decode()
        return (
            f"printf %s '{b64}' | base64 -d > /tmp/slides_fix.py && "
            f"python3 /tmp/slides_fix.py /app/tools/mcp_servers "
            f"> /tmp/slides_fix.log 2>&1; "
        )

    async def _preinstall_python_packages(self, environment: BaseEnvironment) -> None:
        """Root-install `self._preinstall_packages` into the sandbox's system
        Python so the `code_execution` tool can import them.

        Sometimes the images don't ship pandas/openpyxl/etc. in the code sandbox, and the agent
        can't self-install (its pip user-base is the locked /.apps_data). We install as root
        (environment.exec runs as root) into /usr/local/lib site-packages, which the code tool's
        python3 sees. Idempotent (~10s cold, ~2s "already satisfied") and best-effort — a failure
        here must not abort the trial.
        """
        if not self._preinstall_packages:
            return
        pkgs = " ".join(shlex.quote(p) for p in self._preinstall_packages)
        cmd = f"python3 -m pip install --no-cache-dir {pkgs}"
        try:
            t0 = time.time()
            result = await environment.exec(cmd, timeout_sec=300)
            elapsed = time.time() - t0
            rc = getattr(result, "return_code", None)
            if rc == 0:
                self.logger.info(
                    f"[preinstall] installed {len(self._preinstall_packages)} " f"packages in {elapsed:.0f}s"
                )
            else:
                # Non-zero (e.g. no network / no pip): log tail, keep going.
                tail = (getattr(result, "stderr", "") or getattr(result, "stdout", "") or "")[-300:]
                self.logger.warning(f"[preinstall] pip returned rc={rc} in {elapsed:.0f}s (non-fatal): {tail}")
        except Exception as e:
            self.logger.warning(f"[preinstall] skipped (non-fatal): {e!r}")

    ####################################
    # run() and its helper methods
    ####################################

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        """
        Run the MCP tool-calling loop:
        1. Connect to MCP gateway and load tools
        2. Send system prompt + user instruction to LLM
        3. Loop: parse tool_calls -> execute via MCP -> append results
        4. Stop when LLM returns no tool_calls or max_steps reached

        In TITO mode, maintains a strictly-appending token list alongside
        messages, calls /v1/completions instead of /chat/completions, and
        records per-step transitions via TITOAgentState.
        """
        if self._use_hints:
            instruction = self._maybe_add_hints(instruction)
        if self._extra_prompt:
            instruction = f"{instruction}\n\n{self._extra_prompt}"

        mcp_client, connected_client, tools = await self._setup_mcp_client_and_tools()
        session = connected_client.session

        # Build initial messages
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": instruction},
        ]

        # TITO state
        tito: TITOAgentState | None = None
        if self._use_tito:
            from apex_agents_skyrl_recipe.agents.tito import (
                TITOAgentState,
                tokenize_initial_prompt,
            )

            prompt_ids = tokenize_initial_prompt(
                messages,
                tools,
                self._tito_tokenizer,
            )
            tito = TITOAgentState(prompt_ids, self._tito_tokenizer, tools)
            self.logger.info(f"TITO: initial prompt tokenized, {len(prompt_ids)} tokens")

        trajectory_path = self.logs_dir / "trajectory.json"

        start_time = time.time()
        total_input_tokens = 0
        total_output_tokens = 0
        total_cost = 0.0
        finalized = False
        step = -1
        raised_error: Exception | None = None
        step_timings: list[dict[str, Any]] = []
        repair_count = 0  # unparsed-<tool_call> corrections injected so far
        # Surfaced to RL via metadata: "stop" (clean finish), "length" (turn
        # truncated at max_tokens with no tool call), or None (max_steps/error).
        agent_stop_reason: str | None = None

        try:
            for step in range(self._max_steps):
                self.logger.info(f"Step {step + 1}/{self._max_steps}")
                _llm_t0 = time.perf_counter()

                # ── LLM generation ──────────────────────────────────
                try:
                    if self._use_tito:
                        remaining = self._max_context_len - len(tito.tokens)
                        if remaining <= 0:
                            self.logger.error(
                                f"TITO: context limit reached: {len(tito.tokens)}/{self._max_context_len} tokens used, "
                                f"no room for generation."
                            )
                            raise ContextLengthExceededError
                        tito_result = await self._generate_response_tito(
                            model=self.model_name,
                            messages_tokens=tito.tokens,
                            max_context_len=self._max_context_len,
                            tools=tools if tools else None,
                            api_base=self._api_base,
                            api_key=self._api_key,
                            llm_kwargs=self._llm_kwargs,
                            tokenizer=self._tito_tokenizer,
                            tool_call_parser=self._tito_tool_call_parser,
                            reasoning_parser=self._tito_reasoning_parser,
                            backend=self._tito_backend,
                            tinker_sampling_client=self._tinker_sampling_client,
                            logger=self.logger,
                        )
                        response = tito_result.model_response
                    else:
                        response = await self._generate_response(
                            model=self.model_name,
                            messages=messages,
                            tools=tools if tools else None,
                            api_base=self._api_base,
                            llm_kwargs=self._llm_kwargs,
                        )
                except Exception as e:
                    self.logger.error(f"LLM call failed at step {step + 1}: {e}")
                    raised_error = e  # e.g. ContextLengthExceededError
                    break

                _llm_time = time.perf_counter() - _llm_t0
                _num_output_tokens: int | None = None
                _tokens_per_sec: float | None = None
                if self._use_tito and tito_result is not None:
                    _num_output_tokens = len(tito_result.output_token_ids)
                    _tokens_per_sec = _num_output_tokens / _llm_time if _llm_time > 0 else None
                if _tokens_per_sec is not None:
                    self.logger.info(
                        f"Step {step + 1} LLM time: {_llm_time:.2f}s "
                        f"({_num_output_tokens} out tok, {_tokens_per_sec:.1f} tok/s)"
                    )
                else:
                    self.logger.info(f"Step {step + 1} LLM time: {_llm_time:.2f}s")

                # Extract usage
                usage = getattr(response, "usage", None)
                if usage:
                    total_input_tokens += getattr(usage, "prompt_tokens", 0) or 0
                    total_output_tokens += getattr(usage, "completion_tokens", 0) or 0
                    try:
                        total_cost += litellm.completion_cost(completion_response=response)
                    except Exception:
                        pass

                # Update context incrementally
                context.n_input_tokens = total_input_tokens
                context.n_output_tokens = total_output_tokens
                context.cost_usd = total_cost

                # Parse response
                choices = response.choices
                if not choices:
                    self.logger.warning("LLM returned empty choices")
                    break

                choice = choices[0]
                message = choice.message
                tool_calls = getattr(message, "tool_calls", None)

                # Extract fields from response
                reasoning = getattr(message, "reasoning_content", None)
                thinking_blocks = getattr(message, "thinking_blocks", None)
                content = getattr(message, "content", None) or ""

                if reasoning:
                    self.logger.debug(f"Reasoning: {reasoning[:500]}")
                if content and tool_calls:
                    self.logger.info(f"Response: {content[:300]}")

                # Append assistant message with all available fields
                # (matching loop_agent: include reasoning_content, thinking_blocks)
                assistant_msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": content,
                }
                if reasoning and self._store_reasoning_in_messages:
                    assistant_msg["reasoning_content"] = reasoning
                if thinking_blocks and self._store_reasoning_in_messages:
                    assistant_msg["thinking_blocks"] = thinking_blocks
                if tool_calls:
                    assistant_msg["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in tool_calls
                    ]
                messages.append(assistant_msg)

                # vLLM returns finish_reason == "length" both when a turn hits
                # the per-turn max_tokens cap AND when generation runs into the
                # overall context window. Distinguish by whether generation
                # consumed all the remaining context budget:
                #   - num_gen >= remaining: ran out of context -> treat as
                #     ContextLengthExceededError (same as the top-of-loop guard).
                #   - num_gen <  remaining: hit the per-turn cap with budget
                #     left -> stop_reason="length".
                hit_length = self._use_tito and tito_result is not None and tito_result.finish_reason == "length"
                num_gen = len(tito_result.output_token_ids) if (self._use_tito and tito_result is not None) else 0
                hit_context_limit = hit_length and num_gen >= remaining
                hit_max_tokens = hit_length and not hit_context_limit

                if hit_context_limit:
                    # Generation filled the remaining context window. Record the
                    # truncated step so its tokens are kept, then surface it as a
                    # context-length error (handled like the top-of-loop guard).
                    if self._use_tito:
                        tito.record_step(tito_result, step, assistant_msg)
                    self.logger.error(
                        f"TITO: generation ran into the context window at step {step + 1}: "
                        f"{len(tito.tokens)}/{self._max_context_len} tokens used "
                        f"({num_gen} generated, {remaining} budget). Treating as context length exceeded."
                    )
                    raised_error = ContextLengthExceededError()
                    break

                if not tool_calls:
                    # Parser produced no tool_calls. If the model actually TRIED to
                    # call a tool (raw <tool_call> left in the content) but malformed
                    # it, don't silently end at reward 0 — inject a corrective
                    # observation and let it retry (bounded). role="tool" so it
                    # survives _normalize_tool_messages_for_tito (which drops non-tool
                    # roles); tool_call_id is synthetic (there was no parsed call).
                    _content_str = content if isinstance(content, str) else str(content)
                    if (
                        self._use_tito
                        and self._tito_repair_unparsed_tool_calls
                        and not hit_max_tokens
                        and "<tool_call>" in _content_str
                        and repair_count < self._tito_max_repair_attempts
                    ):
                        repair_count += 1
                        correction = (
                            "Your previous message contained a <tool_call> block that "
                            "could not be parsed as a valid tool call. Re-issue the tool "
                            "call using the exact required tool-call format. If instead "
                            "you intended to finish the task, reply with your final answer "
                            "and do NOT include any <tool_call> block."
                        )
                        repair_msgs = [
                            {
                                "role": "tool",
                                "tool_call_id": f"repair_{step}",
                                "content": correction,
                            }
                        ]
                        messages.extend(repair_msgs)
                        if self._use_tito:
                            tito.record_step(tito_result, step, assistant_msg, repair_msgs)
                        step_timings.append(
                            {
                                "step": step,
                                "llm_time_sec": _llm_time,
                                "num_output_tokens": _num_output_tokens,
                                "tokens_per_sec": _tokens_per_sec,
                                "tool_time_sec": 0.0,
                                "tool_calls": [],
                            }
                        )
                        self.logger.warning(
                            f"Step {step + 1}: unparsed <tool_call>; injected correction "
                            f"({repair_count}/{self._tito_max_repair_attempts})"
                        )
                        continue

                    # No tool calls -> agent is done. A length-truncated turn here
                    # is an incomplete trajectory: mark stop_reason="length"
                    if self._use_tito:
                        tito.record_step(tito_result, step, assistant_msg)
                        self.logger.info(f"TITO: final step, total={len(tito.tokens)} tokens")
                    finalized = True
                    agent_stop_reason = "length" if hit_max_tokens else "stop"
                    if hit_max_tokens:
                        self.logger.warning(
                            f"Step {step + 1} hit max_tokens with no tool call; "
                            f"ending trajectory with stop_reason=length"
                        )
                    step_timings.append(
                        {
                            "step": step,
                            "llm_time_sec": _llm_time,
                            "num_output_tokens": _num_output_tokens,
                            "tokens_per_sec": _tokens_per_sec,
                            "tool_time_sec": 0.0,
                            "tool_calls": [],
                        }
                    )
                    self.logger.info(f"Agent finished after {step + 1} steps " f"(stop_reason={agent_stop_reason})")
                    break

                # Execute each tool call via MCP (persistent session,
                # one reconnection attempt on fatal MCP errors)
                _tool_t0 = time.perf_counter()
                try:
                    (
                        tool_response_messages,
                        per_tool_timings,
                    ) = await self._execute_tool_calls(session, tool_calls)
                except (
                    McpError,
                    RuntimeError,
                    BrokenResourceError,
                    ClosedResourceError,
                ) as e:
                    if not is_fatal_mcp_error(e):
                        raise
                    self.logger.warning(f"MCP session died at step {step + 1}, " f"attempting reconnect: {e!r}")
                    await self._log_mcp_diagnostics()
                    await self._disconnect_mcp_gracefully(mcp_client)
                    try:
                        connected_client = await self._connect_mcp_with_retry(mcp_client)
                        session = connected_client.session
                        (
                            tool_response_messages,
                            per_tool_timings,
                        ) = await self._execute_tool_calls(session, tool_calls)
                    except Exception:
                        self.logger.error("MCP reconnection failed, ending run")
                        raise
                _tool_time = time.perf_counter() - _tool_t0
                self.logger.info(f"Step {step + 1} tool time: {_tool_time:.2f}s ({len(per_tool_timings)} calls)")
                messages.extend(tool_response_messages)
                step_timings.append(
                    {
                        "step": step,
                        "llm_time_sec": _llm_time,
                        "num_output_tokens": _num_output_tokens,
                        "tokens_per_sec": _tokens_per_sec,
                        "tool_time_sec": _tool_time,
                        "tool_calls": per_tool_timings,
                    }
                )

                # Inject a budget warning into this turn's observation once the
                # window projected for the NEXT turn drops to/below the ratio.
                # record_step (below) appends this turn's generation AND
                # observation to tito.tokens, both charged against that next
                # window, so `remaining` subtracts both from the current
                # (pre-generation) count. num_gen is exact; the observation isn't
                # tokenized until record_step, so estimate it from its text
                # (usually the dominant term — template framing is omitted, so the
                # estimate runs slightly optimistic).
                if self._use_tito and self._tito_budget_warning_ratio is not None:
                    assert self._tito_tokenizer is not None
                    obs_text = self._observation_text(tool_response_messages)
                    n_obs_est = len(self._tito_tokenizer.encode(obs_text, add_special_tokens=False)) if obs_text else 0
                    remaining = self._max_context_len - len(tito.tokens) - num_gen - n_obs_est
                    threshold = self._tito_budget_warning_ratio * self._max_context_len
                    if remaining <= threshold:
                        pct_left = max(0, round(remaining / self._max_context_len * 100))
                        template = self._tito_budget_warning_text or DEFAULT_BUDGET_WARNING
                        warning = template.replace("{pct}", str(pct_left))
                        if self._append_budget_warning(tool_response_messages, warning):
                            self.logger.warning(
                                f"Injected budget warning at step {step + 1} "
                                f"(~{pct_left}% / {remaining} tokens left)"
                            )

                # ── TITO: record generation + observation ─────────
                if self._use_tito:
                    tito.record_step(tito_result, step, assistant_msg, tool_response_messages)
                    self.logger.info(f"TITO: step {step}, total={len(tito.tokens)} tokens")

                # Save trajectory incrementally
                save_trajectory_to_disk(trajectory_path, messages, start_time, finalized=False)

        finally:
            # Gracefully disconnect the persistent MCP session
            await self._disconnect_mcp_gracefully(mcp_client)

            status = "completed" if finalized else "failed"
            save_trajectory_to_disk(trajectory_path, messages, start_time, finalized, status)

            try:
                await environment.upload_file(str(trajectory_path), "/logs/agent/trajectory.json")
            except Exception as e:
                self.logger.warning(f"Failed to upload trajectory to sandbox: {e}")

            # Build context metadata
            metadata: dict[str, Any] = {
                "all_messages": messages,
                "n_episodes": step + 1,
                "summarization_count": 0,
                "status": status,
                "step_timings": step_timings,
                # "length" (max_tokens truncation), "stop" (clean finish), or
                # None (max_steps/error).
                "stop_reason": agent_stop_reason,
            }

            # In TITO mode: store the three parallel lists in metadata,
            # dump transitions to disk for debugging
            if self._use_tito and tito is not None:
                tito.check_invariants()
                metadata["tito_tokens"] = tito.tokens
                metadata["tito_loss_mask"] = tito.loss_mask
                metadata["tito_logprobs"] = tito.logprobs

                if self._tito_dump_transitions:
                    tito.save_transitions(self.logs_dir / "tito_transitions.json")
                tito.save_debug(self.logs_dir / "tito_debug.txt")

            context.metadata = metadata

            self.logger.info(
                f"Agent {status}: {total_input_tokens} input tokens, "
                f"{total_output_tokens} output tokens, ${total_cost:.4f}"
            )

            # Log failure diagnostics when the agent did not finalize
            if not finalized:
                self._log_failure_diagnostics(
                    step=step,
                    max_steps=self._max_steps,
                    raised_error=raised_error,
                )

        # Re-raise after finalization so trial.py records ExceptionInfo.
        if raised_error is not None:
            raise raised_error

    @staticmethod
    def _observation_text(messages: list[dict[str, Any]]) -> str:
        """Concatenate the text content of tool/observation messages.

        A cheap proxy for an observation's token count before ``record_step``
        tokenizes it. Handles string and list-of-blocks content; non-text blocks
        (e.g. images) and template framing are ignored, so it slightly
        undercounts.
        """
        parts: list[str] = []
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        parts.append(block.get("text", ""))
        return "\n".join(parts)

    @staticmethod
    def _append_budget_warning(
        tool_response_messages: list[dict[str, Any]],
        warning: str,
    ) -> bool:
        """Append ``warning`` to the most recent tool message's content.

        Mutates the last ``role == "tool"`` message in place (the dicts are
        shared with the running ``messages`` list, so the warning is also
        recorded in the trajectory). Handles both string content and the
        Anthropic list-of-blocks content shape. Returns ``True`` if a tool
        message was found and updated, ``False`` otherwise (e.g. the step
        produced only image-workaround user messages, in which case the caller
        leaves the warning pending for the next step).
        """
        for msg in reversed(tool_response_messages):
            if msg.get("role") != "tool":
                continue
            content = msg.get("content")
            if isinstance(content, list):
                content.append({"type": "text", "text": warning})
            else:
                msg["content"] = f"{content}\n\n{warning}" if content else warning
            return True
        return False

    def _log_failure_diagnostics(
        self,
        *,
        step: int,
        max_steps: int,
        raised_error: Exception | None,
    ) -> None:
        """Log a human-readable summary of why the agent failed."""
        if step + 1 >= max_steps:
            reason = f"max turns reached ({max_steps})"
        elif raised_error is not None:
            reason = f"exception at step {step + 1}: {type(raised_error).__name__}: {raised_error}"
        else:
            reason = f"exited at step {step + 1}/{max_steps}"

        self.logger.warning(f"Failure reason: {reason}")

    def _build_mcp_config(self) -> dict[str, Any]:
        """Build the MCP client configuration dict."""
        if not self._mcp_gateway_url:
            raise RuntimeError("MCP gateway URL not set — was setup() called?")
        return {
            "mcpServers": {
                "gateway": {
                    "transport": "streamable-http",
                    "url": self._mcp_gateway_url,
                }
            }
        }

    async def _log_mcp_diagnostics(self) -> None:
        """Log sandbox diagnostics to help debug MCP connection failures."""
        if not self._environment:
            return
        try:
            result = await self._environment.exec(
                "echo '=== gateway process ===' && "
                "ps aux | grep 'runner.main\\|uvicorn' | grep -v grep && "
                "echo '=== port 8000 ===' && "
                "(ss -tlnp 2>/dev/null || netstat -tlnp 2>/dev/null) | grep 8000 && "
                "echo '=== health check ===' && "
                "curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/health --max-time 3 && echo && "
                "echo '=== resources ===' && "
                "CPU_NS=$(cat /sys/fs/cgroup/cpuacct/cpuacct.usage 2>/dev/null || echo 0) && "
                'echo "cpu: $((CPU_NS / 1000000))ms used, nproc=$(nproc 2>/dev/null || echo n/a)" && '
                'echo "memory: $(( $(cat /sys/fs/cgroup/memory/memory.usage_in_bytes 2>/dev/null || echo 0) / 1048576 )) MB" && '
                "echo '=== OOM kills ===' && "
                "dmesg 2>/dev/null | grep -i 'oom\\|killed process' | tail -5 || echo 'no OOM events' && "
                "echo '=== start.log tail ===' && "
                "tail -20 /tmp/start.log 2>/dev/null || echo 'no start.log'",
                timeout_sec=10,
            )
            self.logger.warning(f"MCP diagnostics:\n{(result.stdout or '').strip()}")
        except Exception as diag_err:
            self.logger.warning(f"Could not collect MCP diagnostics: {diag_err}")

    async def _connect_mcp_with_retry(
        self,
        mcp_client: FastMCPClient,
        max_retries: int = 3,
        base_delay: float = 5.0,
    ) -> FastMCPClient:
        """Connect the MCP client with exponential backoff retries.

        Returns the connected client (caller must disconnect via
        _disconnect_mcp_gracefully when done).
        """
        for attempt in range(1, max_retries + 1):
            try:
                t0 = time.time()
                connected = await mcp_client.__aenter__()
                elapsed = time.time() - t0
                self.logger.info(f"MCP session connected (attempt {attempt}/{max_retries}, " f"{elapsed:.1f}s)")
                return connected
            except (McpError, RuntimeError, OSError) as e:
                elapsed = time.time() - t0
                self.logger.warning(
                    f"MCP connect attempt {attempt}/{max_retries} failed " f"after {elapsed:.1f}s: {e!r}"
                )
                await self._log_mcp_diagnostics()
                if attempt == max_retries:
                    self.logger.error("MCP connection failed after all retries, giving up")
                    raise
                delay = base_delay * (2 ** (attempt - 1))
                self.logger.info(f"Retrying MCP connection in {delay:.0f}s...")
                await asyncio.sleep(delay)
        raise RuntimeError("Unreachable")  # satisfies type checker

    async def _disconnect_mcp_gracefully(
        self,
        mcp_client: FastMCPClient,
        timeout: float = 10.0,
    ) -> None:
        """Best-effort MCP disconnect with a timeout. Never raises."""
        try:
            await asyncio.wait_for(
                mcp_client.__aexit__(None, None, None),
                timeout=timeout,
            )
            self.logger.info("MCP session disconnected cleanly")
        except TimeoutError:
            self.logger.warning(f"MCP disconnect timed out after {timeout}s (ignoring)")
        except Exception as e:
            self.logger.warning(f"MCP disconnect error (ignoring): {e!r}")

    async def _setup_mcp_client_and_tools(
        self,
    ) -> tuple[FastMCPClient, FastMCPClient, list[dict[str, Any]]]:
        """
        Build, connect, and load tools from the MCP gateway.

        Returns (mcp_client, connected_client, tools). The caller owns the
        session and must call _disconnect_mcp_gracefully when done.
        """
        mcp_config = self._build_mcp_config()
        mcp_client = FastMCPClient(mcp_config)

        self.logger.info("Loading MCP tools...")
        connected_client = await self._connect_mcp_with_retry(mcp_client)
        tools = None
        load_tools_max_retries = 3
        TOOL_LOAD_TIMEOUT = 60.0
        for load_attempt in range(1, load_tools_max_retries + 1):
            try:
                tools = await asyncio.wait_for(
                    load_mcp_tools(connected_client.session, format="openai"),
                    timeout=TOOL_LOAD_TIMEOUT,
                )
                break
            except (asyncio.TimeoutError, Exception) as e:
                self.logger.warning(f"load_mcp_tools attempt {load_attempt}/{load_tools_max_retries} " f"failed: {e!r}")
                if load_attempt == load_tools_max_retries:
                    self.logger.error("Failed to load MCP tools after all retries")
                    await self._log_mcp_diagnostics()
                    await self._disconnect_mcp_gracefully(mcp_client)
                    raise
                # Reconnect MCP before retrying
                self.logger.info("Reconnecting MCP before retrying tool load...")
                await self._disconnect_mcp_gracefully(mcp_client)
                mcp_client = FastMCPClient(self._build_mcp_config())
                connected_client = await self._connect_mcp_with_retry(mcp_client)
        tool_names = [t.get("function", {}).get("name", "?") for t in tools]
        self.logger.info(f"Loaded {len(tools)} tools: {', '.join(tool_names)}")

        # Log expected vs actually-loaded servers so we can spot silent
        # degradation.  Currently if a server doesn't come up in time its
        # tools are simply absent from the list and the trial continues —
        # there is no hard gate that waits for every expected server.
        # TODO: consider adding a readiness gate (retry tool loading or
        # wait for a start.log sentinel) to avoid silent partial-tool runs.
        if self._expected_mcp_servers:
            loaded_servers = [
                s for s in self._expected_mcp_servers if any(name.startswith(f"{s}_") for name in tool_names)
            ]
            missing_servers = [s for s in self._expected_mcp_servers if s not in loaded_servers]
            self.logger.info(
                f"Expected {len(self._expected_mcp_servers)} MCP servers "
                f"{self._expected_mcp_servers}, "
                f"loaded {len(loaded_servers)}: {loaded_servers}"
            )
            if missing_servers:
                self.logger.warning(f"MCP servers not represented in loaded tools: {missing_servers}")

        # Flatten the single input/request envelope so the model sees a flat tool surface
        # (== eval images, ~23k tokens smaller) and qwen3_xml types the now-top-level args natively.
        # Re-wrapped on dispatch in _execute_tool_calls via self._wrapper_map.
        tools, self._wrapper_map = flatten_envelope_tools(tools)
        if self._wrapper_map:
            self.logger.info(
                f"Flattened {len(self._wrapper_map)} envelope tools "
                f"(wrap-on-dispatch): {sorted(self._wrapper_map)[:6]}..."
            )

        return mcp_client, connected_client, tools

    async def _execute_tool_calls(self, session, tool_calls) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """
        Execute tool calls via MCP using an existing session, and return tool
        messages.  Four levels of error handling.

        Raises McpError or RuntimeError when ``is_fatal_mcp_error(e)`` is True
        so that the caller (``run()``) can attempt a reconnection.

        Returns (messages, per_tool_timings) where per_tool_timings is a list
        of dicts: {"name": str, "duration_sec": float, "outcome": str}.
        """
        messages = []
        per_tool_timings: list[dict[str, Any]] = []

        for tc in tool_calls:
            tool_name = tc.function.name or "unknown"
            self.logger.info(f"Tool call: {tool_name}({tc.function.arguments[:200]})")

            # Dispatch: the catalog was flattened, so the model called this tool with flat args — re-wrap
            # under input/request before handing it to the MCP server (== call_args = {route.wrapper: args}).
            wrapper = self._wrapper_map.get(tool_name)
            if wrapper:
                tc.function.arguments = rewrap_envelope_arguments(tc.function.arguments, wrapper)

            _tool_t0 = time.perf_counter()
            outcome = "ok"
            try:
                call_result = await asyncio.wait_for(
                    call_openai_tool(session, tc),
                    timeout=self._tool_call_timeout,
                )
            except TimeoutError:
                # Tool call error 1: Tool timed out. Report to LLM and continue.
                error_msg = f"Tool {tool_name} timed out after {self._tool_call_timeout}s"
                self.logger.error(error_msg)
                messages.append(get_error_tool_message(tc.id, tool_name, error_msg))
                per_tool_timings.append(
                    {
                        "name": tool_name,
                        "duration_sec": time.perf_counter() - _tool_t0,
                        "outcome": "timeout",
                    }
                )
                continue
            except Exception as e:
                # Tool call error 2: Fatal MCP errors. End the run by raising to Trial level.
                if is_fatal_mcp_error(e):
                    error_msg = f"Fatal MCP error during tool {tool_name}: {e!r}"
                    self.logger.error(error_msg)
                    messages.append(get_error_tool_message(tc.id, tool_name, error_msg))
                    per_tool_timings.append(
                        {
                            "name": tool_name,
                            "duration_sec": time.perf_counter() - _tool_t0,
                            "outcome": "fatal_mcp_error",
                        }
                    )
                    raise

                # Tool call error 3: Non-fatal MCP errors. Report to LLM and continue.
                error_msg = f"Error calling tool {tool_name}: {e!r}"
                self.logger.error(error_msg)
                messages.append(get_error_tool_message(tc.id, tool_name, error_msg))
                per_tool_timings.append(
                    {
                        "name": tool_name,
                        "duration_sec": time.perf_counter() - _tool_t0,
                        "outcome": "mcp_error",
                    }
                )
                continue

            # Tool call error 4: No content returned. Report to LLM and continue.
            if not call_result.content:
                error_msg = f"Call result is not valid, received {call_result.content}"
                self.logger.error(error_msg)
                messages.append(get_error_tool_message(tc.id, tool_name, error_msg))
                per_tool_timings.append(
                    {
                        "name": tool_name,
                        "duration_sec": time.perf_counter() - _tool_t0,
                        "outcome": "empty_content",
                    }
                )
                continue

            # Tool call success: Convert MCP content blocks to messages
            tool_msgs = content_blocks_to_messages(
                call_result.content,
                tc.id,
                tool_name,
                self.model_name,
                self.logger,
            )

            # Truncate tool result messages if needed
            if self._tool_result_max_chars is not None:
                head = self._tool_result_max_chars * 2 // 3
                tail = self._tool_result_max_chars - head
                truncate_tool_result_messages(
                    tool_msgs,
                    self.logger,
                    head_chars=head,
                    tail_chars=tail,
                )

            messages.extend(tool_msgs)
            per_tool_timings.append(
                {
                    "name": tool_name,
                    "duration_sec": time.perf_counter() - _tool_t0,
                    "outcome": outcome,
                }
            )

            # Log the result text
            result_text = ""
            for block in call_result.content:
                if isinstance(block, TextContent):
                    result_text += block.text[:200]
            self.logger.info(f"Tool {tool_name} result: {result_text[:300]}")
        return messages, per_tool_timings

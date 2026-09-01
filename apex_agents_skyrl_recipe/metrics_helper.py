import re

# Tool-result failure signatures. A tool RESULT is a FAILURE iff its text matches one of these
# (agent dispatch errors + MCP validation/param/output errors). Erroring *code* run by code_execution
# (success:false / "Command execution timed out") and valid-but-empty results
# ("No files found") count as SUCCESS — the tool itself ran.
_TOOL_ERR = re.compile(
    "|".join(
        [
            r"Tool [\w.\-]+ timed out after \d+\s*s",  # agent tool-call timeout (NOT code "Command execution timed out")
            r"Error calling tool ",
            r"Fatal MCP error during tool",
            r"Call result is not valid, received",
            r"Output validation error",
            r"Internal error: Validation error",
            r"Internal error: \d+ validation error",
            r"validation error[s]? for call",
            r"Invalid params",
            r"Invalid operations payload",
            r"Missing required argument",
            r"Unexpected keyword argument",
            r"Input should be a valid",
            r"is not of type",
        ]
    )
)

# Known MCP servers in the Archipelago tool set. Flattened tool names are
# ``{server}_{tool}``, but server names can themselves contain underscores
# (``code_execution``), so we bucket by longest-prefix match against this list
# rather than splitting on ``_``. The standard set is the first 9; the last two
# are legacy/rare (world-223 stirrup image). Tools that match none fall into
# ``"other"`` — add new servers here to give them their own metric.
_MCP_SERVERS = (
    "calendar",
    "chat",
    "code_execution",
    "excel",
    "filesystem",
    "mail",
    "pdfs",
    "powerpoint",
    "word",
    "stirrup_code_execution",
    "web_search",
)
_MCP_SERVERS_BY_LEN = sorted(_MCP_SERVERS, key=len, reverse=True)
# All buckets that may appear as a metric key, so per-server metrics are emitted
# for every server every batch (0.0 when unused) — stable keys for dashboards.
_MCP_SERVER_BUCKETS = _MCP_SERVERS + ("other",)


# Raw, additive tool-call metric keys. We emit these per generate call so
# ``concatenate_generator_outputs`` (which sums unknown numeric metric keys) folds
# them correctly across calls; the micro-averaged rates are then re-derived from the
# summed counts by ``finalize_tool_call_metrics``. Emitting the ratios directly would
# break under the fully-async trainer, which issues one generate call per trajectory —
# summing per-call ratios pushes them above 1.
_KEY_TOOL_CALLS_TOTAL = "generate/tool_calls_total"
_KEY_SUCCESSFUL_TOOL_CALLS_TOTAL = "generate/successful_tool_calls_total"
_KEY_CODE_EXEC_TOOL_CALLS_TOTAL = "generate/code_exec_tool_calls_total"
_KEY_NUM_TOOL_TRAJECTORIES = "generate/num_tool_trajectories"
_KEY_TOOL_CALLS_BY_SERVER_TOTAL_PREFIX = "generate/tool_calls_by_server_total"


def finalize_tool_call_metrics(
    rollout_metrics: dict,
    total_tool_calls: int,
    total_successful_tool_calls: int,
    total_code_exec_tool_calls: int,
    num_tool_trajectories: int,
    server_totals: dict,
) -> dict:
    """Write raw tool-call counts and their micro-averaged rates in place.

    Raw counts are additive, so ``concatenate_generator_outputs`` (which sums
    numeric metric keys) folds them correctly across generate calls. Rates are
    floored on a denominator of 1 so every key — including all per-server
    buckets — is emitted even with zero tool calls, keeping dashboard keys
    stable.
    """
    rollout_metrics[_KEY_TOOL_CALLS_TOTAL] = total_tool_calls
    rollout_metrics[_KEY_SUCCESSFUL_TOOL_CALLS_TOTAL] = total_successful_tool_calls
    rollout_metrics[_KEY_CODE_EXEC_TOOL_CALLS_TOTAL] = total_code_exec_tool_calls
    rollout_metrics[_KEY_NUM_TOOL_TRAJECTORIES] = num_tool_trajectories
    for server in _MCP_SERVER_BUCKETS:
        rollout_metrics[f"{_KEY_TOOL_CALLS_BY_SERVER_TOTAL_PREFIX}/{server}"] = server_totals.get(server, 0)

    total = total_tool_calls
    denom = max(1, total)
    ntraj = max(1, num_tool_trajectories)
    rollout_metrics["generate/avg_tool_calls_per_trajectory"] = total / ntraj
    rollout_metrics["generate/tool_call_success_rate"] = (
        rollout_metrics.get(_KEY_SUCCESSFUL_TOOL_CALLS_TOTAL, 0) / denom
    )
    rollout_metrics["generate/code_exec_tool_call_frac"] = (
        rollout_metrics.get(_KEY_CODE_EXEC_TOOL_CALLS_TOTAL, 0) / denom
    )
    for server in _MCP_SERVER_BUCKETS:
        rollout_metrics[f"generate/tool_call_frac/{server}"] = (
            rollout_metrics.get(f"{_KEY_TOOL_CALLS_BY_SERVER_TOTAL_PREFIX}/{server}", 0) / denom
        )
    return rollout_metrics


def _server_of(name: str) -> str:
    """Map a flattened tool name to its MCP server bucket (or ``"other"``)."""
    n = (name or "").lower()
    for s in _MCP_SERVERS_BY_LEN:
        if n == s or n.startswith(f"{s}_"):
            return s
    return "other"


def _tool_result_text(content) -> str:
    """Flatten a tool message's content to text (handles str and list-of-blocks)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join((b.get("text") or b.get("content") or "") if isinstance(b, dict) else str(b) for b in content)
    return str(content)


def _is_code_exec_tool(name: str) -> bool:
    """Whether a tool name is a code-execution tool in the Archipelago tool set.

    Covers the standard ``code_execution_code_exec`` (and any other
    ``code_execution_*``) plus the legacy world-223 ``stirrup_code_execution``
    server (whose shell tool is ``run_shell``).
    """
    n = (name or "").lower()
    return "code_execution" in n or n == "run_shell"


def _compute_tool_metrics(metadata: dict) -> tuple[int, int, int, dict]:
    """Count tool calls from an agent's messages.

    Returns ``(total, successful, code_exec, by_server)`` where ``by_server`` is
    a ``{server: count}`` dict over ``_MCP_SERVER_BUCKETS``. One
    ``role == "tool"`` message corresponds to one tool call
    (``content_blocks_to_messages`` emits a single tool message per call; image
    workarounds become separate ``user`` messages, not tool messages). Success
    uses the text-regex definition shared with the offline scorer.
    """
    total = successful = code_exec = 0
    by_server: dict = {s: 0 for s in _MCP_SERVER_BUCKETS}
    for msg in metadata.get("all_messages", []) or []:
        if msg.get("role") != "tool":
            continue
        total += 1
        name = msg.get("name", "")
        text = _tool_result_text(msg.get("content")).lstrip()[:150]
        if not _TOOL_ERR.search(text):
            successful += 1
        if _is_code_exec_tool(name):
            code_exec += 1
        by_server[_server_of(name)] += 1
    return total, successful, code_exec, by_server

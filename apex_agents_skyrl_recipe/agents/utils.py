"""Helper methods for archipelago agent.."""

import copy
import json
import logging
import time
from pathlib import Path
from typing import Any

from mcp import McpError
from mcp.types import ContentBlock, ImageContent, TextContent


def get_error_tool_message(
    tool_call_id: str,
    name: str,
    error_content: str,
) -> dict[str, str]:
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "name": name,
        "content": error_content,
    }


def save_trajectory_to_disk(
    path: Path,
    messages: list[dict],
    start_time: float,
    finalized: bool,
    status: str = "running",
) -> None:
    """Save trajectory to disk."""

    def _serialize_messages(messages: list[dict]) -> list[dict]:
        """Make messages JSON-serializable."""
        result = []
        for msg in messages:
            serialized = {}
            for k, v in msg.items():
                try:
                    json.dumps(v)
                    serialized[k] = v
                except (TypeError, ValueError):
                    serialized[k] = str(v)
            result.append(serialized)
        return result

    trajectory = {
        "messages": _serialize_messages(messages),
        "status": status,
        "time_elapsed": time.time() - start_time,
        "finalized": finalized,
    }
    path.write_text(json.dumps(trajectory, indent=2, default=str))


#####################################
# Methods below are copied from https://github.com/Mercor-Intelligence/archipelago
#####################################


def content_blocks_to_messages(
    content_blocks: list[ContentBlock],
    tool_call_id: str,
    name: str,
    model: str,
    logger: logging.Logger,
) -> list[dict[str, Any]]:
    """
    Exactly copied from https://github.com/Mercor-Intelligence/archipelago/blob/main/agents/runner/utils/mcp.py.

    Convert MCP content blocks to a single LiteLLM tool message.

    Each tool_use must have exactly one tool_result. This function combines all
    content blocks into a single tool message to satisfy API requirements for
    Anthropic, OpenAI, and other providers.

    Args:
        content_blocks: MCP content blocks from tool result
        tool_call_id: The tool call ID to associate with the result
        name: The tool name
        model: The model being used

    Returns:
        List of messages: always exactly one tool message, plus optional user
        messages for images on non-Anthropic providers.
    """
    # Anthropic supports images directly in tool results
    supports_image_tool_results = "anthropic" in model.lower()

    text_contents: list[str] = []
    image_data_uris: list[str] = []

    for content_block in content_blocks:
        match content_block:
            case TextContent():
                block = TextContent.model_validate(content_block)
                text_contents.append(block.text)

            case ImageContent():
                block = ImageContent.model_validate(content_block)
                data_uri = f"data:{block.mimeType};base64,{block.data}"
                image_data_uris.append(data_uri)

            case _:
                logger.warning(f"Content block type {content_block.type} not supported")
                text_contents.append("Unable to parse tool call response")

    messages: list[dict[str, Any]] = []

    if supports_image_tool_results:
        content: list[dict[str, Any]] = []
        for text in text_contents:
            content.append({"type": "text", "text": text})
        for data_uri in image_data_uris:
            content.append({"type": "image_url", "image_url": {"url": data_uri}})

        tool_message: dict[str, Any] = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": name,
            "content": content if content else [{"type": "text", "text": ""}],
        }
        messages.append(tool_message)
    else:
        content = [{"type": "text", "text": text} for text in text_contents]

        if image_data_uris and not content:
            content.append({"type": "text", "text": f"Image(s) returned by {name} tool"})

        tool_message = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": name,
            "content": content if content else [{"type": "text", "text": ""}],
        }
        messages.append(tool_message)

        # Add image workaround: user messages with images
        for data_uri in image_data_uris:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                }
            )

    return messages


def _normalize_tool_messages_for_tito(
    tool_messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Normalize tool response messages for tokenization via chat template.

    The GLM-4.7 chat template expects tool message content to be a string.
    MCP tool results may have list-of-dicts content blocks; flatten them.
    Also drops image-workaround user messages (images can't be tokenized).
    """
    normalized = []
    for msg in tool_messages:
        if msg.get("role") != "tool":
            # Skip image-workaround user messages
            continue
        msg = dict(msg)
        content = msg.get("content", "")
        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    text_parts.append(block)
                else:
                    text_parts.append(str(block))
            msg["content"] = "\n".join(text_parts) if text_parts else ""
        normalized.append(msg)
    return normalized


def is_fatal_mcp_error(exception: Exception) -> bool:
    """Determine if an exception is fatal and should immediately end the agent run.

    Based on https://github.com/Mercor-Intelligence/archipelago/blob/main/agents/runner/utils/error.py
    with additional patterns discovered from production trial analysis.

    Fatal errors indicate the MCP session/connection is dead and cannot recover.
    Non-fatal errors can be reported to the LLM and the agent can continue.

    Args:
        exception: The exception to check.

    Returns:
        True if the error is fatal (session terminated, connection dead),
        False if the error is recoverable.
    """
    # Check for MCP-specific errors
    if isinstance(exception, McpError):
        # Check error code - handle both positive 32600 (current MCP bug) and
        # negative -32600 (JSON-RPC 2.0 standard) for forward compatibility
        error_code = getattr(exception.error, "code", None) if hasattr(exception, "error") else None
        if error_code in (32600, -32600):
            return True

        error_str = str(exception)
        # Fallback to string matching for robustness
        if "Session terminated" in error_str:
            return True

        # MCP gateway connection timeout — the server is unreachable.
        # This is raised by fastmcp when httpx.ConnectTimeout occurs.
        if "Timed out while waiting for response" in error_str:
            return True

    # Check for FastMCP client disconnection errors
    if isinstance(exception, RuntimeError):
        error_str = str(exception)
        # FastMCP raises this when the client session has been closed/corrupted
        if "Client is not connected" in error_str:
            return True
        # httpx transport failures — server disconnected or refused
        if "Client failed to connect" in error_str:
            return True
        if "Server disconnected" in error_str:
            return True

    # anyio stream broken — the underlying transport is dead
    try:
        from anyio import BrokenResourceError, ClosedResourceError

        if isinstance(exception, (BrokenResourceError, ClosedResourceError)):
            return True
    except ImportError:
        pass

    return False


# --------------------------------------------------------------
# Tool call format quirks of Archipelago MCP servers
# --------------------------------------------------------------


def _schema_expects_container(schema: Any) -> bool:
    """True if a JSON-schema fragment expects an array or object value.

    Covers plain ``{"type": "array"|"object"}``, union types, ``$ref`` to a
    model, presence of ``items``/``properties``, and ``anyOf``/``allOf``/
    ``oneOf`` combinators (used by pydantic models exposed over MCP, e.g. a
    single ``request`` parameter typed as a model).
    """
    if not isinstance(schema, dict):
        return False
    t = schema.get("type")
    if t in ("array", "object"):
        return True
    if isinstance(t, list) and ("array" in t or "object" in t):
        return True
    if "$ref" in schema or "items" in schema or "properties" in schema:
        return True
    for key in ("anyOf", "allOf", "oneOf"):
        for sub in schema.get(key, []) or []:
            if _schema_expects_container(sub):
                return True
    return False


def _resolve_schema_ref(schema: Any, root: dict[str, Any]) -> Any:
    """Resolve a local ``{"$ref": "#/$defs/Foo"}`` against the schema ``root``.

    Follows local refs (``#/...``, e.g. into ``$defs``/``definitions``) so a
    pydantic model exposed as a single envelope param can be inspected. Returns
    the referenced subschema, or ``schema`` unchanged for a non-local /
    unresolvable / cyclic ref.
    """
    seen: set[str] = set()
    while isinstance(schema, dict) and "$ref" in schema:
        ref = schema["$ref"]
        if not isinstance(ref, str) or not ref.startswith("#/") or ref in seen:
            return schema
        seen.add(ref)
        node: Any = root
        for part in ref[2:].split("/"):
            part = part.replace("~1", "/").replace("~0", "~")
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return schema
        schema = node
    return schema


def coerce_stringified_container_args(name: str, arguments_json: str, tools: list[dict[str, Any]] | None) -> str:
    """Coerce JSON-stringified top-level array/object args back to containers.

    Belt-and-suspenders for vLLM's XML tool parsers: ``qwen3_xml`` types each
    ``<parameter=...>`` value from the tool schema, but a container arg can still
    arrive JSON-stringified (``content='[{...}]'`` instead of ``content=[{...}]``)
    in edge cases. Re-parse any top-level string arg whose declared schema type is
    a container and whose string parses as a JSON list/dict.

    With :func:`flatten_envelope_tools` advertising flat schemas, the model's
    args are already top-level and the parser types them natively, so this is a
    cheap no-op safety net. Scalar args and already-correct containers are
    untouched, so it is safe for every parser.
    """
    if not tools:
        return arguments_json
    props: dict[str, Any] | None = None
    for tool in tools:
        fn = tool.get("function", tool) if isinstance(tool, dict) else {}
        if fn.get("name") == name:
            props = (fn.get("parameters") or {}).get("properties") or {}
            break
    if not props:
        return arguments_json
    try:
        args = json.loads(arguments_json)
    except (json.JSONDecodeError, TypeError):
        return arguments_json
    if not isinstance(args, dict):
        return arguments_json
    changed = False
    for key, val in list(args.items()):
        if not isinstance(val, str) or not _schema_expects_container(props.get(key)):
            continue
        try:
            parsed = json.loads(val)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, (list, dict)):
            args[key] = parsed
            changed = True
    return json.dumps(args) if changed else arguments_json


def _schema_is_object_model(schema: Any) -> bool:
    """True if a schema fragment denotes an object/pydantic-model (not scalar/array).

    Distinguishes an *envelope* parameter (a single ``request``/``input`` model)
    from a plain array param (which must be passed directly, never wrapped) and
    from a scalar param. Recurses through ``anyOf``/``allOf``/``oneOf``.
    """
    if not isinstance(schema, dict):
        return False
    t = schema.get("type")
    if t == "object" or "$ref" in schema or "properties" in schema:
        return True
    if isinstance(t, list) and "object" in t:
        return True
    for key in ("anyOf", "allOf", "oneOf"):
        for sub in schema.get(key, []) or []:
            if _schema_is_object_model(sub):
                return True
    return False


def _envelope_param_name(name: str, tools: list[dict[str, Any]] | None) -> str | None:
    """Return the sole envelope-param name of a tool, or ``None``.

    Some MCP servers expose a tool as a single object/model parameter — e.g.
    ``excel_read_tab(input: ReadTabRequest)`` or ``code_exec(request:
    CodeExecRequest)``. Such a tool's schema has exactly one (required) property
    whose type is an object/``$ref`` model. We return that property name (read
    from the schema, not hard-coded, since servers use both ``input`` and
    ``request``). Flat tools (multiple properties, or a single scalar/array
    property) return ``None``.
    """
    if not tools:
        return None
    for tool in tools:
        fn = tool.get("function", tool) if isinstance(tool, dict) else {}
        if fn.get("name") != name:
            continue
        params = fn.get("parameters") or {}
        props = params.get("properties") or {}
        if len(props) != 1:
            return None
        ((key, schema),) = props.items()
        required = params.get("required") or [key]
        if key in required and _schema_is_object_model(schema):
            return key
        return None
    return None


def _referenced_def_names(schema: Any) -> set[str]:
    """Collect local ``$ref`` target names (``#/$defs/X`` → ``X``) within ``schema``."""
    names: set[str] = set()
    stack = [schema]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/"):
                names.add(ref.rsplit("/", 1)[-1])
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    return names


def flatten_envelope_tools(
    tools: list[dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Hoist each single ``input``/``request`` envelope model to the top level of
    the *advertised* schema.

    The model then sees a **flat** tool surface — identical to the eval/flat
    images and ~23k tokens smaller than the nested envelope schema — and vLLM's
    ``qwen3_xml`` parser types the now-top-level fields natively (``pages`` →
    list, ``max_rows`` → int) via its own ``_convert_param_value``. Returns
    ``(flat_tools, wrapper_map)`` where ``wrapper_map[name]`` is the wrapper to
    re-apply on dispatch (``call_args = {wrapper: args}``).

    A no-op on already-flat tools (multi-prop, or single scalar/array param), so
    it is safe across the flat (old/eval) and envelope (dev-1928) images alike.
    This is the canonical fix: the model never emits a nested envelope, so no
    call-side wrapping/coercion shims are needed — just
    :func:`rewrap_envelope_arguments` on dispatch.
    """
    if not tools:
        return tools or [], {}
    flat: list[dict[str, Any]] = []
    wrapper_map: dict[str, str] = {}
    for tool in tools:
        fn = tool.get("function", tool) if isinstance(tool, dict) else {}
        name = fn.get("name")
        params = fn.get("parameters") or {}
        wrapper = _envelope_param_name(name, [tool]) if name else None
        if not wrapper:
            flat.append(tool)  # already flat — leave untouched
            continue
        inner = _resolve_schema_ref((params.get("properties") or {}).get(wrapper), params)
        if not isinstance(inner, dict) or "properties" not in inner:
            flat.append(tool)  # couldn't resolve the model → leave wrapped
            continue
        new_params = copy.deepcopy(inner)
        # Carry over only the $defs/definitions the inlined schema still references
        # (transitively). The wrapper model itself is now inlined at top level, so
        # keeping its $def would *duplicate* the whole schema — pruning to the
        # referenced subset keeps nested models resolvable without that bloat.
        for defs_key in ("$defs", "definitions"):
            src = params.get(defs_key)
            if not isinstance(src, dict):
                continue
            needed: set[str] = set()
            frontier = _referenced_def_names(new_params)
            while frontier:
                nm = frontier.pop()
                if nm in needed or nm not in src:
                    continue
                needed.add(nm)
                frontier |= _referenced_def_names(src[nm])
            if needed:
                new_params[defs_key] = {nm: copy.deepcopy(src[nm]) for nm in needed}
        new_fn = {**fn, "parameters": new_params}
        flat.append({**tool, "function": new_fn} if "function" in tool else new_fn)
        wrapper_map[name] = wrapper
    return flat, wrapper_map


def rewrap_envelope_arguments(arguments_json: str, wrapper: str | None) -> str:
    """Re-wrap flat tool args under ``wrapper`` for dispatch to the MCP server.

    The inverse of :func:`flatten_envelope_tools`: the model called the tool with
    flat args (because the catalog was flattened), so before dispatch we nest
    them back under ``input``/``request`` (== foundry's ``call_args = {route.
    wrapper: normalized_args}``). Idempotent: if the args are already the lone
    ``{wrapper: {...}}`` envelope, they are returned unchanged.
    """
    if not wrapper:
        return arguments_json
    try:
        args = json.loads(arguments_json or "{}")
    except (json.JSONDecodeError, TypeError):
        return arguments_json
    if not isinstance(args, dict):
        return arguments_json
    if set(args) == {wrapper} and isinstance(args[wrapper], dict):
        return arguments_json  # already wrapped
    return json.dumps({wrapper: args})

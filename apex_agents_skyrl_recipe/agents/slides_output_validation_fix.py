#!/usr/bin/env python3
"""Runtime fix for the dev-1928 PowerPoint MCP 'Output validation error: None is
not of type string' bug — applied inside the sandbox before start.sh.

Root cause (dev-1928 `dev-061726-*` images only): the newer slides server flattens
each tool's *output* schema via `_flatten_tool_schemas()`, emitting optional string
fields as {"type":"string","nullable":true}. `nullable` is OpenAPI, not JSON
Schema, so fastmcp's output validation ignores it, enforces `type:"string"`, and
rejects the null that appears on every success path (e.g. `"error": null`). The
tool actually succeeds, but the wrapper turns the result into an MCP error.

Verified (no-GPU stdio smoke, 2026-06-26): the bug reproduces only on `dev-061726-*`
images (slides server has the `_flatten_tool_schemas` code path); the older `dev-*`
and `eval-*` images do not flatten output schemas, so they are unaffected and this
script is a no-op on them.

Two idempotent prongs:
  1. main.py: nullable:true -> type:["<t>","null"]  (the validator accepts null)
  2. schema.py FlatBaseModel.model_dump/json -> exclude_none=True  (belt & suspenders)

Self-contained: auto-discovers the slides server under /app/tools/mcp_servers,
GATES on the broken code path (only patches images that flatten output schemas),
and patches the venv-imported mcp_schema copy (not just the source). Safe to run on
every world; exits 0 without changes when the pattern is absent.

Usage: python slides_output_validation_fix.py [MCP_SERVERS_ROOT]
       (default root: /app/tools/mcp_servers)
"""

import sys
from pathlib import Path

OUTPUT_SCHEMA_LINE = "tool.output_schema = flatten_schema(output_schema)"
OUTPUT_SCHEMA_PATCHED = "tool.output_schema = _json_schema_nullable(flatten_schema(output_schema))"

SCHEMA_NEEDLE = '''class FlatBaseModel(BaseModel):
    """BaseModel subclass that generates flattened JSON schemas.

    Use this instead of BaseModel for models that need LLM-compatible schemas.
    """

    @classmethod
    def model_json_schema(cls, **kwargs: Any) -> dict[str, Any]:
        """Generate a flattened JSON schema."""
        return flatten_schema(super().model_json_schema(**kwargs))
'''

SCHEMA_REPL = (
    SCHEMA_NEEDLE
    + """
    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        kwargs.setdefault("exclude_none", True)
        return super().model_dump(**kwargs)

    def model_dump_json(self, **kwargs: Any) -> str:
        kwargs.setdefault("exclude_none", True)
        return super().model_dump_json(**kwargs)
"""
)

HELPER = """def _json_schema_nullable(schema):
    if isinstance(schema, dict):
        out = {key: _json_schema_nullable(value) for key, value in schema.items()}
        nullable = bool(out.pop("nullable", False))
        if nullable:
            schema_type = out.get("type")
            if isinstance(schema_type, str):
                out["type"] = [schema_type, "null"]
            elif isinstance(schema_type, list):
                out["type"] = schema_type if "null" in schema_type else [*schema_type, "null"]
            else:
                any_of = out.get("anyOf")
                if isinstance(any_of, list):
                    out["anyOf"] = [*any_of, {"type": "null"}]
                else:
                    out["anyOf"] = [{"type": "null"}]
        return out
    if isinstance(schema, list):
        return [_json_schema_nullable(item) for item in schema]
    return schema
"""


def _find_slides_main(root: Path) -> Path | None:
    matches = list(root.glob("*/mcp_servers/slides_server/main.py"))
    if not matches:
        matches = list(root.rglob("slides_server/main.py"))
    return matches[0] if matches else None


def _patch_main(p: Path) -> bool:
    """Prong 1. Returns True if this is a broken (flattening) image."""
    text = p.read_text()
    if OUTPUT_SCHEMA_LINE not in text and OUTPUT_SCHEMA_PATCHED not in text:
        print(f"[slides-fix] no output-schema flattening in {p} — image not affected; no-op")
        return False
    if OUTPUT_SCHEMA_PATCHED in text and "def _json_schema_nullable(schema):" in text:
        print(f"[slides-fix] main.py already patched: {p}")
        return True
    updated = text
    if "def _json_schema_nullable(schema):" not in updated:
        needle = "\nasync def _flatten_tool_schemas() -> None:\n"
        if needle in updated:
            updated = updated.replace(needle, "\n" + HELPER + needle, 1)
        else:
            updated = HELPER + "\n" + updated  # fallback: prepend helper
    updated = updated.replace(OUTPUT_SCHEMA_LINE, OUTPUT_SCHEMA_PATCHED)
    if updated != text:
        p.write_text(updated)
        print(f"[slides-fix] PATCHED main.py: {p}")
    return True


def _patch_schema(p: Path) -> None:
    """Prong 2 (belt & suspenders). Patches one mcp_schema/schema.py copy."""
    if not p.exists():
        return
    text = p.read_text()
    if "def model_dump(self, **kwargs: Any)" in text:
        print(f"[slides-fix] schema.py already patched: {p}")
        return
    if SCHEMA_NEEDLE not in text:
        print(f"[slides-fix] FlatBaseModel needle not found, skipping: {p}")
        return
    p.write_text(text.replace(SCHEMA_NEEDLE, SCHEMA_REPL, 1))
    print(f"[slides-fix] PATCHED schema.py: {p}")


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "/app/tools/mcp_servers")
    main_py = _find_slides_main(root)
    if main_py is None:
        print(f"[slides-fix] no slides_server/main.py under {root}; no-op")
        return 0
    server_root = main_py.parent.parent.parent  # <server_root>/mcp_servers/slides_server/main.py

    # GATE: only patch images whose slides server flattens output schemas (dev-1928).
    affected = _patch_main(main_py)
    if not affected:
        return 0

    # Patch the venv-IMPORTED copy (what the running server actually loads) AND the source.
    for schema_py in [
        *server_root.glob(".venv/lib/python*/site-packages/mcp_schema/schema.py"),
        server_root / "packages/mcp_schema/mcp_schema/schema.py",
    ]:
        _patch_schema(schema_py)
    print("[slides-fix] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

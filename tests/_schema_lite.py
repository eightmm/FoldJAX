"""A deliberately small JSON Schema validator for FoldJAX's published schemas.

`jsonschema` is not a FoldJAX dependency, and the output contract should not
gain one just to be tested. This implements exactly the keywords the schemas
in ``src/foldjax/schemas`` use; `unsupported_keywords` lets a test fail the
moment a schema starts using one it does not, because a validator that skips
an unknown keyword passes anything.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Any

#: Keywords that constrain an instance.
ASSERTIONS = frozenset(
    {
        "type",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "enum",
        "const",
        "minimum",
        "maximum",
        "minLength",
        "pattern",
        "oneOf",
        "anyOf",
        "allOf",
        "$ref",
    }
)
#: Keywords that only annotate or organize.
ANNOTATIONS = frozenset({"$schema", "$id", "$defs", "title", "description"})
SUPPORTED = ASSERTIONS | ANNOTATIONS

_TYPES = {
    "object": lambda value: isinstance(value, dict),
    "array": lambda value: isinstance(value, list),
    "string": lambda value: isinstance(value, str),
    "integer": lambda value: isinstance(value, int) and not isinstance(value, bool),
    "number": lambda value: isinstance(value, (int, float))
    and not isinstance(value, bool),
    "boolean": lambda value: isinstance(value, bool),
    "null": lambda value: value is None,
}


def _subschemas(schema: Any) -> Iterator[dict[str, Any]]:
    if not isinstance(schema, dict):
        return
    yield schema
    for key in ("properties", "$defs"):
        for child in (schema.get(key) or {}).values():
            yield from _subschemas(child)
    for key in ("items", "additionalProperties"):
        if isinstance(schema.get(key), dict):
            yield from _subschemas(schema[key])
    for key in ("oneOf", "anyOf", "allOf"):
        for child in schema.get(key) or ():
            yield from _subschemas(child)


def unsupported_keywords(schema: dict[str, Any]) -> set[str]:
    """Every keyword used anywhere in ``schema`` that this validator ignores."""
    return {
        keyword
        for sub in _subschemas(schema)
        for keyword in sub
        if keyword not in SUPPORTED
    }


def _equal(left: Any, right: Any) -> bool:
    # JSON equality: True is not 1.
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    return left == right


def errors(instance: Any, schema: dict[str, Any], root: dict[str, Any] | None = None,
           path: str = "$") -> list[str]:
    """Every violation of ``schema`` by ``instance``, as readable strings."""
    root = schema if root is None else root
    found: list[str] = []
    if "$ref" in schema:
        reference = schema["$ref"]
        if not reference.startswith("#/$defs/"):
            raise ValueError(f"unsupported $ref {reference!r}")
        found += errors(instance, root["$defs"][reference.split("/")[-1]], root, path)
    if "type" in schema:
        allowed = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        if not any(_TYPES[name](instance) for name in allowed):
            return found + [f"{path}: {instance!r} is not of type {allowed}"]
    if "const" in schema and not _equal(instance, schema["const"]):
        found.append(f"{path}: {instance!r} != const {schema['const']!r}")
    if "enum" in schema and not any(_equal(instance, item) for item in schema["enum"]):
        found.append(f"{path}: {instance!r} not in {schema['enum']!r}")
    if _TYPES["number"](instance):
        if "minimum" in schema and instance < schema["minimum"]:
            found.append(f"{path}: {instance!r} < minimum {schema['minimum']}")
        if "maximum" in schema and instance > schema["maximum"]:
            found.append(f"{path}: {instance!r} > maximum {schema['maximum']}")
    if isinstance(instance, str):
        if "minLength" in schema and len(instance) < schema["minLength"]:
            found.append(f"{path}: shorter than {schema['minLength']}")
        if "pattern" in schema and re.search(schema["pattern"], instance) is None:
            found.append(f"{path}: {instance!r} does not match {schema['pattern']!r}")
    if isinstance(instance, dict):
        for name in schema.get("required", ()):
            if name not in instance:
                found.append(f"{path}: missing required {name!r}")
        properties = schema.get("properties", {})
        for name, value in instance.items():
            if name in properties:
                found += errors(value, properties[name], root, f"{path}.{name}")
            elif "additionalProperties" in schema:
                extra = schema["additionalProperties"]
                if extra is False:
                    found.append(f"{path}: unexpected property {name!r}")
                elif isinstance(extra, dict):
                    found += errors(value, extra, root, f"{path}.{name}")
    if isinstance(instance, list) and isinstance(schema.get("items"), dict):
        for index, item in enumerate(instance):
            found += errors(item, schema["items"], root, f"{path}[{index}]")
    for sub in schema.get("allOf", ()):
        found += errors(instance, sub, root, path)
    if "anyOf" in schema and not any(
        not errors(instance, sub, root, path) for sub in schema["anyOf"]
    ):
        found.append(f"{path}: matches none of anyOf")
    if "oneOf" in schema:
        matches = [not errors(instance, sub, root, path) for sub in schema["oneOf"]]
        if sum(matches) != 1:
            detail = [errors(instance, sub, root, path) for sub in schema["oneOf"]]
            found.append(f"{path}: matches {sum(matches)} of oneOf: {detail}")
    return found

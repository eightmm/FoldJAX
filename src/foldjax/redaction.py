"""Keep credentials out of plans and run manifests.

Backend options are intentionally open-ended, which is useful for native model
features but also means they can carry MSA server passwords and API keys.  The
backend still receives the original mapping; only user-facing provenance is
redacted here.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

REDACTED = "[REDACTED]"
_PARTS = re.compile(r"[^a-z0-9]+")
_SECRET_PARTS = {
    "auth",
    "authorization",
    "cookie",
    "credential",
    "credentials",
    "passphrase",
    "passwd",
    "password",
    "pwd",
    "secret",
    "sig",
    "signature",
}
_COMPACT_CONTAINS = {
    "accesskey",
    "accesstoken",
    "apikey",
    "authheader",
    "authtoken",
    "authorization",
    "bearertoken",
    "clientsecret",
    "cookie",
    "credential",
    "credentials",
    "idtoken",
    "jwt",
    "passphrase",
    "passwd",
    "password",
    "privatekey",
    "pwd",
    "refreshtoken",
    "secret",
    "signature",
}
_URL_SECRET_NAMES = {"bearer", "key"}
_HEADER_LINE = re.compile(r"^\s*([!#$%&'*+.^_`|~0-9A-Za-z-]+)\s*:(.*)$")


def _secret_key(key: object) -> bool:
    text = str(key).strip().lower()
    parts = tuple(part for part in _PARTS.split(text) if part)
    if any(part in _SECRET_PARTS for part in parts):
        return True
    compact = _PARTS.sub("", text)
    if any(secret in compact for secret in _COMPACT_CONTAINS):
        return True
    # A bare token is sensitive, but a substring match would also hide public
    # options such as ``tokenizer`` and ``token_count``.
    return compact.endswith(("jwt", "token"))


def _public_string(value: str) -> str:
    """Remove credentials embedded in an otherwise public URL option."""
    lines = value.splitlines(keepends=True)
    if len(lines) > 1 or (lines and lines[0].endswith(("\n", "\r"))):
        return "".join(_public_header_line(line) for line in lines)
    return _public_header_line(value)


def _public_header_line(value: str) -> str:
    """Redact one possible HTTP-header line, then scrub URL credentials."""
    content = value.rstrip("\r\n")
    ending = value[len(content) :]
    header = _HEADER_LINE.fullmatch(content)
    if header is not None and _secret_key(header.group(1)):
        return f"{header.group(1)}: {REDACTED}{ending}"
    flag, separator, _item = content.partition("=")
    if separator and flag.startswith("--") and _secret_key(flag[2:]):
        return f"{flag}={REDACTED}{ending}"
    try:
        parts = urlsplit(content)
    except ValueError:
        return value
    if not parts.scheme or not parts.netloc:
        return value
    netloc = parts.netloc.rsplit("@", 1)[-1]
    query = urlencode(
        [
            (
                name,
                REDACTED
                if name.strip().lower() in _URL_SECRET_NAMES or _secret_key(name)
                else item,
            )
            for name, item in parse_qsl(parts.query, keep_blank_values=True)
        ],
        doseq=True,
    )
    fragment = parts.fragment
    if "=" in fragment:
        fragment = urlencode(
            [
                (
                    name,
                    REDACTED
                    if name.strip().lower() in _URL_SECRET_NAMES
                    or _secret_key(name)
                    else item,
                )
                for name, item in parse_qsl(fragment, keep_blank_values=True)
            ],
            doseq=True,
        )
    return urlunsplit((parts.scheme, netloc, parts.path, query, fragment)) + ending


def _secret_cli_flag(value: str) -> bool:
    """Whether one argv token names a flag whose following value is secret."""
    token = value.strip()
    if not token.startswith("--"):
        return False
    flag = token.split("=", 1)[0][2:]
    return bool(flag) and _secret_key(flag)


def _public_sequence(value: list[Any] | tuple[Any, ...]) -> list[Any]:
    """Redact sequence values, retaining enough argv shape for provenance."""
    public: list[Any] = []
    redact_next = False
    for item in value:
        if redact_next:
            public.append(REDACTED)
            redact_next = False
            continue
        if isinstance(item, str) and _secret_cli_flag(item):
            public.append(_public_string(item))
            redact_next = "=" not in item
            continue
        public.append(redact(item))
    return public


def redact(value: Any, *, key: object | None = None) -> Any:
    """Return a JSON-friendly copy with values under secret-looking keys hidden."""
    if key is not None and _secret_key(key):
        return REDACTED
    if isinstance(value, Mapping):
        return {str(name): redact(item, key=name) for name, item in value.items()}
    if (
        isinstance(value, tuple)
        and len(value) == 2
        and isinstance(value[0], str)
    ):
        return [redact(value[0]), redact(value[1], key=value[0])]
    if isinstance(value, (list, tuple)):
        return _public_sequence(value)
    if isinstance(value, str):
        return _public_string(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)


def public_options(options: Mapping[str, Any]) -> dict[str, Any]:
    """The safe, JSON-friendly form of a backend option mapping."""
    return {str(key): redact(value, key=key) for key, value in options.items()}

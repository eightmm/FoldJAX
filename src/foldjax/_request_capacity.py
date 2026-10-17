"""Estimate the same public token target for planning and capacity warnings."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from foldjax.schema import PaddingConfig, PredictionRequest


def estimate_request_tokens(
    request: PredictionRequest, document: Mapping[str, Any] | None = None
) -> dict[str, Any] | None:
    from foldjax.input import read_job_document
    from foldjax.padding import resolve_axis
    from foldjax.slurm import estimate_tokens

    if document is None and request.input_format == "foldjax":
        try:
            document = read_job_document(Path(request.input))
        except (OSError, ValueError):
            return None
    if not isinstance(document, Mapping):
        return None
    tokens, notes = estimate_tokens(document)
    record: dict[str, Any] = {"tokens": tokens}
    if notes:
        record["tokens_not_counted"] = notes
    try:
        buckets = None
        if request.model == "alphafold3":
            from foldjax.backends.alphafold3 import _prediction_buckets

            buckets = _prediction_buckets(request, dict(request.options))
        record["token_bucket"] = (
            tokens
            if request.padding is None and buckets is None
            else resolve_axis(
                tokens, request.padding or PaddingConfig(), "tokens", buckets=buckets
            )
        )
    except ValueError as error:
        record["token_bucket"] = None
        record["reason"] = str(error)
    return record

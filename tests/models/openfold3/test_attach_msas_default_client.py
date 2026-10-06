"""`attach_msas` without a ``backend``: the documented call must build a client.

The default client used to be constructed with no endpoint, so the README's
``attach_msas(spec, alignment_dir=...)`` raised ``TypeError`` before searching.
Nothing here touches the network: the client's ``search`` is replaced.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from foldjax.models.openfold3.data import attach_msas
from foldjax.search import MsaPayload
from foldjax.search import msa as search_msa


def _spec(sequence: str) -> dict:
    return {
        "queries": {
            "q": {
                "chains": [
                    {
                        "molecule_type": "protein",
                        "chain_ids": ["A"],
                        "sequence": sequence,
                    }
                ]
            }
        }
    }


@pytest.mark.parametrize(
    ("environment", "expected"),
    [
        ({}, ("https://api.colabfold.com", "colabfold-mmseqs2")),
        (
            {
                "FOLDJAX_MSA_SERVER_URL": "http://127.0.0.1:9/",
                "FOLDJAX_MSA_SERVER_VERSION": "mine",
            },
            ("http://127.0.0.1:9", "mine"),
        ),
    ],
)
def test_default_client_uses_the_configured_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, environment, expected
) -> None:
    monkeypatch.delenv("FOLDJAX_MSA_SERVER_URL", raising=False)
    monkeypatch.delenv("FOLDJAX_MSA_SERVER_VERSION", raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    seen: list[tuple[str, str]] = []

    def search(self, sequence: str, **_options) -> MsaPayload:
        seen.append((self.host_url, self.version))
        return MsaPayload(paired=f">q\n{sequence}\n", unpaired=f">q\n{sequence}\n")

    monkeypatch.setattr(search_msa.RemoteMMseqs2Client, "search", search)
    updated = attach_msas(_spec("MKTAYIAK"), alignment_dir=tmp_path / "align")

    assert seen == [expected]
    assert updated["queries"]["q"]["chains"][0]["main_msa_file_paths"]

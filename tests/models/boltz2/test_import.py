import json
import subprocess
import sys

from foldjax.models.boltz2 import __version__


def test_import() -> None:
    assert __version__


def test_import_is_lazy_and_torch_free() -> None:
    code = """
import json
import sys
import foldjax.models.boltz2
print(json.dumps({name: name in sys.modules for name in (
    'torch', 'boltz', 'jax', 'foldjax.models.boltz2.api'
)}))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == {
        "torch": False,
        "boltz": False,
        "jax": False,
        "foldjax.models.boltz2.api": False,
    }

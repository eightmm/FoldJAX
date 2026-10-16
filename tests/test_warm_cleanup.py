"""Private warm outputs are removed on errors and process cancellation."""

from __future__ import annotations

import json
import os
import select
import signal
import subprocess
import sys
import textwrap

import pytest

from foldjax.warmup import _warm_directory


def test_warm_directory_removes_partial_outputs_after_an_exception(tmp_path):
    with pytest.raises(RuntimeError, match="prediction failed"):
        with _warm_directory(tmp_path, prefix=".boltz2-") as scratch:
            (scratch / "partial.cif").write_text("unfinished")
            raise RuntimeError("prediction failed")

    assert not list(tmp_path.iterdir())


@pytest.mark.skipif(os.name != "posix", reason="POSIX signal behavior")
def test_warm_directory_preserves_an_existing_sigterm_handler(tmp_path):
    previous = signal.getsignal(signal.SIGTERM)
    received = []

    def handler(signum, _frame):
        received.append(signum)

    signal.signal(signal.SIGTERM, handler)
    try:
        with _warm_directory(tmp_path, prefix=".boltz2-") as scratch:
            os.kill(os.getpid(), signal.SIGTERM)
            assert received == [signal.SIGTERM]
            assert scratch.is_dir()
        assert signal.getsignal(signal.SIGTERM) is handler
        assert not scratch.exists()
    finally:
        signal.signal(signal.SIGTERM, previous)


@pytest.mark.skipif(os.name != "posix", reason="POSIX signals and pipe readiness")
def test_sigterm_removes_warm_outputs_and_stops_the_process(tmp_path):
    source = textwrap.dedent("""
        import json
        import signal
        import sys
        from pathlib import Path
        from foldjax.warmup import _warm_directory

        try:
            with _warm_directory(Path(sys.argv[1]), prefix='.alphafold3-') as scratch:
                (scratch / 'partial.cif').write_text('unfinished')
                print(json.dumps({'scratch': str(scratch)}), flush=True)
                try:
                    while True:
                        signal.pause()
                except SystemExit:
                    raise RuntimeError('native SystemExit adapter intercepted signal')
        finally:
            print(f'restored={signal.getsignal(signal.SIGTERM) == signal.SIG_DFL}',
                  flush=True)
    """)
    child = subprocess.Popen(
        [sys.executable, "-c", source, str(tmp_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert select.select([child.stdout], [], [], 20)[0], "warm did not start"
        ready = child.stdout.readline()
        assert ready, child.stderr.read()
        scratch = json.loads(ready)["scratch"]
        assert os.path.isdir(scratch)
        child.send_signal(signal.SIGTERM)
        stdout, stderr = child.communicate(timeout=10)
        assert child.returncode == 128 + signal.SIGTERM, stderr
        assert "restored=True" in stdout
        assert not list(tmp_path.iterdir())
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=10)

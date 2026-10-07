<!-- W3 (packaging, CI, docs, version): entries for the next CHANGELOG.md
"Unreleased" section. Merge them under the matching headings. -->

### Added

- **`--templates required` (`templates="required"`).** Searches exactly as
  `auto` does and fails the run when the search cannot run, when a searched
  chain keeps no template, or when the job has no protein chain to search --
  where `auto` warns and folds template-free, which a batch only sees as a
  successful run. `template_max_date` accepts either searching policy, and
  `foldjax_run.json`'s `templates` enum gains `required`.

- **A `templates` extra (`kalign-python`)** for the Kalign realignment behind
  template search, so it no longer needs the whole `openfold3-preprocess`
  extra (which still includes it). Not a base dependency: its wheels cover
  x86-64 Linux and macOS only, and an aarch64 install would build it from C.

- **`foldjax_source` in `foldjax_run.json`:** the package version and, in a
  checkout, `git describe --always --dirty` (null for an installed wheel or
  without git). Optional, like the other fields added within schema 1.0.

- **A `Dockerfile`** (CUDA 13 base, `uv sync --frozen` from the lockfile,
  weights and caches on a store mounted at `FOLDJAX_HOME=/foldjax`), with
  `CUDA_EXTRA`, `ALPHAFOLD3`, `OPENFOLD3_PREPROCESS` and `TEMPLATES` build
  arguments; the AlphaFold 3 variant carries the toolchain its first-use
  extension build needs. See docs/install.md#docker.

- **A pre-commit configuration:** `ruff check`, and `ruff format` applied to
  the staged lines only (`scripts/ruff_format_changed_lines.py`), since most
  files predate `ruff format` and the stock hook would reformat them whole.

### Changed

- **CI finishes again.** The single 30-minute job had not completed since
  2026-09-11 (the suite needs 75-90 minutes on one runner). It is now six
  parallel shards (`.github/workflows/tests.yml`: the orchestration suite, one
  per large port, and a catch-all for the rest of `tests/models`), with
  coverage combined across shards for the 80% gate, lint and the lockfile check
  in their own job, AlphaFold 3's compiled runtime cached instead of rebuilt
  inside a test, runners pinned to `ubuntu-24.04`, and actions on current
  Node 24 majors (`checkout@v7`, `setup-uv@v10`, `cache@v6`,
  `upload-artifact@v7`, `download-artifact@v8`). Tests marked `slow` -- now
  also three multi-minute context-parallel compiles -- and the RCSB `network`
  tests run in a new nightly workflow. The CPU parity subset stays manual: it
  needs weights beyond the Actions cache and fixtures that exist on one host
  (docs/parity-cpu.md).

- **`foldjax doctor`** checks template realignment the way the search does
  (`find_spec("kalign")`), lists the `FOLDJAX_TEMPLATE_*` variables, labels
  the Protenix-native template lines `protenix-native` and points them at
  `--option template_mmcif_dir` rather than an internal environment variable,
  prints `pip install 'foldjax[extra]'` or `uv sync --inexact --extra extra`
  to match how FoldJAX was installed (`--inexact`, because a bare `uv sync`
  uninstalls the extras it is not given), and reports the checkout's
  `git describe` and any stale editable-install metadata version.

- **`foldjax compare` help** says nucleic acids are fitted on C4' and names
  `compare_structures.csv`.

### Fixed

- The banner naming the vendored Boltz-2 parity modules left uncollected
  without torch (24 modules) never printed: `pytest_report_header` sat in
  `tests/models/conftest.py`, where pytest does not call it. It now runs from
  `tests/conftest.py`, and a test asserts it reaches the session header.

- `tests/models/boltz2/conftest.py` no longer sets the deprecated
  `JAX_PLATFORM_NAME` at import, which was inert or forced every later suite
  onto the CPU depending on collection order.

- Documentation drift: README's CPU-only recipe (undone by the next `uv run`;
  now `UV_NO_DEFAULT_GROUPS=1`), README links made absolute for PyPI, a pip
  install section, examples linked and corrected (`--msa auto`, no Chai or
  "OpenFold3 refuses any other name" claims), the Colab notebook pinned to the
  `v0.1.0` tag (and a test that the pin is a tag), AlphaFold 3's recycle count
  in python-api.md (3; upstream's CLI default is 10), nonexistent
  `--no-compile`/`--msa-deletions` in cli.md, the Protenix weights label in
  cli.md's example, `Job.write` bonds in input.md, the Kalign version claim in
  model-versions.md.

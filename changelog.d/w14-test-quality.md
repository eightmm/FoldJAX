### Fixed

- **An in-process `foldjax predict` leaves progress as it found it.**
  `cli.main(["predict", ...])` without `--quiet` turned progress lines on for
  the rest of the process, so a notebook or test that called it kept getting
  stage lines on stderr -- and search warnings that should have been
  `UserWarning`s arrived as progress lines instead. The command now restores
  the host's setting when it returns or raises.
- **A generated output directory containing `..` is refused before it is
  created.** The run-root check compared unnormalized paths, so
  `<root>/../x` passed it, the directory was created outside the root, and only
  then refused.

### Changed

- **The 2x2-grid arms of the Protenix and OpenFold3 atom context-parallel
  tests run nightly** (`slow`). Pull requests keep the 1-D arm and the 3x3 grid,
  whose odd side is what tells a ring hop's direction apart.
- **AlphaFold 3 runtime tests build the runtime once per session** and skip
  with the build's error where it cannot be built, instead of each retrying a
  multi-minute CMake build. `FOLDJAX_REQUIRE_AF3_RUNTIME=1` (set on CI's core
  shard) makes that a failure.
- **Every test starts from fresh process state.** An autouse fixture restores
  `os.environ`, the JAX persistent-cache settings, progress, and the memory
  policy's one-time warnings and recorded decision after each test, and resets
  the last three before it; six tests that passed only because no earlier test
  had turned progress on now pass either way.

### Added

- Tests for refusals that had none or that a mutant survived: a negative,
  one-sided or one-past-the-end template index map; the reason each changed
  request field gives on `--resume` (now including `templates`,
  `template_max_date`, the resolved input path and unverifiable recorded
  dependencies); an MSA cache entry recorded under another key; every
  generated-directory symlink and escape refusal in `foldjax.api`,
  `foldjax.output`, `foldjax.input` and `foldjax.template_search`; hand-forged
  `torch.save` archives; and the `pb_valid` rule that a check PoseBusters could
  not compute does not count as passed. The OpenFold3 trunk-only and Protenix
  confidence-backend tests now run the code instead of reading its source.

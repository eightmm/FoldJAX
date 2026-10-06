## Unreleased

### Added

- **`THIRD_PARTY_NOTICES`** lists every third-party file FoldJAX carries with
  its licence and copyright: AlphaFold 3, the OpenFold3 data pipeline, the
  adapted Boltz preprocessing, Biohub's ESMFold2 constants, the
  Transformers-derived chemistry tables, and the Biotite functions inside the
  OpenFold3 pipeline, whose BSD-3-Clause text it reproduces. It also names the
  installed dependencies whose terms are not permissive, among them NVIDIA's
  proprietary cuEquivariance ops and CUDA library wheels. The wheel installs it
  in `foldjax-*.dist-info/licenses/` beside `LICENSE` and `NOTICE`
  (`license-files`), and the Docker image copies it.
- **Nightly dependency audit**: `pip-audit` checks the lockfile's default,
  all-extras CUDA 13 and all-extras CUDA 12 environments and fails the nightly
  on a known vulnerability, and the run uploads CycloneDX SBOMs of both CUDA
  generations. PR CI is unchanged, so a newly published upstream advisory
  cannot fail an unrelated pull request.

### Changed

- **`hydra-core` is no longer a dependency.** Nothing imported it (or
  `omegaconf`), and it carried GHSA-2cp2-2r3c-7p7r; `omegaconf` and
  `antlr4-python3-runtime` leave the lock with it.
- **Security patch releases in the lock**: urllib3 2.8.0, multidict 6.9.1,
  oauthlib 4.0.0 and werkzeug 3.1.9 (all transitive). `pip-audit` reports no
  known vulnerability in any exported environment.
- **`docs/licences.md`** says which AlphaFold 3 terms ship in the wheel
  (`OUTPUT_TERMS_OF_USE.md` among them) and which installs pull in NVIDIA's
  proprietary GPU wheels: the CUDA extras, the default `gpu` group of a bare
  `uv sync`, and the Docker image.

### Fixed

- **Attribution text.** The root `NOTICE` no longer calls all five ports
  independent reimplementations: it names the upstream source the Boltz-2,
  OpenFold3 and ESMFold2 ports carry, gives OpenFold3's holder as upstream's
  LICENSE does (AlQuraishi Laboratory) and OpenDDE's as its source headers do
  (Aureka AI Research), and drops a stale OpenFold3 line count and an edit
  marker for a file that is not vendored. The Boltz-2 `NOTICE` lists the
  adapted `parse/{a3m,csv}.py`, `write/{mmcif,pdb}.py` and
  `crop/{affinity,cropper}.py`; its `LICENSE` names the real path
  (`src/foldjax/models/boltz2/data/`); the OpenDDE `NOTICE`'s copyright line
  names a holder; the OpenFold3 `NOTICE` uses upstream's holder and points to
  the Biotite licence.

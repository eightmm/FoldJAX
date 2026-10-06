# Licenses and parameter terms

Per model: the code license FoldJAX vendors under, and the terms the
published parameters carry, which are not always the same thing.

| model | vendored at | code license | parameters / additional terms | upstream |
|---|---|---|---|---|
| `alphafold3` | `foldjax.models.alphafold3` | Apache-2.0 | [AlphaFold 3 Model Parameters Terms](https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md): non-commercial use by/for non-commercial organizations; must be received directly from Google; redistribution restricted | [google-deepmind/alphafold3](https://github.com/google-deepmind/alphafold3) |
| `boltz2` | `foldjax.models.boltz2` | MIT | MIT, code and weights; academic and commercial use | [jwohlwend/boltz](https://github.com/jwohlwend/boltz) |
| `esmfold2` | `foldjax.models.esmfold2` | MIT + third-party notices | MIT for ESMFold2 and ESMC-6B; [Biohub Acceptable Use Policy](https://biohub.org/acceptable-use-policy/) also applies | [biohub/ESMFold2](https://huggingface.co/biohub/ESMFold2) |
| `opendde` | `foldjax.models.opendde` | Apache-2.0 | Apache-2.0 released checkpoints | [aurekaresearch/OpenDDE](https://huggingface.co/aurekaresearch/OpenDDE#license) |
| `openfold3` | `foldjax.models.openfold3` | Apache-2.0 | Apache-2.0 model and parameters; academic and commercial use | [OpenFold/OpenFold3](https://huggingface.co/OpenFold/OpenFold3) |
| `protenix` | `foldjax.models.protenix` | Apache-2.0 | Apache-2.0 code and model parameters of the v1.x and earlier checkpoints (the `released`, `base-20250630` and v0.5.0 mini profiles); academic and commercial use. **Protenix-v2 parameters (`--profile v2`) are proprietary:** upstream's README states they "are proprietary and confidential information of the rights holder, are not released under any open-source license, and may not be reproduced, distributed, sublicensed, disclosed, or otherwise transferred to any third party in any form without the express prior written consent of the rights holder." FoldJAX never downloads them; you supply the file | [bytedance/Protenix](https://github.com/bytedance/Protenix) |

These are publisher summaries, not legal advice. Review the linked current
terms before use; third-party chemistry assets, sequence databases and other
referenced data retain their own licenses and terms.

## Terms that ship with the code

AlphaFold 3's `OUTPUT_TERMS_OF_USE.md`, `WEIGHTS_TERMS_OF_USE.md` and
`WEIGHTS_PROHIBITED_USE_POLICY.md` are installed with the wheel, beside the
vendored source in `foldjax/models/alphafold3/_upstream/` (the output terms
also sit in its `alphafold3/` package). They are not software licences and
FoldJAX grants nothing under them: the output terms govern what you may do
with predictions made using AlphaFold 3 parameters, independently of the
Apache-2.0 code.

The repository's
[`THIRD_PARTY_NOTICES`](https://github.com/eightmm/FoldJAX/blob/main/THIRD_PARTY_NOTICES),
installed in the wheel's `foldjax-*.dist-info/licenses/` beside `LICENSE` and
`NOTICE`, lists every carried third-party file with its licence and copyright,
including the Biotite (BSD-3-Clause) functions inside the vendored OpenFold3
pipeline, whose licence text it reproduces.

## Proprietary GPU dependencies

FoldJAX is Apache-2.0, but a GPU environment installs NVIDIA software that is
not open source. `cuequivariance-ops-cu12`/`-cu13` and
`cuequivariance-ops-jax-cu12`/`-cu13` are under NVIDIA's proprietary Software
License Agreement, and the `nvidia-*` CUDA library wheels JAX's CUDA plugin
pulls in are under NVIDIA's proprietary terms (`nvidia-ml-py` is BSD). They
arrive through the `cuda12` and `cuda13` extras, through the default `gpu`
dependency group that a bare `uv sync` enables, and through the Docker image,
which always installs one of the two CUDA extras. A CPU-only install
(`uv sync --no-default-groups`, or `pip install foldjax` without a CUDA
extra) installs none of them. Anyone redistributing a GPU environment or
image redistributes these wheels under NVIDIA's terms.

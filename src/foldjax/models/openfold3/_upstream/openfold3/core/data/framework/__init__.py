# Copyright 2026 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from pathlib import Path

# VENDORED CHANGE. Upstream built each module name out of the *path*:
#
#     __import__(".".join(list(path.parts[-6:-1]) + [stem]))
#
# which reconstructs `openfold3.core.data.framework.single_datasets.<mod>` only
# while this package sits exactly six components from the filesystem root of its
# import path. Vendored under `foldjax.models.openfold3._upstream`, the last six
# components are unchanged but the package prefix is not, so every name it built
# pointed at a top-level `openfold3` that does not exist here.
#
# Deriving the name from `__package__` instead is equivalent upstream and correct
# anywhere: it asks the import system where this module actually lives rather
# than inferring it from a directory layout.


def _import_all_py_files_from_dir(directory: Path):
    """Import every Python file in `directory`, a sibling subpackage."""
    package = f"{__package__}.{directory.name}"
    for path in sorted(directory.glob("*.py")):
        if path.stem == "__init__":
            continue
        __import__(f"{package}.{path.stem}")
    return


_import_all_py_files_from_dir(Path(__file__).parent / "single_datasets")

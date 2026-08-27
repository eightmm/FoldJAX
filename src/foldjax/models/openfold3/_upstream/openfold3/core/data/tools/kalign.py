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

from functools import lru_cache

import kalign

_KALIGN_CACHE_LIMIT = 32


@lru_cache(maxsize=_KALIGN_CACHE_LIMIT)
def _run_kalign_cached(sequences: tuple[str, ...]) -> list[str]:
    """Wrapper around kalign.align with bounded recent-job reuse.

    Both the input tuple and the complete aligned output can contain tens of
    thousands of residues.  Keeping 512 distinct calls therefore pins old
    query/template strings long after their jobs finish.  A small LRU retains
    the useful within-job and immediate retry hits without letting a service's
    mixed-query history become a large resident sequence store.
    """
    return kalign.align(list(sequences))


def run_kalign(
    sequences: list[str],
) -> list[str]:
    """Runs Kalign on the provided A3M string and returns the aligned sequences.

    Args:
        sequences (list[str]):
            Sequences to be aligned. In the template pipeline,
            the first sequence is the query, and the rest are templates
            sequences to be realigned to it from hmmsearch.
    Returns:
        list:
            The aligned sequences as a list of strings.
    """
    return _run_kalign_cached(tuple(sequences))

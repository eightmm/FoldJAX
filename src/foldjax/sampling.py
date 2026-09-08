"""Recycle conventions at the common request/native adapter boundary.

Defaults here are FoldJAX policy overrides. An omitted value for other models
must still reach their selected checkpoint/configuration unchanged.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RecyclePolicy:
    # PredictionRequest historically counts additional rounds for these ports.
    request_counts_additional: bool = False
    native_counts_additional: bool = False
    default_native: int | None = None

    def from_trunk_passes(self, passes: int | None) -> int | None:
        return None if passes is None else passes - int(self.request_counts_additional)

    def to_native(self, recycles: int) -> int:
        if self.request_counts_additional == self.native_counts_additional:
            return recycles
        return recycles + int(self.request_counts_additional) - int(
            self.native_counts_additional
        )

    @property
    def native_minimum(self) -> int:
        return 0 if self.native_counts_additional else 1


_POLICIES = {
    # AF3 Algorithm 1: four total passes; Boltz evaluation: five added rounds;
    # ESMFold2 evaluation: ten total loops. Preserve effective cache identities.
    "alphafold3": RecyclePolicy(
        request_counts_additional=True, native_counts_additional=True,
        default_native=3,
    ),
    "boltz2": RecyclePolicy(
        request_counts_additional=True, native_counts_additional=True,
        default_native=5,
    ),
    "esmfold2": RecyclePolicy(
        request_counts_additional=True, native_counts_additional=True,
        default_native=9,
    ),
    "openfold3": RecyclePolicy(request_counts_additional=True),
    "protenix": RecyclePolicy(),
    "opendde": RecyclePolicy(),
}
_DEFAULT_POLICY = RecyclePolicy()


def get_recycle_policy(model: str) -> RecyclePolicy:
    """Keep unregistered adapters' historical identity translation intact."""
    return _POLICIES.get(model, _DEFAULT_POLICY)

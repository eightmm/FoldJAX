"""Backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from foldjax.schema import ModelCapabilities, PredictionRequest, PredictionResult


class Backend(ABC):
    name: str

    #: Option keys that change the compiled program. Only these participate in
    #: the cache namespace, so options that merely affect output formatting do
    #: not fragment the compilation cache.
    compile_options: tuple[str, ...] = ()

    #: Model-neutral sampling knob -> this backend's own option name. Every
    #: model calls "how many structures" something different, so the request
    #: carries one spelling and the adapter translates it.
    sampling_options: dict[str, str] = {}

    def apply_sampling(self, request: PredictionRequest) -> dict[str, Any]:
        """Merge the request's sampling knobs into this backend's options.

        A knob the backend cannot express is an error, and so is setting both
        the neutral knob and its native spelling: silently preferring one would
        change how many structures come back without changing the exit code.
        """
        options = dict(request.options)
        for knob, value in request.sampling.items():
            native = self.sampling_options.get(knob)
            if native is None:
                raise ValueError(f"{self.name} does not support {knob}")
            if native in options:
                raise ValueError(
                    f"{knob} and the native option {native!r} were both set for "
                    f"{self.name}; pass one of them"
                )
            options[native] = value
        return options

    @abstractmethod
    def capabilities(self) -> ModelCapabilities: ...

    @abstractmethod
    def predict(self, request: PredictionRequest) -> PredictionResult: ...

    def cache_profile(self, request: PredictionRequest) -> dict[str, Any]:
        """Return the compile-relevant identity of ``request`` for this backend.

        Values are stringified because option values include ``Path`` objects
        and the profile is hashed as JSON.
        """
        return {
            key: str(request.options[key])
            for key in self.compile_options
            if key in request.options
        }

from abc import ABC, abstractmethod

from foldjax.models.boltz2.data.types import Tokenized


class Cropper(ABC):
    """Base interface for token/atom croppers."""

    @abstractmethod
    def crop(
        self,
        data: Tokenized,
        max_tokens: int,
        max_atoms: int | None = None,
    ) -> Tokenized:
        raise NotImplementedError

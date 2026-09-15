from foldjax.models.boltz2.data._torch import Tensor, torch

pad = torch.nn.functional.pad


def pad_dim(data: Tensor, dim: int, pad_len: float, value: float = 0) -> Tensor:
    """Pad a tensor along a given dimension.

    Parameters
    ----------
    data : Tensor
        The input tensor.
    dim : int
        The dimension to pad.
    pad_len : float
        The padding length.
    value : int, optional
        The value to pad with.

    Returns
    -------
    Tensor
        The padded tensor.

    """
    if pad_len == 0:
        return data

    total_dims = len(data.shape)
    padding = [0] * (2 * (total_dims - dim))
    padding[2 * (total_dims - 1 - dim) + 1] = pad_len
    return pad(data, tuple(padding), value=value)

import numpy as np
import torch
import torch.nn.functional as functional

from foldjax.models.boltz2.bridge.torch_mapping import map_transition_state_dict
from foldjax.models.boltz2.models.primitives.transition import transition_forward

PREFIX = "msa_module.layers.0.msa_transition"


class TorchTransition(torch.nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.norm = torch.nn.LayerNorm(dim, eps=1e-5)
        self.fc1 = torch.nn.Linear(dim, hidden_dim, bias=False)
        self.fc2 = torch.nn.Linear(dim, hidden_dim, bias=False)
        self.fc3 = torch.nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor, chunk_size: int | None = None) -> torch.Tensor:
        x = self.norm(x)
        if chunk_size is None:
            return self.fc3(functional.silu(self.fc1(x)) * self.fc2(x))

        out = x.new_zeros((*x.shape[:-1], self.fc3.out_features))
        hidden_dim = self.fc3.in_features
        for start in range(0, hidden_dim, chunk_size):
            stop = min(start + chunk_size, hidden_dim)
            hidden = functional.silu(
                functional.linear(x, self.fc1.weight[start:stop])
            ) * functional.linear(x, self.fc2.weight[start:stop])
            out = out + functional.linear(hidden, self.fc3.weight[:, start:stop])
        return out


def test_transition_forward_matches_torch_no_chunk() -> None:
    torch.manual_seed(0)
    module = TorchTransition(dim=5, hidden_dim=11).eval()
    x = torch.randn(2, 3, 5, dtype=torch.float32)
    params = map_transition_state_dict(_prefixed_state_dict(module), PREFIX)

    with torch.no_grad():
        expected = module(x)
    actual = transition_forward(params, np.asarray(x))

    np.testing.assert_allclose(
        np.asarray(actual),
        expected.detach().numpy(),
        rtol=1e-5,
        atol=1e-5,
    )


def test_transition_forward_matches_torch_chunked() -> None:
    torch.manual_seed(1)
    module = TorchTransition(dim=5, hidden_dim=11).eval()
    x = torch.randn(2, 3, 5, dtype=torch.float32)
    params = map_transition_state_dict(_prefixed_state_dict(module), PREFIX)

    with torch.no_grad():
        expected = module(x, chunk_size=4)
    actual = transition_forward(params, np.asarray(x), chunk_size=4)

    np.testing.assert_allclose(
        np.asarray(actual),
        expected.detach().numpy(),
        rtol=1e-5,
        atol=1e-5,
    )


def _prefixed_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        f"{PREFIX}.{key}": value.detach().clone()
        for key, value in module.state_dict().items()
    }

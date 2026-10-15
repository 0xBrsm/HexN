"""Checkpoint state survives automatic device-specific Adam execution."""
import copy

import pytest
import torch

from hexn.optim import adam, adamw


@pytest.mark.parametrize("factory,kind", [(adam, torch.optim.Adam), (adamw, torch.optim.AdamW)])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_restore_preserves_moments_and_hyperparameters_and_places_steps(factory, kind, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("GPU required")
    parameter = torch.nn.Parameter(torch.tensor([1., 2.], device=device))
    old = kind([parameter], lr=.002, eps=1e-5, fused=False, foreach=False)
    parameter.grad = torch.tensor([.2, -.3], device=device)
    old.step()
    saved = copy.deepcopy(old.state_dict())
    # Simulate a checkpoint loaded on CPU before the trainer moves its net.
    for state in saved["state"].values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.cpu()
    new = factory([parameter], lr=.5, eps=.1)
    new.load_state_dict(saved)
    fused = device == "cuda"
    group = new.param_groups[0]
    assert group["fused"] is fused
    assert group["foreach"] is (False if fused else None)
    assert group["lr"] == .002 and group["eps"] == 1e-5
    assert saved["param_groups"][0]["fused"] is False
    assert new.state[parameter]["step"].device.type == device
    for key in ("step", "exp_avg", "exp_avg_sq"):
        assert torch.equal(new.state[parameter][key].cpu(), saved["state"][0][key])
    parameter.grad = torch.tensor([.1, -.2], device=device)
    new.step()
    assert new.state[parameter]["step"].item() == 2
    assert torch.isfinite(parameter).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_fused_adam_supports_grad_scaler_after_restoring_unfused_state():
    parameter = torch.nn.Parameter(torch.ones(2, device="cuda"))
    old = torch.optim.Adam([parameter], fused=False)
    parameter.grad = torch.ones_like(parameter)
    old.step()
    optimizer = adam([parameter])
    optimizer.load_state_dict(old.state_dict())
    optimizer.zero_grad(set_to_none=True)
    scaler = torch.amp.GradScaler("cuda")
    scaler.scale(parameter.square().sum()).backward()
    scaler.step(optimizer)
    scaler.update()
    assert optimizer.state[parameter]["step"].item() == 2
    assert torch.isfinite(parameter).all()

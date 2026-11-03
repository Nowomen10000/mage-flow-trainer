"""Fail before model/cache loading when the selected varlen kernel is unavailable."""

import torch


def _probe_varlen(device, dtype):
    from torch.nn.attention.varlen import varlen_attn

    # Exercise forward AND backward; importing the Python API does not mean the
    # wheel includes its CUDA kernels (notably on native Windows). No RNG draws.
    with torch.enable_grad(), torch.cuda.device(device):
        q, k, v = [torch.ones(266, 2, 128, device=device, dtype=dtype,
                             requires_grad=True) for _ in range(3)]
        cu = torch.tensor([0, 129, 266], device=device, dtype=torch.int32)
        out = varlen_attn(q, k, v, cu, cu, 137, 137)
        gradients = torch.autograd.grad(out.float().square().mean(), (q, k, v))
        if not all(torch.isfinite(x).all().item() for x in (out, *gradients)):
            raise RuntimeError("varlen forward/backward produced non-finite values")


def check_attention_backend(backend, device, dtype):
    if backend != "torch_varlen":
        return
    device = torch.device(device)
    try:
        if device.type != "cuda":
            raise RuntimeError("torch_varlen requires a supported GPU kernel")
        _probe_varlen(device, dtype)
    except (ImportError, RuntimeError, NotImplementedError) as exc:
        raise RuntimeError(
            f"torch_varlen forward/backward is unavailable with PyTorch {torch.__version__} "
            f"on {device} ({dtype}). The Python API alone is insufficient: native Windows "
            "wheels may omit the required Flash Attention kernel. Select "
            "train.attention_backend='sdpa' (and disable resolution packing if required "
            "by your model), or use a compatible PyTorch build/Linux/WSL2. "
            "A newer version alone does not guarantee kernel support. "
            f"Original error: {exc}"
        ) from exc

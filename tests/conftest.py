"""Shared fixtures for opt-in distributed GPU regressions."""
import os

import pytest


@pytest.fixture(scope="session")
def ddp_device():
    # Recreating NCCL's default group between tests can race its TCPStore
    # teardown. Keep one group for the entire two-rank pytest invocation.
    import torch
    import torch.distributed as dist

    if (os.environ.get("MAGEFLOW_GPU_TESTS") != "1" and os.environ.get("BLUFLOW_GPU_TESTS") != "1") or os.environ.get("WORLD_SIZE") != "2":
        pytest.skip("requires two-rank GPU opt-in")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    try:
        yield torch.device("cuda", rank)
    finally:
        dist.destroy_process_group()

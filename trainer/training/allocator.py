"""Process-local allocator defaults; call before importing Torch."""
import os
import sys


def configure_allocator():
    # Either spelling is a deliberate user choice, including an empty value.
    if "PYTORCH_ALLOC_CONF" in os.environ or "PYTORCH_CUDA_ALLOC_CONF" in os.environ:
        return
    # The pinned Windows build omits PYTORCH_C10_DRIVER_API_SUPPORTED:
    # expandable_segments is ignored there and only produces a warning.
    if sys.platform == "linux":
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

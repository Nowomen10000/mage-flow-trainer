"""CPU DDP integration: torchrun --standalone --nproc_per_node=2 THIS_FILE ROOT KIND."""
import os
import sys
from pathlib import Path
import torch
import torch.distributed as dist
from test_resolution_schedule import Samples, make_trainer, enable_checkpoint_saving


def main():
    root = Path(sys.argv[1])
    root.mkdir(parents=True, exist_ok=True)
    # Materialize fixtures once before each rank fingerprints the same cache files.
    dist.init_process_group('gloo')
    try:
        if dist.get_rank() == 0:
            Samples(root)
        dist.barrier()
        first = make_trainer(root, kind=sys.argv[2], accumulation=2)
        first.accelerator.end_training = lambda: None
        enable_checkpoint_saving(first, root)
        first.train()
        assert first.stage_images == [12, 8, 12]
        assert first.stage_attempts == [1, 1, 3]
        assert first.scheduler.scheduler.last_epoch == 10
        final = {k:v.detach().clone() for k,v in first._unwrap(first.transformer).state_dict().items()}
        trace = list(first.trace)
        for p in first.transformer.parameters():
            reference = p.detach().clone()
            dist.broadcast(reference, 0)
            assert torch.equal(reference, p)
        first.accelerator.free_memory()
        resumed = make_trainer(root, kind=sys.argv[2], accumulation=2)
        resumed.accelerator.end_training = lambda: None
        enable_checkpoint_saving(resumed, root)
        resumed._resume(str(root/'output'/'staged-test-step000003-state'))
        resumed.train()
        assert resumed.stage_images == [12, 8, 12]
        assert resumed.trace == trace[6:]
        for k,v in resumed._unwrap(resumed.transformer).state_dict().items():
            torch.testing.assert_close(v, final[k], rtol=0, atol=0)
        dist.barrier()
        if dist.get_rank() == 0:
            print('PASS: two-rank',sys.argv[2],'stages, accumulation, sample counts, synchronized weights, and exact full-state resume.')
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()

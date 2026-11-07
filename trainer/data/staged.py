"""Rank-sharded, full-size batches from a single tier's deterministic sample stream."""
import hashlib
import json
import random
from pathlib import Path
from .dataset import BucketBatchSampler


class StagedBatchSampler(BucketBatchSampler):
    def __init__(self, dataset, batch_size, schedule, steps, *, seed=0, rank=0, world=1, accumulation=1):
        super().__init__(dataset, batch_size, seed=seed)
        self.schedule = schedule
        self.budgets = schedule.budgets(steps)
        self.rank, self.world, self.accumulation = rank, world, accumulation
        self.indices = {s.resolution: [i for i, e in enumerate(dataset.entries) if e.resolution == s.resolution]
                        for s in schedule.stages}
        if any(not indices for indices in self.indices.values()):
            raise ValueError('Every resolution stage needs at least one eligible sample')
        self.stage = self.cursor = self.count = 0
        files = {}
        entries = []
        for e in dataset.entries:
            path = str(Path(e.latent_path).resolve())
            if path not in files:
                st = Path(path).stat()
                files[path] = (st.st_size, st.st_mtime_ns)
            entries.append((path, e.bucket, e.resolution, e.tags, e.nl))
        self.identity = hashlib.sha256(json.dumps(dict(
            entries=entries, files=files, budgets=self.budgets,
            resolutions=[s.resolution for s in schedule.stages], batch_size=batch_size,
            world=world, accumulation=accumulation, seed=seed,
        ), sort_keys=True).encode()).hexdigest()

    def configure(self, stage, cursor, updates):
        self.stage, self.cursor = stage, cursor
        self.count = updates * self.accumulation

    def __len__(self):
        return self.count

    def __iter__(self):
        # Snapshot state before prefetch begins. The consumed cursor is maintained
        # by the training loop, never inferred from how far workers prefetched.
        stage, cursor, count = self.stage, self.cursor, self.count
        resolution = self.schedule.stages[stage].resolution
        indices = self.indices[resolution]
        bs = self._by_tier[resolution]
        cycle_cached, order = -1, None
        for micro in range(cursor, cursor + count):
            first = (micro * self.world + self.rank) * bs
            batch = []
            for pos in range(first, first + bs):
                cycle, offset = divmod(pos, len(indices))
                if cycle != cycle_cached:
                    order = list(indices)
                    random.Random(self.seed + 1000003 * stage + 9176 * cycle).shuffle(order)
                    cycle_cached = cycle
                # Also give each stage a distinct deterministic caption stream.
                batch.append((order[offset], cycle + 1000003 * stage, None, None))
            yield batch

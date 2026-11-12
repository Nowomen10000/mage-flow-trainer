"""Experimental MOSS adapter learning-rate multipliers, indexed by joint block."""
from dataclasses import dataclass, field
import math


def default_ranges():
    return [dict(start=start, end=end, lr_mul=1.0)
            for start, end in ((0, 24), (25, 32), (33, 41))]


@dataclass
class BlockLRConfig:
    enabled: bool = False
    ranges: list[dict] = field(default_factory=default_ranges)

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError('block_lr.enabled must be true or false')
        if not isinstance(self.ranges, list):
            raise ValueError('block_lr.ranges must be an array of block ranges')
        covered = set()
        for row in self.ranges:
            if not isinstance(row, dict) or set(row) != {'start', 'end', 'lr_mul'}:
                raise ValueError('Each block_lr range needs exactly start, end and lr_mul')
            start, end, multiplier = row['start'], row['end'], row['lr_mul']
            if (type(start) is not int or type(end) is not int
                    or not 0 <= start <= end <= 41):
                raise ValueError('block_lr ranges must have inclusive integer indices 0 <= start <= end <= 41')
            if (isinstance(multiplier, bool) or not isinstance(multiplier, (float, int))
                    or not math.isfinite(multiplier) or multiplier < 0):
                raise ValueError('block_lr.lr_mul must be finite and nonnegative (0 freezes adapters in that range)')
            indices = set(range(start, end + 1))
            if covered & indices:
                raise ValueError('block_lr ranges must not overlap')
            covered.update(indices)

    def expanded(self):
        """Unlisted blocks, and all blocks while disabled, retain multiplier 1."""
        values = [1.0] * 42
        if self.enabled:
            for row in self.ranges:
                values[row['start']:row['end'] + 1] = [float(row['lr_mul'])] * (row['end'] - row['start'] + 1)
        return values

    def check_resume(self, saved_multipliers):
        # Old states predate this option and used multiplier 1 everywhere.
        expected = [1.0] * 42 if saved_multipliers is None else saved_multipliers
        if expected != self.expanded():
            raise ValueError('Full-state resume block LR multipliers changed; use weights-only resume to start a new LR plan')

    def summary(self):
        values = self.expanded()
        spans = []
        start = 0
        for end in range(42):
            if end == 41 or values[end + 1] != values[start]:
                spans.append(f'{start}-{end}: x{values[start]:g}')
                start = end + 1
        return 'block LR ' + ', '.join(spans) + ' (text refiner unchanged)'

"""Treat malformed Triton group indexes as cache misses, not training failures."""

import json
import logging

from triton.runtime.cache import FileCacheManager


class RecoveringFileCacheManager(FileCacheManager):
    def get_group(self, filename):
        try:
            return super().get_group(filename)
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Do not unlink: another DDP rank may have repaired this index since
            # we read it. Triton recompiles on a miss and atomically replaces it.
            # Permission/disk errors still propagate instead of being hidden.
            logging.getLogger(__name__).warning(
                "Ignoring malformed Triton cache index %s; rebuilding the kernel.",
                self._make_path(f"__grp__{filename}"),
            )
            return None

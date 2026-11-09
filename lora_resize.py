#!/usr/bin/env python
"""Open the LoRA rank-resize utility, or pass INPUT --rank N for CLI use."""
from trainer.tools.lora_resize import main

if __name__ == '__main__':
    raise SystemExit(main())

#!/usr/bin/env python
"""Standalone entry point for the bundled MOSS Q/K/V converter."""
from trainer.tools.lora_qkv_convert import convert_file, gui, main

if __name__ == '__main__':
    raise SystemExit(main())

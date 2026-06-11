"""Common modules for Audio Deepfake Source Attribution (ADSA).

This package is shared by both experimental trials:
  * Trial 1 (single PTM representation)  -> ``main_single.py``
  * Trial 2 (two-PTM fused representation) -> ``main_fusion.py``

The code is intentionally modular so individual PTM extractors, the CNN
front-end, the loss, and the metrics can be swapped/compared in isolation
to justify which representation performs better.
"""

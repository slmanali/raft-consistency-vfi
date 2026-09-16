# Validation record

The initial authoring environment ran 18 automated tests successfully:

- Zero motion, signed translation and subpixel mass conservation.
- Reverse-flow sampling at the mapped coordinate.
- Inconsistent and out-of-bounds flow handling.
- Weighted collisions, identical fallback and exact temporal endpoints.
- Nonfinite-flow rejection and analytically checked image metrics.
- Manifest import and validation, average evaluation, external float scoring.

Run them with `python -m unittest discover -s tests -v` after installing the package.
These tests verify the CPU reference implementation. They do not demonstrate an
improvement on real video.

PyTorch, Torchvision, LPIPS, pretrained weights and SNU-FILM images were not
available in the authoring environment. RAFT/RIFE adapters and GPU execution have
therefore not been exercised. A small integration run, followed by the full
predeclared evaluation, is required before reporting experimental conclusions.

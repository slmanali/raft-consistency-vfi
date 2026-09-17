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

At initial authoring, pretrained-model dependencies and SNU-FILM were unavailable.
The following subsequent run supplies experimental evidence from the laptop.

## Completed RAFT experiment: 17 September 2026

The author supplied `raft_ctv2_cu124_results.zip`, produced with clean source commit
`2e3b7f1f650633ed592557b3536ff94ad44733bc`. Its metadata records a completed
1,240-triplet RAFT/LPIPS run on an RTX 3050 Laptop GPU, PyTorch 2.6.0+cu124 and
Torchvision 0.21.0+cu124. RAFT uses C_T_V2, 20 updates and float32; alpha=0.01,
beta=0.5. Synthesis uses the NumPy CPU implementation.

The supplied archive was checked independently in the authoring environment:

- 4,960 unique `(id, variant)` rows and 310 triplets in each difficulty.
- All 16 summary rows and all 2,480 paired differences reproduced exactly.
- Metric values are finite; variants agree on split, group and image dimensions.
- Saved examples and their PSNR/SSIM were inspected. PNG quantization explains
  the small difference from metrics measured before image export.

This audit validates the supplied logs and saved examples. It does not rerun RAFT
on the authoring machine. The source manifest was supplied separately after the
archive audit. Its SHA-256 matches the run log exactly, and all 1,240 ids, split
labels and sequence groups match the metric records. Each triplet has distinct
paths in one sequence directory and symmetric frame indices. The frame step is
1, 2, 4 or 8 in Easy, Medium, Hard or Extreme, respectively. See
[`manifest_check.json`](results/2026-09-17/manifest_check.json).
The original source image files and official test lists were not independently
compared. Full float predictions and flows are also absent.
RIFE has not been run or compared. See [the results record](results/2026-09-17/README.md)
for numerical findings, hashes and measurement limits.

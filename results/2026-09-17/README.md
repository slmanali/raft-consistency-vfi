# RAFT consistency experiment — 17 September 2026

The supplied run completed on 1,240 SNU-FILM triplets, 310 per difficulty.
At the fixed settings tested, geometric validity alone (`raft_uniform`) gives
better mean PSNR, SSIM and LPIPS than consistency weighting in every difficulty.
This is a negative result for these particular synthesis rules and parameters.

## Overall results

These are arithmetic means of per-triplet measurements. Higher PSNR and SSIM
and lower LPIPS are better. Every method uses the same 1,240 triplets.

| Variant | PSNR, dB | SSIM | LPIPS | Fallback pixels, % | Mean time, s |
|:--|--:|--:|--:|--:|--:|
| Average | 26.664 | 0.7922 | 0.1218 | n/a | 0.002 |
| RAFT uniform | **31.254** | **0.9030** | **0.0661** | 1.14 | 2.269 |
| RAFT hard | 30.382 | 0.8979 | 0.0710 | 8.32 | 2.458 |
| RAFT soft | 30.647 | 0.8998 | 0.0707 | 3.98 | 2.458 |

The CSV column `hole_fraction` measures the fraction of pixels with accumulated
splat weight at most 1e-8, before filling with the unwarped temporal average.
Its recorded zero for the Average baseline is a placeholder; this measurement
does not apply to a method that does not splat.

Hard weighting lowers mean PSNR by 0.872 dB relative to uniform; soft weighting
lowers it by 0.607 dB. Hard improves PSNR in 83/1,240 triplets (6.7%), soft in
155/1,240 (12.5%). When all four difficulties are combined within each recorded
sequence group, both weighted variants have lower mean PSNR in all 31 groups.
These are descriptive paired comparisons, with no independence assumption or
significance test. Some individual triplets improve.

| Difficulty | Uniform PSNR | Hard PSNR | Soft PSNR |
|:--|--:|--:|--:|
| Easy | 37.496 | 36.575 | 36.899 |
| Medium | 33.732 | 32.769 | 33.075 |
| Hard | 29.087 | 28.211 | 28.461 |
| Extreme | 24.701 | 23.973 | 24.154 |

## Conditions and interpretation

- Clean experiment commit: `2e3b7f1f650633ed592557b3536ff94ad44733bc`.
- RAFT Large C_T_V2, 20 updates, float32, no resizing.
- Fixed alpha=0.01 and beta=0.5; no post-result threshold tuning.
- RTX 3050 6GB Laptop GPU; Python 3.12.3, PyTorch 2.6.0+cu124,
  Torchvision 0.21.0+cu124, LPIPS package 0.1.4 (AlexNet, metric version 0.1).
- The same two flow fields, geometric validity check, forward bilinear splat,
  joint normalization and fallback rule are used for all three RAFT variants.
- Eight original image sizes, up to 1280 by 720. Two warmups per image size.
- NumPy synthesis runs on the CPU. Timing includes both flow directions,
  preparation, transfers, weighting and synthesis, excluding metrics and file I/O.
  Flow is reused experimentally but its full cost is charged to each variant.

The larger fallback region is consistent with removal or attenuation of useful
contributions, but this run does not isolate how much of the quality loss is
caused by fallback filling rather than the changed combination of surviving
contributions. It does not show that consistency is unhelpful with other
thresholds, learned fusion or different hole restoration.

Times describe one mixed CPU/GPU run. They are not GPU-only latency, repeated
timing estimates or evidence of energy efficiency. No RIFE result is included.

## Verification

The uploaded archive `raft_ctv2_cu124_results.zip` has SHA-256:

`6f5bcebd3899c157219d179e1543826528e66092539867b591ffaf51fc3b298b`

The audit checked:

- `run.json` is marked `completed`; 1,240 unique ids and 4,960 unique id/variant rows.
- 310 triplets in each difficulty; variants agree on group and image dimensions.
- All 16 summary rows reproduce exactly from the per-image records.
- All 2,480 recorded paired differences reproduce exactly.
- All reported numeric metrics are finite.
- PSNR and SSIM of 16 saved predictions were recomputed from their PNG exports.
  The small differences from the logged float-image values are consistent with
  8-bit rounding (maximum absolute PSNR deviation: 0.044644 dB). Article tables
  use the original logged values, not the quantized exports.

This checks the supplied records and examples; it is not a second GPU execution.
The archive omits the source manifest, full float predictions, flow arrays and
hole masks. The original `data/snu_film.csv` was subsequently supplied separately.
Its SHA-256 exactly matches the recorded value:

`0c849ffa7390531391a6b91b567e2a0f4faec2edccd4644f14701d9daab21867`

All 1,240 manifest ids, split labels and sequence groups match the metric records.
The manifest has 1,240 unique ordered triplets, with distinct paths in the same
sequence directory and the target index midway between the input indices. Frame
steps are 1, 2, 4 and 8 in Easy, Medium, Hard and Extreme. Each of the 31 groups
has ten triplets per subset. These checks establish consistency with the logged
run. Original source image files and official test lists were not independently
compared.

## Files

- `run.json`: unchanged run metadata.
- `summary.csv`: unchanged difficulty-level summary from the supplied run.
- `overall_summary.csv`: means over all 1,240 triplets.
- `paired_summary.csv`: hard/soft minus uniform, including per-triplet win counts.
- `saved_examples.csv`: original metric records for the four saved triplets.
- `png_metric_check.csv`: quantized-export PSNR/SSIM checks.
- `audit.json`: counts, checks and hashes of the supplied source files.
- `manifest_check.json`: verification of the manifest supplied after the archive.

The original per-image records and complete example images remain in the source
archive; this directory is a compact analysis record. The updated
[Russian manuscript](../../paper/article_ru.md) and its
[English version](../../paper/article_en.md) contain the method, complete
quality table, limitations and the selected Extreme example. The sample id was
selected before the full run; the common crop was chosen during analysis and
does not replace the aggregate evaluation.

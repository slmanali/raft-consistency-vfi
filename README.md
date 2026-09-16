# RAFT consistency for video frame interpolation

Code and a Russian manuscript for a controlled experiment on forward–backward
flow consistency. Two RAFT flow fields are reused across three synthesis variants:
geometric validity alone, a binary consistency mask, and a continuous weight.
The interpolation operator and hole-filling rule are identical.

**Status:** reference implementation and article draft. CPU geometry, metrics,
manifest handling and evaluation tests have been run. Pretrained RAFT, RIFE and
LPIPS integration require validation on the experiment machine. No SNU-FILM
benchmark results are included. The article has no invented numerical results.

## Research question

Does consistency weighting reduce interpolation errors, and how much does it
increase reliance on fallback pixels? Consistency is a heuristic for correspondence
reliability, not a calibrated confidence score or a ground-truth occlusion mask.
Known optical flow and splatting methods are credited in the manuscript.

## Install and run the CPU checks

Use Python 3.10 or newer:

```bash
git clone https://github.com/slmanali/raft-consistency-vfi.git
cd raft-consistency-vfi
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
rcvfi demo --output outputs/demo
```

The demo uses a supplied constant translation field. It tests coordinate geometry;
it is not an optical-flow benchmark. It needs no network, model or dataset.

For learned models, install a compatible PyTorch/Torchvision pair using the
[official installation instructions](https://pytorch.org/get-started/locally/),
then install the optional dependencies:

```bash
python -m pip install -e '.[models]'
```

The first RAFT or LPIPS run downloads pretrained weights. The default experiment
uses the explicit RAFT Large `C_T_V2` weights, 20 updates, float32, and no resizing.
The selected checkpoint hash and package versions are saved with each run.
Do not use a moving `DEFAULT` weight alias for published experiments.

## Prepare SNU-FILM

Obtain the frames and official `test-easy.txt`, `test-medium.txt`, `test-hard.txt`
and `test-extreme.txt` lists from the [authors](https://myungsub.github.io/CAIN/).
Put the frame tree under `data/SNU-FILM` and the lists under `data/SNU-FILM/lists`.
Keep the original sequence directories. Data and pretrained weights are excluded
from version control.

```bash
rcvfi make-snu-manifest \
  --data-root data/SNU-FILM \
  --list-root data/SNU-FILM/lists \
  --output data/snu_film.csv
```

Each manifest row contains `id,input0,target,input1,split,group`. The middle image
is ground truth. Paths in hand-authored CSV manifests are relative to the manifest
directory or absolute. The importer validates file existence, distinct triplet
paths, sequence directories and ids. Inspect `group` before using sequence-level
statistics: a directory must correspond to one source sequence.

Do not tune on official test triplets. If a tuning set is added, separate source
videos, including videos shared between difficulty modes. Dataset provenance and
overlap with pretrained-model training data need to be reported for publication.

## Run the controlled experiment

```bash
rcvfi evaluate \
  --manifest data/snu_film.csv \
  --method raft --device cuda --weights C_T_V2 --iterations 20 \
  --alpha 0.01 --beta 0.5 --warmup 2 --lpips \
  --output outputs/raft_ctv2
```

This evaluates `average`, `raft_uniform`, `raft_hard` and `raft_soft` using the same
triplets. `raft_uniform` already rejects endpoints outside the image; this avoids
confounding basic boundary validity with consistency. All variants use bilinear
forward splatting with joint normalization and the unwarped temporal average for
pixels with accumulated weight at most 1e-8.

Use `--split easy` or `--limit 10` for an installation smoke run; do not mix such
runs with full benchmark results. Images are not saved by default. Supply
`--save-ids` with a text file of sample ids selected before inspecting results to
save only those cases, including the input images, target and weight maps.

Every completed run writes:

- `run.json`: completion status, manifest hash, arguments, weights and environment.
- `per_image.csv`: one row per triplet and method.
- `summary.csv`: means within each SNU-FILM mode.
- `paired_deltas.csv`: hard/soft minus uniform for each triplet in RAFT runs;
  undefined differences involving infinite PSNR are left empty.

The output directory must be new. A failed or interrupted run remains marked
`running`; partial CSV files must not be reported as a completed benchmark.
Perfect reconstruction has infinite PSNR and remains `inf` in CSV, without an
arbitrary cap. LPIPS is absent when `--lpips` is not supplied; it is never replaced
by another metric.

## RIFE baseline

Use the [official ECCV2022-RIFE repository](https://github.com/hzwer/ECCV2022-RIFE)
and its paper checkpoint. Do not mix Practical-RIFE weights with this adapter.
Clone the code into `third_party/ECCV2022-RIFE` and put its `flownet.pkl` checkpoint
at `weights/rife/flownet.pkl`. Install the dependencies required by that checkout.
The adapter checks that the checkout is clean and records its exact commit.

```bash
python scripts/export_rife.py \
  --repo third_party/ECCV2022-RIFE \
  --weights weights/rife/flownet.pkl \
  --manifest data/snu_film.csv --device cuda \
  --output outputs/rife_export

rcvfi score \
  --manifest data/snu_film.csv \
  --predictions outputs/rife_export/predictions \
  --provenance outputs/rife_export/provenance.json \
  --timings outputs/rife_export/timings.csv \
  --device cuda --lpips --output outputs/rife_scores
```

The adapter preserves the original BGR model input convention and exports RGB
float32 `.npy` predictions, avoiding PNG quantization. It saves all predictions
temporarily for scoring. PNG predictions from another implementation are also
accepted by `score`, but their 8-bit quantization is recorded. No learned-model
adapter has been executed in the authoring environment; validate a small run
against the selected upstream implementation before the full experiment.

## Evaluation and timing

PSNR uses the MSE over all RGB pixels in [0,1], followed by a per-triplet average.
SSIM uses an 11-by-11 Gaussian window, sigma 1.5, population covariance, K1=0.01,
K2=0.03 and an RGB channel mean. Only the SSIM map is cropped by five pixels.
LPIPS uses AlexNet, version 0.1, with input mapped to [-1,1]. These definitions may
differ from published tables; compare outputs scored here, not copied numbers.

The NumPy synthesis is a CPU reference. RAFT may run on CUDA, but this is a mixed
CPU/GPU pipeline. Times include tensor preparation, device transfers, both flow
directions, weighting and synthesis. They exclude file I/O, model loading,
warmup, metric evaluation and visual export. All three RAFT timings include the
full flow cost, even though flows are computed once and reused in the experiment.
They are per-variant pipeline costs, not total experiment wall time.

Report device and software versions. Do not describe these numbers as GPU-only
latency or use them to claim energy efficiency. Repeated measurements are needed
for timing uncertainty. SNU-FILM metrics alone do not measure temporal flicker,
true occlusion accuracy, or optical-flow endpoint error.

## Article and source files

- `paper/article_ru.md`: Russian manuscript with equations and nine references.
- `src/raft_consistency_vfi/core.py`: weighting and deterministic forward splatting.
- `src/raft_consistency_vfi/models.py`: optional Torchvision RAFT adapter.
- `src/raft_consistency_vfi/cli.py`: evaluation and external prediction scoring.
- `tests/`: geometric and workflow tests independent of learned models.

To rebuild Word output, install Pandoc and python-docx, then run:

```bash
python -m pip install python-docx
python scripts/build_manuscript.py --output paper/article_ru.docx
```

Pandoc writes native Word equations. Inspect the rendered pages before submitting.
The manuscript currently establishes the method and protocol. Experimental
results, hardware details and the final conclusions must come from actual runs.

## Attribution

The manuscript cites RAFT, UnFlow, Softmax Splatting, RIFE, SNU-FILM, SSIM and LPIPS.
This repository contains an independent NumPy implementation of the mathematical
operations and adapters that load separately obtained dependencies. Third-party
code, weights and datasets remain subject to their respective terms. No license
for the original research code has been selected in this initial draft.

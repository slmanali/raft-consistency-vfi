# RAFT consistency for video frame interpolation

Code and a Russian manuscript for a controlled experiment on forward–backward
flow consistency. Two RAFT flow fields are reused across three synthesis variants:
geometric validity alone, a binary consistency mask, and a continuous weight.
The interpolation operator and hole-filling rule are identical.

**Status:** the RAFT/LPIPS experiment completed on 17 September 2026 on all
1,240 SNU-FILM triplets (310 per difficulty). Its 4,960 metric records reproduce
the reported summaries and paired differences. With the fixed settings tested,
RAFT uniform has better mean PSNR, SSIM and LPIPS than hard and soft weighting
in every difficulty. See the [results and verification record](results/2026-09-17/README.md).
The Russian manuscript now includes these measurements. RIFE remains pending.

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

## Install learned models on the experiment laptop

The reported Linux laptop has an RTX 3050 with 6 GB VRAM, NVIDIA driver 565.77,
and CUDA Toolkit 12.4. Its `nvidia-smi` display of CUDA 12.7 describes the driver's
CUDA support; `nvcc --version` describes the separately installed compiler.
PyTorch's prebuilt wheels bring their own CUDA runtime dependencies. The local
toolkit does not choose which wheel pip installs.

An unconstrained `pip install -e '.[models]'` selected PyTorch 2.14.0 with CUDA 13
in the reported installation. CUDA 13 requires driver 580 or newer, so it cannot
initialize on this driver. See NVIDIA's
[compatibility table](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).
The following profile uses the official PyTorch 2.6.0 / Torchvision 0.21.0 CUDA
12.4 pair from the [PyTorch version table](https://pytorch.org/get-started/previous-versions/).
It requires no change to the laptop's system CUDA installation.

Run from the repository root with Python 3.12. A separate environment avoids
mixing dependencies from the earlier CUDA 13 installation:

```bash
python3.12 -m venv .venv-cu124
source .venv-cu124/bin/activate
python -m pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -c constraints-cu124.txt -e '.[models]'
python -m pip check
```

Install the PyTorch wheels first: the constraints file fixes the builds but does
not supply a package index. Keep using this constraints file for subsequent
model dependency installs in this environment. For another GPU or driver, select
and record an appropriate pair using the
[official instructions](https://pytorch.org/get-started/locally/).

Verify that this Python environment can execute a CUDA operation:

```bash
python - <<'PY'
import torch
import torchvision

print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("PyTorch CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
assert torch.cuda.is_available(), "CUDA initialization failed in this environment"
print("GPU:", torch.cuda.get_device_name(0))
x = torch.ones((32, 32), device="cuda")
y = x @ x
torch.cuda.synchronize()
assert y[0, 0].item() == 32.0
print("CUDA operation: OK")
PY
```

Expected versions are `2.6.0+cu124`, `0.21.0+cu124` and runtime `12.4`.
This checks CUDA execution. On a new installation, validate the RAFT and LPIPS
integrations with the small dataset run below before a full experiment.

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
  --data-root /home/salman/Documents/GitHub/PhaseNet/SNU-FILM \
  --list-root /home/salman/Documents/GitHub/PhaseNet/SNU-FILM/eval_modes \
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

After preparing the manifest, start with one triplet:

```bash
rcvfi evaluate \
  --manifest data/snu_film.csv --split easy --limit 1 \
  --method raft --device cuda --weights C_T_V2 --iterations 20 \
  --alpha 0.01 --beta 0.5 --warmup 1 --lpips \
  --output outputs/raft_cu124_smoke
```

The first run downloads weights. A successful run writes `run.json` with
`"status": "completed"` and the metric CSV files. This is an installation check,
not a benchmark result. `--limit` reduces the number of triplets, not the memory
required by each image. Full-resolution RAFT can exceed 6 GB because its
all-pairs correlation volume grows quadratically with the number of feature
pixels, as follows from the correlation tensor shape in the
[Torchvision implementation](https://docs.pytorch.org/vision/0.21/_modules/torchvision/models/optical_flow/raft.html).
If this run reports out-of-memory, keep the error and image dimensions so that
a memory strategy can be chosen and documented before the full experiment.

After the small run succeeds, evaluate the complete manifest:

```bash
rcvfi evaluate \
  --manifest data/snu_film.csv \
  --method raft --device cuda --weights C_T_V2 --iterations 20 \
  --alpha 0.01 --beta 0.5 --warmup 2 --lpips \
  --output outputs/raft_ctv2_cu124
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
The earlier CUDA initialization failure may already have created
`outputs/raft_ctv2`; the commands above use separate destinations. Choose another
new output name if you repeat either command.
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
  --manifest /home/salman/Documents/GitHub/PhaseNet/SNU-FILM/snu_film.csv --device cuda \
  --output outputs/rife_export

rcvfi score \
  --manifest /home/salman/Documents/GitHub/PhaseNet/SNU-FILM/snu_film.csv \
  --predictions outputs/rife_export/predictions \
  --provenance outputs/rife_export/provenance.json \
  --timings outputs/rife_export/timings.csv \
  --device cuda --lpips --output outputs/rife_scores
```

The adapter preserves the original BGR model input convention and exports RGB
float32 `.npy` predictions, avoiding PNG quantization. It saves all predictions
temporarily for scoring. PNG predictions from another implementation are also
accepted by `score`, but their 8-bit quantization is recorded. The completed
experiment uses RAFT and LPIPS on the experiment laptop. The RIFE adapter has
not yet been validated against the selected upstream implementation; start
with a small run before the full RIFE comparison.

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
- `paper/article_en.md`: English version with the same equations, measurements and references.
- `src/raft_consistency_vfi/core.py`: weighting and deterministic forward splatting.
- `src/raft_consistency_vfi/models.py`: optional Torchvision RAFT adapter.
- `src/raft_consistency_vfi/cli.py`: evaluation and external prediction scoring.
- `tests/`: geometric and workflow tests independent of learned models.

To rebuild Word output, install Pandoc and python-docx, then run:

```bash
python -m pip install python-docx
python scripts/build_manuscript.py --output paper/article_ru.docx
python scripts/build_manuscript.py --language en --output paper/article_en.docx
```

Pandoc writes native Word equations. Inspect the rendered pages before submitting.
The manuscript reports the completed RAFT experiment, hardware and software,
paired comparisons and limitations. The measurements support a negative result
for these fixed weighting rules; they do not establish that consistency is
unhelpful in other interpolation architectures.

## Attribution

The manuscript cites RAFT, UnFlow, Softmax Splatting, RIFE, SNU-FILM, SSIM and LPIPS.
This repository contains an independent NumPy implementation of the mathematical
operations and adapters that load separately obtained dependencies. Third-party
code, weights and datasets remain subject to their respective terms. No license
for the original research code has been selected in this initial draft.

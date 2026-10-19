# VLM Defect Inspection

Fine-tuning a small vision-language model (Qwen2-VL-2B, QLoRA) to inspect industrial parts on the MVTec AD dataset, and benchmarking it honestly against PatchCore, a strong anomaly-detection baseline.

The goal is not to beat PatchCore at detection. It is to test whether a small VLM can add something PatchCore cannot: a named defect type and a plain-language location, with a confidence score.

## Status

| Week | Focus | Status |
|---|---|---|
| 1 | Data exploration + leakage-safe split | Done |
| 2 | VLM fine-tuning (QLoRA + Unsloth, Kaggle T4) | Next |
| 3 | Evaluation vs PatchCore + failure analysis | Planned |
| 4 | Quantization (FP16 / INT8 / INT4) + API serving (vLLM, FastAPI) | Planned |

## Week 1 findings

- **5354 images**, 15 categories (10 objects, 5 textures), 73 category/defect-type pairs.
- **The train split has 0 defective images.** All 1258 defects are in test, so the VLM's defect examples must come from the test split. This makes leakage control the central design problem.
- **72.9% of test images are defective**, so a model that always answers "defect" scores ~73% accuracy. Evaluation uses AUROC and F1, not accuracy.
- **39.7% of defects cover under 1% of the image.** Screw and grid combine tiny defects (median 0.29% and 0.60%) with grayscale images, so they are predicted to be the hardest categories.

## The split

The test set is split 50/50, stratified by `category/defect_type` (seed 42), so even rare types appear on both sides.

| Role | Images | Used for |
|---|---|---|
| `train_good` | 3629 | PatchCore memory bank; source of good examples for the VLM |
| `vlm_train` | 862 | VLM fine-tuning only |
| `eval` | 863 | Scoring **both** models; neither ever trains on it |

Known limitations:
- Scores are on our eval half, not the full published test set, so PatchCore is re-run on the same half for a fair comparison.
- Rare defect types have only 4-5 eval images, so results are reported per category and per kind (object/texture), not per defect type. Toothbrush and transistor are flagged as low-confidence.

The split is enforced by `check_split()` and covered by unit tests, including a test that a deliberately injected leak is caught.

## Repository layout

```
src/defect_vlm/split.py      leakage-safe split + checks
tests/                       unit tests (pytest)
notebooks/01_mvtec_exploration.ipynb   dataset exploration
notebooks/02_split.ipynb               creates splits.csv on Kaggle
```

## Run the tests

```
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
pytest -q
```

## Dataset and license

MVTec AD (Bergmann et al., CVPR 2019), released under CC BY-NC-SA 4.0. This is a non-commercial portfolio project.

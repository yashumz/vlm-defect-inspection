"""Leakage-safe split of MVTec AD for VLM fine-tuning vs evaluation.

Roles assigned to every image:
  train_good : MVTec train/ images (all good). PatchCore memory bank; the VLM
               may also sample "good" examples from here.
  vlm_train  : half of MVTec test/ (good + defect). The VLM fine-tunes on these.
  eval       : the other half of test/. BOTH models are scored here and
               neither model ever trains on it.

The test half is split stratified by "category/defect_type", so every defect
type (even rare ones like wood/color with 8 images) appears in both halves.
"""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

ROLES = ("train_good", "vlm_train", "eval")


def make_split(manifest: pd.DataFrame, seed: int = 42, eval_frac: float = 0.5) -> pd.DataFrame:
    df = manifest.copy().reset_index(drop=True)
    df["stratum"] = df["category"] + "/" + df["defect_type"]
    df["role"] = "train_good"

    test_idx = df.index[df["split"] == "test"]
    vlm_idx, eval_idx = train_test_split(
        test_idx,
        test_size=eval_frac,
        stratify=df.loc[test_idx, "stratum"],
        random_state=seed,
    )
    df.loc[vlm_idx, "role"] = "vlm_train"
    df.loc[eval_idx, "role"] = "eval"
    return df


def check_split(df: pd.DataFrame) -> None:
    """Raise AssertionError if the split leaks or is unbalanced."""
    vlm = set(df.loc[df.role == "vlm_train", "path"])
    ev = set(df.loc[df.role == "eval", "path"])
    tr = set(df.loc[df.role == "train_good", "path"])

    assert not (vlm & ev), f"LEAK: {len(vlm & ev)} images in both vlm_train and eval"
    assert not (tr & ev), "LEAK: a train image is in eval"
    assert df["path"].is_unique, "duplicate paths in manifest"
    assert set(df["role"]) <= set(ROLES), f"unknown role: {set(df['role']) - set(ROLES)}"
    assert (df.loc[df.role == "train_good", "label"] == "good").all(), "train_good must be all good"
    assert (df.loc[df.split == "train", "role"] == "train_good").all(), "train images must stay train_good"

    defect_strata = set(df.loc[df.label == "defect", "stratum"])
    for role in ("vlm_train", "eval"):
        have = set(df.loc[(df.role == role) & (df.label == "defect"), "stratum"])
        missing = defect_strata - have
        assert not missing, f"{role} is missing defect types: {sorted(missing)}"

    for cat, g in df[df.role == "eval"].groupby("category"):
        assert set(g["label"]) == {"good", "defect"}, f"eval/{cat} needs both good and defect images"

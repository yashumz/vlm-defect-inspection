import pandas as pd
import pytest

from defect_vlm.split import check_split, make_split


def fake_manifest():
    rows = []
    spec = {
        "bottle": {"good_train": 20, "good_test": 6, "broken": 8, "contamination": 9},
        "wood": {"good_train": 15, "good_test": 5, "color": 8, "hole": 10},
    }
    for cat, s in spec.items():
        for i in range(s["good_train"]):
            rows.append(dict(category=cat, split="train", defect_type="good", label="good", path=f"{cat}/train/good/{i}.png"))
        for dtype, n in s.items():
            if dtype.startswith("good"):
                continue
            for i in range(n):
                rows.append(dict(category=cat, split="test", defect_type=dtype, label="defect", path=f"{cat}/test/{dtype}/{i}.png"))
        for i in range(s["good_test"]):
            rows.append(dict(category=cat, split="test", defect_type="good", label="good", path=f"{cat}/test/good/{i}.png"))
    return pd.DataFrame(rows)


def test_split_passes_all_checks():
    check_split(make_split(fake_manifest()))


def test_no_image_in_two_roles_and_nothing_lost():
    m = fake_manifest()
    s = make_split(m)
    assert len(s) == len(m)
    assert s.groupby("path")["role"].nunique().max() == 1


def test_rare_type_in_both_halves():
    s = make_split(fake_manifest())
    wood_color = s[s.stratum == "wood/color"]
    assert set(wood_color.role) == {"vlm_train", "eval"}


def test_train_images_never_evaluated():
    s = make_split(fake_manifest())
    assert not ((s.split == "train") & (s.role == "eval")).any()


def test_same_seed_same_split():
    m = fake_manifest()
    a, b = make_split(m, seed=42), make_split(m, seed=42)
    assert (a.role == b.role).all()


def test_check_catches_a_leak():
    s = make_split(fake_manifest())
    leaked = s[s.role == "eval"].head(1).assign(role="vlm_train")
    with pytest.raises(AssertionError, match="LEAK"):
        check_split(pd.concat([s, leaked], ignore_index=True))

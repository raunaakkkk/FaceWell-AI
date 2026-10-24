import importlib.util
import sys
from pathlib import Path

import pytest
from PIL import Image


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "prepare_mrl_dataset.py"
SPEC = importlib.util.spec_from_file_location("prepare_mrl_dataset", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def record(name: str, label: str, subject: str) -> object:
    return MODULE.ParsedImage(Path(name), name, subject, label)


def test_parse_mrl_filename_maps_state_field():
    assert MODULE.parse_mrl_filename("s0001_01842_0_0_1_0_0_01.png") == ("s0001", "open")
    assert MODULE.parse_mrl_filename("s0001_00001_0_0_0_0_0_01.png") == ("s0001", "closed")


def test_parse_mrl_filename_rejects_ambiguous_names():
    with pytest.raises(ValueError):
        MODULE.parse_mrl_filename("s0001_00001_0_0_2_0_0_01.png")
    with pytest.raises(ValueError):
        MODULE.parse_mrl_filename("not-an-mrl-image.png")


def test_split_is_reproducible_and_group_disjoint():
    records = [
        record(f"s{i:04d}_{j:05d}_0_0_{j % 2}_0_0_01.png", "open" if j % 2 else "closed", f"s{i:04d}")
        for i in range(1, 7)
        for j in range(1, 5)
    ]
    first = MODULE.split_by_subject(records, seed=7)
    second = MODULE.split_by_subject(records, seed=7)
    assert [[item.relative_path for item in first[key]] for key in MODULE.SPLITS] == [
        [item.relative_path for item in second[key]] for key in MODULE.SPLITS
    ]
    owners = {item.subject_id: split for split, items in first.items() for item in items}
    assert all(all(owners[item.subject_id] == split for item in items) for split, items in first.items())


def test_inspect_images_reports_corrupt_files_without_relabeling(tmp_path):
    valid = tmp_path / "s0001_00001_0_0_0_0_0_01.png"
    Image.new("L", (2, 2)).save(valid)
    corrupt = tmp_path / "s0002_00001_0_0_1_0_0_01.png"
    corrupt.write_bytes(b"not an image")
    records, issues = MODULE.inspect_images(tmp_path)
    assert [(item.subject_id, item.label) for item in records] == [("s0001", "closed")]
    assert issues[0]["path"] == corrupt.name
    assert issues[0]["issue"].startswith("unreadable_image:")

"""Prepare reproducible, subject-disjoint MRL eye-state manifests."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    from PIL import Image
except ImportError:  # pragma: no cover - reported by the CLI
    Image = None


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".pgm"}
SPLITS = ("train", "validation", "test")
SPLIT_FRACTIONS = {"train": 0.70, "validation": 0.15, "test": 0.15}


@dataclass(frozen=True)
class ParsedImage:
    path: Path
    relative_path: str
    subject_id: str
    label: str


def parse_mrl_filename(filename: str) -> tuple[str, str]:
    """Return (subject_id, label) for the MRL filename convention.

    MRL names are sXXXX_YYYYY_A_B_C_D_E_FF.png. D (parts[4]) is
    the eye state: 0 is closed and 1 is open.
    """
    parts = Path(filename).stem.split("_")
    if len(parts) != 8 or not parts[0].startswith("s") or not parts[0][1:].isdigit():
        raise ValueError("expected sXXXX_YYYYY_A_B_C_D_E_FF filename")
    if parts[4] not in {"0", "1"}:
        raise ValueError("eye-state field must be 0 or 1")
    return parts[0], "open" if parts[4] == "1" else "closed"


def inspect_images(mrl_root: Path) -> tuple[list[ParsedImage], list[dict[str, str]]]:
    """Parse filenames and verify image readability without changing labels."""
    parsed: list[ParsedImage] = []
    issues: list[dict[str, str]] = []
    for path in sorted(mrl_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        try:
            subject_id, label = parse_mrl_filename(path.name)
        except ValueError as exc:
            issues.append({"path": str(path.relative_to(mrl_root)), "issue": f"invalid_filename: {exc}"})
            continue
        if Image is None:
            raise RuntimeError("Pillow is required to verify image readability")
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception as exc:
            issues.append({"path": str(path.relative_to(mrl_root)), "issue": f"unreadable_image: {exc}"})
            continue
        parsed.append(
            ParsedImage(
                path=path,
                relative_path=path.relative_to(mrl_root).as_posix(),
                subject_id=subject_id,
                label=label,
            )
        )
    return parsed, issues


def _target_counts(total: int) -> dict[str, int]:
    targets = {split: int(total * SPLIT_FRACTIONS[split]) for split in SPLITS}
    targets["train"] += total - sum(targets.values())
    return targets


def split_by_subject(records: Iterable[ParsedImage], seed: int = 20261024) -> dict[str, list[ParsedImage]]:
    """Assign complete subject groups to deterministic, approximately balanced splits."""
    groups: dict[str, list[ParsedImage]] = defaultdict(list)
    for record in records:
        groups[record.subject_id].append(record)
    if len(groups) < 3:
        raise ValueError("at least three subject groups are required for train/validation/test")

    total = sum(len(items) for items in groups.values())
    class_totals = Counter(record.label for record in records)
    targets = _target_counts(total)
    target_classes = {
        split: {label: class_totals[label] * SPLIT_FRACTIONS[split] for label in class_totals}
        for split in SPLITS
    }
    rng = random.Random(seed)
    subject_ids = list(groups)
    rng.shuffle(subject_ids)
    subject_ids.sort(key=lambda subject: len(groups[subject]), reverse=True)
    assignments = {split: [] for split in SPLITS}
    counts = {split: Counter() for split in SPLITS}

    for index, subject_id in enumerate(subject_ids):
        group = groups[subject_id]
        group_counts = Counter(record.label for record in group)
        if index < len(SPLITS):
            chosen = SPLITS[index]
            assignments[chosen].extend(group)
            counts[chosen].update(group_counts)
            continue
        candidates = []
        for split in SPLITS:
            new_size = len(assignments[split]) + len(group)
            size_ratio = new_size / max(targets[split], 1)
            class_error = sum(
                abs(counts[split][label] + group_counts[label] - target_classes[split][label])
                / max(class_totals[label], 1)
                for label in class_totals
            )
            candidates.append((size_ratio + 0.1 * class_error, len(assignments[split]), split))
        _, _, chosen = min(candidates)
        assignments[chosen].extend(group)
        counts[chosen].update(group_counts)

    return {split: sorted(items, key=lambda item: item.relative_path) for split, items in assignments.items()}


def write_manifests(splits: dict[str, list[ParsedImage]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, records in splits.items():
        with (output_dir / f"{split}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["image_path", "label", "subject_id"])
            writer.writeheader()
            writer.writerows(
                {"image_path": item.relative_path, "label": item.label, "subject_id": item.subject_id}
                for item in records
            )


def prepare(mrl_root: Path, output_dir: Path, report_path: Path, seed: int) -> dict:
    records, issues = inspect_images(mrl_root)
    splits = split_by_subject(records, seed)
    write_manifests(splits, output_dir)
    report = {
        "dataset_root": str(mrl_root),
        "seed": seed,
        "filename_convention": "sXXXX_YYYYY_A_B_C_D_E_FF.png; D=0 closed, D=1 open",
        "subject_grouping": {"enabled": True, "subject_count": len({item.subject_id for item in records})},
        "filename_label_counts": dict(sorted(Counter(item.label for item in records).items())),
        "verified_readable_counts": dict(
            sorted(Counter(item.label for split in splits.values() for item in split).items())
        ),
        "issues": issues,
        "issue_counts": dict(Counter(issue["issue"].split(":", 1)[0] for issue in issues)),
        "splits": {
            split: {
                "image_count": len(items),
                "subject_count": len({item.subject_id for item in items}),
                "class_counts": dict(sorted(Counter(item.label for item in items).items())),
            }
            for split, items in splits.items()
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mrl-root", type=Path, default=Path("data/raw/mrl-eye/mrleyedataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/mrl-eye"))
    parser.add_argument("--report", type=Path, default=Path("reports/mrl_dataset_preparation.json"))
    parser.add_argument("--seed", type=int, default=20261024)
    args = parser.parse_args()
    if not args.mrl_root.is_dir():
        print(f"MRL dataset directory does not exist: {args.mrl_root}", file=sys.stderr)
        return 2
    report = prepare(args.mrl_root, args.output_dir, args.report, args.seed)
    print(json.dumps({"verified_readable_counts": report["verified_readable_counts"], "splits": report["splits"], "issues": report["issue_counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FaceWell AI - Dataset Download Script
======================================
Downloads and verifies two datasets:
  1. MRL Eye Dataset (mrl.cs.vsb.cz)         -> D:/Facewell/data/raw/mrl-eye
  2. CNN-Based Drowsiness Detection (HF)      -> D:/Facewell/data/raw/cnn-drowsiness

Usage:
    python scripts/download_datasets.py [--dataset mrl|hf|both]

Re-runnable: skips already-completed work automatically.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

# ---- Force UTF-8 console output on Windows (avoids cp1252 UnicodeEncodeError) ----
import io as _io
sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = _io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


# ---- Dependency bootstrap ----

def _ensure_package(pkg_import: str, pip_name: str) -> None:
    """Install a package if it is not already available."""
    try:
        __import__(pkg_import)
    except ImportError:
        print(f"[SETUP] Installing missing package: {pip_name} ...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", pip_name],
            stdout=subprocess.DEVNULL,
        )


_ensure_package("requests", "requests")
_ensure_package("tqdm", "tqdm")
_ensure_package("PIL", "Pillow")
_ensure_package("huggingface_hub", "huggingface_hub")

import requests  # noqa: E402
from tqdm import tqdm  # noqa: E402

# ---- Paths ----

ROOT       = Path(__file__).resolve().parents[1]   # D:/Facewell
RAW        = ROOT / "data" / "raw"
DOWNLOADS  = RAW / "downloads"
MRL_DIR    = RAW / "mrl-eye"
HF_DIR     = RAW / "cnn-drowsiness"
REPORT_DIR = ROOT / "reports"

for _d in (DOWNLOADS, MRL_DIR, HF_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---- Constants ----

MRL_URL     = "http://mrl.cs.vsb.cz/data/eyedataset/mrlEyes_2018_01.zip"
MRL_ARCHIVE = DOWNLOADS / "mrlEyes_2018_01.zip"
MRL_SOURCE  = "http://mrl.cs.vsb.cz/eyedataset"
MRL_LICENCE = (
    "Free for academic / non-commercial research. "
    "Citation required: Fusek et al., VSB-TUO Media Research Lab."
)

HF_REPO_ID  = "notgoodkeeper/cnn-based-drowsiness-detection-data"
HF_SOURCE   = "https://huggingface.co/datasets/notgoodkeeper/cnn-based-drowsiness-detection-data"
HF_LICENCE  = "MIT (https://choosealicense.com/licenses/mit/)"

CHUNK_SIZE   = 1 << 20   # 1 MB
MAX_RETRIES  = 5
TIMEOUT      = 60         # seconds
BACKOFF_BASE = 4          # seconds


# ================================================================================
# UTILITIES
# ================================================================================

def _banner(msg: str) -> None:
    width = 72
    print("\n" + "=" * width)
    print(f"  {msg}")
    print("=" * width)


def _log(level: str, msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{level}] {msg}")


def _human_size(nbytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} PB"


def _count_images(folder: Path, exts=(".png", ".jpg", ".jpeg", ".bmp", ".pgm")) -> int:
    return sum(1 for f in folder.rglob("*") if f.suffix.lower() in exts)


def _image_extensions(folder: Path) -> dict:
    counts: dict = {}
    for f in folder.rglob("*"):
        if f.is_file():
            ext = f.suffix.lower()
            if ext in (".png", ".jpg", ".jpeg", ".bmp", ".pgm", ".gif", ".tiff"):
                counts[ext] = counts.get(ext, 0) + 1
    return counts


def _sample_image_dims(folder: Path, max_samples: int = 20) -> list:
    """Return [(width, height), ...] for up to max_samples images."""
    try:
        from PIL import Image
    except ImportError:
        return []
    dims = []
    for f in folder.rglob("*"):
        if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".pgm"):
            try:
                with Image.open(f) as img:
                    dims.append(img.size)
            except Exception:
                pass
            if len(dims) >= max_samples:
                break
    return dims


# ================================================================================
# DOWNLOAD WITH RETRY + RESUME
# ================================================================================

def download_file(url: str, dest: Path, desc: str = "") -> None:
    """Download url -> dest with retry, resume (Range header), and progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers: dict = {}

    start_byte = dest.stat().st_size if dest.exists() else 0
    if start_byte:
        headers["Range"] = f"bytes={start_byte}-"
        _log("INFO", f"Resuming from byte {start_byte:,} ({_human_size(start_byte)})")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers=headers, stream=True, timeout=TIMEOUT)

            if resp.status_code == 416:
                _log("INFO", "Server: range not satisfiable - file may already be complete.")
                return

            resp.raise_for_status()

            total = int(resp.headers.get("Content-Length", 0))
            if resp.status_code == 206:
                total += start_byte

            mode = "ab" if start_byte else "wb"
            label = desc or dest.name

            with open(dest, mode) as fh, tqdm(
                total=total,
                initial=start_byte,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=label[:50],
                dynamic_ncols=True,
            ) as bar:
                for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        fh.write(chunk)
                        bar.update(len(chunk))
            return  # success

        except (requests.RequestException, IOError) as exc:
            wait = BACKOFF_BASE * (2 ** (attempt - 1))
            _log("WARN", f"Attempt {attempt}/{MAX_RETRIES} failed: {exc}")
            if attempt < MAX_RETRIES:
                _log("INFO", f"Retrying in {wait}s ...")
                time.sleep(wait)
                if dest.exists():
                    start_byte = dest.stat().st_size
                    headers["Range"] = f"bytes={start_byte}-"
            else:
                raise RuntimeError(
                    f"Download failed after {MAX_RETRIES} attempts: {url}"
                ) from exc


# ================================================================================
# DATASET 1 - MRL EYE DATASET
# ================================================================================

def _mrl_class_dist(result: dict) -> None:
    """Parse MRL filenames for open/closed eye class distribution.
    MRL filename: sXXXX_YYYYY_A_B_C_D_E_FF.png  -- parts[4] = eye state (0=closed, 1=open)
    """
    open_c = closed_c = unknown_c = 0
    for img in MRL_DIR.rglob("*.png"):
        parts = img.stem.split("_")
        if len(parts) >= 6:
            state = parts[4]
            if state == "1":
                open_c += 1
            elif state == "0":
                closed_c += 1
            else:
                unknown_c += 1
        else:
            unknown_c += 1
    result["class_distribution"] = {
        "open": open_c,
        "closed": closed_c,
        "unknown_or_unparseable": unknown_c,
    }
    _log("INFO", f"Eye state - Open: {open_c:,}  Closed: {closed_c:,}  Unknown: {unknown_c:,}")


def download_mrl() -> dict:
    """
    Download and extract the MRL Eye Dataset.
    Source 1:  http://mrl.cs.vsb.cz/data/eyedataset/mrlEyes_2018_01.zip  (official)
    Source 2:  Kaggle CLI - kaggle datasets download imadeddinedjerarda/mrl-eye-dataset
    Source 3:  Manual browser download instructions printed if both fail.
    """
    _banner("DATASET 1 - MRL Eye Dataset")

    result = {
        "name": "MRL Eye Dataset",
        "source_url": MRL_SOURCE,
        "download_url": MRL_URL,
        "licence": MRL_LICENCE,
        "archive_path": str(MRL_ARCHIVE),
        "extract_dir": str(MRL_DIR),
        "archive_size_bytes": None,
        "archive_size_human": None,
        "extracted": False,
        "image_count": 0,
        "image_extensions": {},
        "sample_dims": [],
        "class_distribution": {},
        "errors": [],
        "status": "INCOMPLETE",
    }

    # Fast-path: already extracted
    existing_imgs = _count_images(MRL_DIR)
    if existing_imgs >= 1000:
        _log("INFO", f"Already extracted: {existing_imgs:,} images found. Skipping download.")
        result["extracted"] = True
        result["image_count"] = existing_imgs
        result["image_extensions"] = _image_extensions(MRL_DIR)
        result["sample_dims"] = _sample_image_dims(MRL_DIR)
        if MRL_ARCHIVE.exists():
            sz = MRL_ARCHIVE.stat().st_size
            result["archive_size_bytes"] = sz
            result["archive_size_human"] = _human_size(sz)
        _mrl_class_dist(result)
        result["status"] = "COMPLETE"
        return result

    # Fast-path: archive already on disk (but not yet extracted)
    archive_ready = MRL_ARCHIVE.exists() and MRL_ARCHIVE.stat().st_size > 10_000_000
    if archive_ready:
        _log("INFO", f"Archive already on disk ({_human_size(MRL_ARCHIVE.stat().st_size)}), skipping download.")
    else:
        # Source 1: Official VSB server
        _log("INFO", f"Trying official server: {MRL_URL}")
        official_ok = False
        try:
            download_file(MRL_URL, MRL_ARCHIVE, "MRL Eye Dataset (official)")
            official_ok = True
        except RuntimeError as exc:
            _log("WARN", f"Official server failed: {exc}")
            result["errors"].append(f"Official server failed: {exc}")

        # Source 2: Kaggle CLI fallback
        if not official_ok:
            _log("INFO", "Trying Kaggle CLI fallback ...")
            try:
                subprocess.run(
                    ["pip", "install", "--quiet", "kaggle"],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                proc = subprocess.run(
                    [
                        sys.executable, "-m", "kaggle",
                        "datasets", "download",
                        "-d", "imadeddinedjerarda/mrl-eye-dataset",
                        "-p", str(DOWNLOADS),
                        "--unzip",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=600,
                )
                if proc.returncode == 0:
                    _log("INFO", "Kaggle download succeeded - moving images to mrl-eye/ ...")
                    for ext in (".png", ".jpg", ".jpeg", ".bmp", ".pgm"):
                        for f in DOWNLOADS.rglob(f"*{ext}"):
                            try:
                                rel = f.relative_to(DOWNLOADS)
                                dest = MRL_DIR / rel
                                dest.parent.mkdir(parents=True, exist_ok=True)
                                if not dest.exists():
                                    shutil.copy2(f, dest)
                            except Exception:
                                pass
                    result["extracted"] = True
                    img_count = _count_images(MRL_DIR)
                    result["image_count"] = img_count
                    result["image_extensions"] = _image_extensions(MRL_DIR)
                    result["sample_dims"] = _sample_image_dims(MRL_DIR)
                    _mrl_class_dist(result)
                    if img_count >= 1000:
                        result["status"] = "COMPLETE"
                    return result
                else:
                    raise RuntimeError(f"kaggle CLI error: {proc.stderr.strip()}")
            except Exception as kexc:
                _log("WARN", f"Kaggle fallback failed: {kexc}")
                result["errors"].append(f"Kaggle fallback failed: {kexc}")

            # Source 3: Manual download instructions
            manual_msg = (
                "MANUAL ACTION REQUIRED - both automatic sources failed.\n"
                "\n"
                "The VSB university server (mrl.cs.vsb.cz) is unreachable and the Kaggle\n"
                "CLI is not configured. To download manually:\n"
                "\n"
                "OPTION A - Kaggle browser download (recommended, free account):\n"
                "  1. Go to: https://www.kaggle.com/datasets/imadeddinedjerarda/mrl-eye-dataset\n"
                "  2. Click 'Download' (login required)\n"
                "  3. Save the ZIP as: D:\\Facewell\\data\\raw\\downloads\\mrlEyes_2018_01.zip\n"
                "  4. Re-run: python scripts/download_datasets.py --dataset mrl\n"
                "     (the script will detect the archive and extract automatically)\n"
                "\n"
                "OPTION B - Kaggle API key:\n"
                "  1. kaggle.com -> Settings -> Account -> Create New Token\n"
                "  2. Save kaggle.json to C:\\Users\\<YOU>\\.kaggle\\kaggle.json\n"
                "  3. pip install kaggle\n"
                "  4. Re-run: python scripts/download_datasets.py --dataset mrl\n"
                "\n"
                "OPTION C - Official URL (when VSB server is back online):\n"
                "  http://mrl.cs.vsb.cz/data/eyedataset/mrlEyes_2018_01.zip"
            )
            _log("ERROR", manual_msg)
            result["errors"].append(manual_msg)
            result["status"] = "INCOMPLETE - MANUAL DOWNLOAD REQUIRED"
            return result

    # Integrity check
    sz = MRL_ARCHIVE.stat().st_size
    result["archive_size_bytes"] = sz
    result["archive_size_human"] = _human_size(sz)
    _log("INFO", f"Archive size on disk: {_human_size(sz)}")

    if not zipfile.is_zipfile(MRL_ARCHIVE):
        msg = f"Archive is not a valid ZIP file: {MRL_ARCHIVE}"
        result["errors"].append(msg)
        _log("ERROR", msg)
        return result
    _log("INFO", "ZIP integrity check passed [OK]")

    # Extract (skip existing files)
    _log("INFO", f"Extracting to {MRL_DIR} ...")
    with zipfile.ZipFile(MRL_ARCHIVE, "r") as zf:
        members = zf.infolist()
        _log("INFO", f"ZIP contains {len(members):,} entries.")
        for member in tqdm(members, desc="Extracting", unit="file", dynamic_ncols=True):
            target = MRL_DIR / member.filename
            if target.exists() and not member.is_dir():
                continue
            zf.extract(member, MRL_DIR)

    result["extracted"] = True
    _log("INFO", "Extraction complete [OK]")

    # Verify
    img_count = _count_images(MRL_DIR)
    result["image_count"] = img_count
    result["image_extensions"] = _image_extensions(MRL_DIR)
    result["sample_dims"] = _sample_image_dims(MRL_DIR)
    _log("INFO", f"Images found: {img_count:,}")
    _mrl_class_dist(result)

    if img_count >= 1000:
        result["status"] = "COMPLETE"
    else:
        result["errors"].append(
            f"Unexpectedly low image count ({img_count}); expected ~84,898."
        )

    return result


# ================================================================================
# DATASET 2 - CNN DROWSINESS (HUGGING FACE)
# ================================================================================

def download_hf() -> dict:
    """
    Download the HuggingFace CNN-Drowsiness dataset via snapshot_download.
    Uses LFS-aware / Xet-aware download. No authentication required (MIT dataset).
    """
    _banner("DATASET 2 - CNN-Based Drowsiness Detection (HuggingFace)")

    result = {
        "name": "CNN-Based Drowsiness Detection Dataset",
        "source_url": HF_SOURCE,
        "repo_id": HF_REPO_ID,
        "licence": HF_LICENCE,
        "local_dir": str(HF_DIR),
        "image_count": 0,
        "image_extensions": {},
        "sample_dims": [],
        "csv_files": [],
        "parquet_files": [],
        "class_distribution": {},
        "splits": {},
        "errors": [],
        "status": "INCOMPLETE",
    }

    # Fast-path: already downloaded
    existing_imgs = _count_images(HF_DIR)
    if existing_imgs >= 100:
        _log("INFO", f"Already downloaded: {existing_imgs:,} images found. Verifying ...")
    else:
        _log("INFO", f"Repo ID  : {HF_REPO_ID}")
        _log("INFO", f"Local dir: {HF_DIR}")
        _log("INFO", "Calling huggingface_hub.snapshot_download ... (may take several minutes)")
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=HF_REPO_ID,
                repo_type="dataset",
                local_dir=str(HF_DIR),
                ignore_patterns=["*.git*"],
            )
            _log("INFO", "snapshot_download completed [OK]")
        except Exception as exc:
            msg = (
                f"snapshot_download failed: {exc}\n"
                "Possible causes: network restriction, rate limiting, or missing huggingface_hub.\n"
                "Fix: pip install -U huggingface_hub  then re-run."
            )
            result["errors"].append(msg)
            _log("ERROR", msg)
            return result

    # Verify images
    img_count = _count_images(HF_DIR)
    result["image_count"] = img_count
    result["image_extensions"] = _image_extensions(HF_DIR)
    result["sample_dims"] = _sample_image_dims(HF_DIR)
    _log("INFO", f"Images found: {img_count:,}")

    if img_count == 0:
        result["errors"].append(
            "No image files found after snapshot_download. "
            "The repo may use Git LFS pointers not resolved by snapshot_download. "
            "Ensure git-lfs is installed: https://git-lfs.com"
        )

    # Inventory metadata files
    csv_files = sorted(HF_DIR.rglob("*.csv"))
    parquet_files = sorted(HF_DIR.rglob("*.parquet"))
    result["csv_files"] = [str(f.relative_to(HF_DIR)) for f in csv_files]
    result["parquet_files"] = [str(f.relative_to(HF_DIR)) for f in parquet_files]
    _log("INFO", f"CSV files    : {len(csv_files)}")
    _log("INFO", f"Parquet files: {len(parquet_files)}")

    # Class distribution from CSV
    class_dist: dict = {}
    split_counts: dict = {}

    if csv_files:
        try:
            import csv
            for csvf in csv_files[:3]:
                with open(csvf, newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        cls = row.get("drowsiness_class") or row.get("class") or "unknown"
                        class_dist[cls] = class_dist.get(cls, 0) + 1
            _log("INFO", f"Class dist from CSV: {class_dist}")
        except Exception as exc:
            _log("WARN", f"Could not parse CSV for class dist: {exc}")

    # Class distribution from Parquet (if CSV empty)
    if not class_dist and parquet_files:
        try:
            import importlib.util
            if importlib.util.find_spec("pandas") is not None:
                import pandas as pd
                for pf in parquet_files[:6]:
                    df = pd.read_parquet(pf)
                    col = (
                        "drowsiness_class" if "drowsiness_class" in df.columns
                        else "class" if "class" in df.columns
                        else None
                    )
                    split_col = "split" if "split" in df.columns else None
                    if col:
                        for k, v in df[col].value_counts().to_dict().items():
                            class_dist[str(k)] = class_dist.get(str(k), 0) + int(v)
                    if split_col:
                        for k, v in df[split_col].value_counts().to_dict().items():
                            split_counts[str(k)] = split_counts.get(str(k), 0) + int(v)
                _log("INFO", f"Class dist from Parquet: {class_dist}")
            else:
                _log("WARN", "pandas not installed; skipping Parquet class count. Run: pip install pandas pyarrow")
        except Exception as exc:
            _log("WARN", f"Could not parse Parquet for class dist: {exc}")

    result["class_distribution"] = class_dist
    result["splits"] = split_counts

    if img_count > 0 and (csv_files or parquet_files):
        result["status"] = "COMPLETE"
    elif img_count > 0:
        result["status"] = "PARTIAL - images present but no metadata files found"
    else:
        result["errors"].append("Dataset download incomplete: no images found on disk.")

    return result


# ================================================================================
# REPORT GENERATION
# ================================================================================

def _fmt_dist(d: dict) -> str:
    if not d:
        return "_No data_"
    lines = ["| Class | Count |", "| --- | --- |"]
    for k, v in sorted(d.items(), key=lambda x: str(x[0])):
        lines.append(f"| {k} | {v:,} |")
    return "\n".join(lines)


def _fmt_dims(dims: list) -> str:
    if not dims:
        return "_Could not sample_"
    unique = sorted(set(dims))
    return ", ".join(f"{w}x{h}" for w, h in unique[:10])


def generate_report(mrl: dict | None, hf: dict | None) -> Path:
    """Write reports/dataset_download_report.md"""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# FaceWell AI - Dataset Download Report",
        "",
        f"**Generated:** {ts}",
        "",
        "---",
        "",
    ]

    def section(ds: dict, idx: int) -> list:
        s = [
            f"## Dataset {idx}: {ds['name']}",
            "",
            f"**Status:** `{ds['status']}`",
            "",
            "### Source & Licence",
            "",
            "| Field | Value |",
            "| --- | --- |",
            f"| Official Page | {ds.get('source_url', 'N/A')} |",
        ]
        if "download_url" in ds:
            s.append(f"| Download URL | {ds['download_url']} |")
        if "repo_id" in ds:
            s.append(f"| HuggingFace Repo | {ds['repo_id']} |")
        s += [
            f"| Licence | {ds.get('licence', 'N/A')} |",
            "",
        ]

        s += ["### Downloaded Files", ""]
        if "archive_path" in ds and ds.get("archive_size_bytes") is not None:
            s += [
                "| File | Size |",
                "| --- | --- |",
                f"| `{Path(ds['archive_path']).name}` | {ds['archive_size_human']} ({ds['archive_size_bytes']:,} bytes) |",
                "",
            ]
        else:
            s.append("_No archive downloaded or size unavailable._\n")

        s += ["### Extraction / Download Status", ""]
        if ds.get("extracted"):
            s.append(f"[OK] Extracted/downloaded to `{ds.get('extract_dir', ds.get('local_dir', 'N/A'))}`\n")
        else:
            s.append("[FAIL] Not completed.\n")

        s += ["### Image Statistics", ""]
        img_count = ds.get("image_count", 0)
        exts = ds.get("image_extensions", {})
        ext_str = ", ".join(f"`{k}` ({v:,})" for k, v in exts.items()) or "_None_"
        s += [
            f"- **Total images on disk:** {img_count:,}",
            f"- **Extensions:** {ext_str}",
            f"- **Sample dimensions (WxH):** {_fmt_dims(ds.get('sample_dims', []))}",
            "",
        ]

        if ds.get("csv_files") is not None:
            s += ["### Metadata Files", ""]
            csv_list = ds.get("csv_files", [])
            pq_list = ds.get("parquet_files", [])
            if csv_list:
                s.append("**CSV files:**")
                for f in csv_list:
                    s.append(f"- `{f}`")
            if pq_list:
                s.append("**Parquet files:**")
                for f in pq_list[:20]:
                    s.append(f"- `{f}`")
                if len(pq_list) > 20:
                    s.append(f"- ... and {len(pq_list) - 20} more")
            if not csv_list and not pq_list:
                s.append("_No CSV or Parquet metadata found._")
            s.append("")

        s += ["### Class-Label Distribution", ""]
        s.append(_fmt_dist(ds.get("class_distribution", {})))
        s.append("")

        if ds.get("splits"):
            s += ["### Split Distribution", ""]
            s.append(_fmt_dist(ds["splits"]))
            s.append("")

        if ds.get("errors"):
            s += ["### Errors / Warnings", ""]
            for e in ds["errors"]:
                s.append(f"- [WARN] {e}")
            s.append("")

        s += ["### Access / Licensing Notes", ""]
        if "MRL" in ds["name"]:
            s.append(
                "- MRL Eye Dataset: Free for **non-commercial academic research**. "
                "Cite VSB-TUO Media Research Lab (Fusek et al.) in any publication."
            )
        else:
            s.append("- HuggingFace dataset: **MIT licence** - permissive open-source use.")
        s += ["", "---", ""]
        return s

    if mrl:
        lines += section(mrl, 1)
    if hf:
        lines += section(hf, 2)

    lines += [
        "## Git / Data Hygiene",
        "",
        "Raw dataset directories are listed in `.gitignore` and will **not** be committed to version control.",
        "",
        "---",
        "",
        "_Report generated by `scripts/download_datasets.py`_",
    ]

    report_path = REPORT_DIR / "dataset_download_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    _log("INFO", f"Report written -> {report_path}")
    return report_path


# ================================================================================
# GITIGNORE
# ================================================================================

def update_gitignore() -> None:
    gitignore = ROOT / ".gitignore"
    rules = [
        "# FaceWell raw datasets (do not commit)",
        "data/raw/downloads/",
        "data/raw/mrl-eye/",
        "data/raw/cnn-drowsiness/",
        "data/raw/",
    ]
    existing = gitignore.read_text(encoding="utf-8") if gitignore.exists() else ""
    new_rules = [r for r in rules if r not in existing]
    if new_rules:
        with open(gitignore, "a", encoding="utf-8") as f:
            f.write("\n" + "\n".join(new_rules) + "\n")
        _log("INFO", f".gitignore updated with {len(new_rules)} new rules.")
    else:
        _log("INFO", ".gitignore already contains dataset exclusion rules.")


# ================================================================================
# REQUIREMENTS FILE
# ================================================================================

def update_requirements() -> None:
    req_file = ROOT / "requirements.txt"
    needed = {"requests", "tqdm", "Pillow", "huggingface_hub"}
    existing_lines = (
        req_file.read_text(encoding="utf-8").splitlines() if req_file.exists() else []
    )
    existing_pkgs = {
        ln.split("==")[0].split(">=")[0].strip().lower()
        for ln in existing_lines if ln.strip()
    }
    to_add = [p for p in sorted(needed) if p.lower() not in existing_pkgs]
    if to_add:
        with open(req_file, "a", encoding="utf-8") as f:
            f.write("\n".join([""] + to_add + [""]))
        _log("INFO", f"requirements.txt updated: added {to_add}")
    else:
        _log("INFO", "requirements.txt already contains required packages.")


# ================================================================================
# MAIN
# ================================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description="FaceWell AI - download and verify datasets")
    parser.add_argument(
        "--dataset",
        choices=["mrl", "hf", "both"],
        default="both",
        help="Which dataset to download (default: both)",
    )
    args = parser.parse_args()

    update_gitignore()
    update_requirements()

    mrl_result: dict | None = None
    hf_result: dict | None = None

    if args.dataset in ("mrl", "both"):
        mrl_result = download_mrl()

    if args.dataset in ("hf", "both"):
        hf_result = download_hf()

    report_path = generate_report(mrl_result, hf_result)

    _banner("SUMMARY")
    all_ok = True
    for label, res in [("MRL Eye Dataset", mrl_result), ("CNN Drowsiness (HF)", hf_result)]:
        if res is None:
            continue
        status = res.get("status", "UNKNOWN")
        icon = "[OK]  " if status == "COMPLETE" else "[FAIL]"
        print(f"  {icon}  {label}: {status}")
        if res.get("errors"):
            for e in res["errors"][:3]:
                first_line = e.split("\n")[0]
                print(f"         [WARN] {first_line}")
        if status != "COMPLETE":
            all_ok = False

    print(f"\n  Report: {report_path}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

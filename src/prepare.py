"""Resize the Fashion Product Images dataset and build the product catalog.

Reads data/fashion-dataset/styles.csv, keeps only products whose image exists on
disk, resizes each image so its short side is 224 px (aspect preserved), writes
JPEG quality 90 into data/images_224/, and writes data/catalog.parquet.

This script never deletes anything. Existing resized images are skipped unless
--overwrite is passed, and a --limit run writes its catalog to a separate
data/catalog.limit{N}.parquet so it cannot truncate the full catalog.

Usage:
    python -m src.prepare
    python -m src.prepare --limit 200
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import sys
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

# --- Paths (verified against the extracted Kaggle archive) -------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "fashion-dataset"
IMAGES_SRC_DIR = RAW_DIR / "images"
STYLES_CSV = RAW_DIR / "styles.csv"
IMAGES_OUT_DIR = DATA_DIR / "images_224"
CATALOG_PATH = DATA_DIR / "catalog.parquet"
LIMITED_CATALOG_TEMPLATE = "catalog.limit{limit}.parquet"

# --- Image processing parameters --------------------------------------------
SHORT_SIDE = 224
JPEG_QUALITY = 90
RESAMPLE = Image.Resampling.LANCZOS

# --- Catalog schema ---------------------------------------------------------
CATALOG_COLUMNS = [
    "id",
    "articleType",
    "subCategory",
    "baseColour",
    "gender",
    "season",
    "usage",
    "productDisplayName",
]

CHUNKSIZE = 64


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------
def load_styles(styles_csv: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    """Parse styles.csv, repairing rows with extra commas in the last column.

    The dataset's known defect is unquoted commas inside productDisplayName,
    which makes those rows produce more fields than the header. Because
    productDisplayName is the final column, the surplus fields can be rejoined
    losslessly rather than discarded. Rows with too *few* fields, an
    unparseable id, or a duplicate id are dropped.

    Returns the parsed frame plus a dict of counts for reporting.
    """
    stats = {
        "rows_read": 0,
        "repaired_extra_fields": 0,
        "dropped_too_few_fields": 0,
        "dropped_bad_id": 0,
        "dropped_duplicate_id": 0,
    }

    with styles_csv.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            raise SystemExit(f"{styles_csv} is empty")

        n_fields = len(header)
        tail_index = n_fields - 1  # productDisplayName
        records: list[dict[str, str]] = []
        seen_ids: set[int] = set()

        for row in reader:
            stats["rows_read"] += 1

            if len(row) > n_fields:
                # Rejoin with the original delimiter; the leading space seen in
                # e.g. "Boss Men Perfume, After Shave Balm ..." lives inside
                # the field itself, so a bare comma reconstructs the text.
                row = row[:tail_index] + [",".join(row[tail_index:])]
                stats["repaired_extra_fields"] += 1
            elif len(row) < n_fields:
                stats["dropped_too_few_fields"] += 1
                continue

            try:
                product_id = int(row[0])
            except ValueError:
                stats["dropped_bad_id"] += 1
                continue

            if product_id in seen_ids:
                stats["dropped_duplicate_id"] += 1
                continue
            seen_ids.add(product_id)

            record = dict(zip(header, row))
            record["id"] = product_id
            records.append(record)

    df = pd.DataFrame.from_records(records)
    missing = [c for c in CATALOG_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(f"{styles_csv} is missing expected columns: {missing}")

    df = df[CATALOG_COLUMNS].copy()
    df["id"] = df["id"].astype("int64")
    for column in CATALOG_COLUMNS[1:]:
        # Empty strings in the raw CSV mean "unknown"; keep them as NA rather
        # than inventing a value.
        df[column] = df[column].astype("string").str.strip().replace("", pd.NA)

    return df, stats


# ---------------------------------------------------------------------------
# Image resizing (runs in worker processes)
# ---------------------------------------------------------------------------
def _target_size(width: int, height: int) -> tuple[int, int]:
    """Scale so the short side is exactly SHORT_SIDE, preserving aspect ratio."""
    if width <= height:
        return SHORT_SIDE, max(SHORT_SIDE, round(height * SHORT_SIDE / width))
    return max(SHORT_SIDE, round(width * SHORT_SIDE / height)), SHORT_SIDE


def resize_one(task: tuple[int, str, str, bool]) -> tuple[int, str, str]:
    """Resize a single image. Returns (id, status, detail).

    status is one of "ok", "skipped", "failed".
    """
    product_id, src, dst, overwrite = task
    dst_path = Path(dst)

    if not overwrite and dst_path.exists():
        return product_id, "skipped", ""

    try:
        with Image.open(src) as img:
            # Fast DCT-domain downscale for JPEG sources; draft() only ever
            # picks a scale that leaves the image at least this large.
            img.draft("RGB", (SHORT_SIDE, SHORT_SIDE))
            img = img.convert("RGB")
            img = img.resize(_target_size(img.width, img.height), RESAMPLE)
            img.save(dst_path, format="JPEG", quality=JPEG_QUALITY)
    except Exception as exc:  # unreadable/truncated files must not kill the run
        return product_id, "failed", f"{type(exc).__name__}: {exc}"

    return product_id, "ok", ""


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "process only the first N products (smoke test); the catalog goes "
            "to data/catalog.limit{N}.parquet, leaving catalog.parquet alone"
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="worker processes for resizing (default: %(default)s)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="re-encode images that already exist in data/images_224/",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="parse the CSV and report counts without writing any files",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    for path in (STYLES_CSV, IMAGES_SRC_DIR):
        if not path.exists():
            raise SystemExit(f"missing input: {path}")

    # A limited run is a smoke test, so it must not truncate the real catalog.
    catalog_path = (
        CATALOG_PATH
        if args.limit is None
        else DATA_DIR / LIMITED_CATALOG_TEMPLATE.format(limit=args.limit)
    )

    df, stats = load_styles(STYLES_CSV)

    # Keep only products whose source image is on disk.
    src_paths = df["id"].map(lambda i: IMAGES_SRC_DIR / f"{i}.jpg")
    exists = src_paths.map(Path.exists)
    dropped_missing_image = int((~exists).sum())
    df = df[exists].reset_index(drop=True)
    src_paths = src_paths[exists].reset_index(drop=True)

    if args.limit is not None:
        df = df.head(args.limit).reset_index(drop=True)
        src_paths = src_paths.head(args.limit).reset_index(drop=True)

    print(f"styles.csv rows read:            {stats['rows_read']}")
    print(f"  repaired (extra commas):       {stats['repaired_extra_fields']}")
    print(f"  dropped (too few fields):      {stats['dropped_too_few_fields']}")
    print(f"  dropped (unparseable id):      {stats['dropped_bad_id']}")
    print(f"  dropped (duplicate id):        {stats['dropped_duplicate_id']}")
    print(f"  dropped (image not on disk):   {dropped_missing_image}")
    total_dropped = (
        stats["dropped_too_few_fields"]
        + stats["dropped_bad_id"]
        + stats["dropped_duplicate_id"]
        + dropped_missing_image
    )
    print(f"  total dropped:                 {total_dropped}")
    print(f"products to process:             {len(df)}")
    print(f"catalog target:                  {catalog_path}")

    if args.dry_run:
        print("\n--dry-run: no images or catalog written")
        return 0

    IMAGES_OUT_DIR.mkdir(parents=True, exist_ok=True)

    tasks = [
        (int(pid), str(src), str(IMAGES_OUT_DIR / f"{pid}.jpg"), args.overwrite)
        for pid, src in zip(df["id"], src_paths)
    ]

    counts = {"ok": 0, "skipped": 0, "failed": 0}
    failures: list[tuple[int, str]] = []
    failed_ids: set[int] = set()

    print(
        f"\nresizing to short side {SHORT_SIDE}px, JPEG quality {JPEG_QUALITY}, "
        f"{args.workers} workers -> {IMAGES_OUT_DIR}"
    )
    with mp.Pool(processes=args.workers) as pool:
        for product_id, status, detail in tqdm(
            pool.imap_unordered(resize_one, tasks, chunksize=CHUNKSIZE),
            total=len(tasks),
            unit="img",
        ):
            counts[status] += 1
            if status == "failed":
                failed_ids.add(product_id)
                if len(failures) < 20:
                    failures.append((product_id, detail))

    print(f"images written:                  {counts['ok']}")
    print(f"images already present (skipped): {counts['skipped']}")
    print(f"images failed:                   {counts['failed']}")
    for product_id, detail in failures:
        print(f"  {product_id}: {detail}", file=sys.stderr)
    if counts["failed"] > len(failures):
        print(f"  ... {counts['failed'] - len(failures)} more", file=sys.stderr)

    # Products whose image could not be encoded do not belong in the catalog.
    if failed_ids:
        df = df[~df["id"].isin(failed_ids)].reset_index(drop=True)

    df.to_parquet(catalog_path, index=False)

    print(f"\ncatalog rows:                    {len(df)}")
    print(f"catalog written:                 {catalog_path}")
    print(f"columns:                         {list(df.columns)}")
    na_counts = df.isna().sum()
    na_counts = na_counts[na_counts > 0]
    if len(na_counts):
        print("missing values (kept as NA, not imputed):")
        for column, count in na_counts.items():
            print(f"  {column}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

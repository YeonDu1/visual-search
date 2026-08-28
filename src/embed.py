"""Encode the catalog's images with CLIP into artifacts/embeddings.npy.

Reads data/catalog.parquet, loads data/images_224/{id}.jpg for every row, runs
openai/clip-vit-base-patch32's vision tower under torch.autocast fp16, and
writes L2-normalized float32 embeddings plus the matching product ids.

Row i of embeddings.npy corresponds to ids.json[i]; both follow catalog order.

A --limit run is a smoke test and writes to artifacts/embeddings.limit{N}.npy
and artifacts/ids.limit{N}.json so it cannot clobber the full artifacts.

Usage:
    python -m src.embed
    python -m src.embed --limit 512
    python -m src.embed --batch-size 128
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm
from transformers import CLIPModel

# --- Paths ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
IMAGES_DIR = DATA_DIR / "images_224"
CATALOG_PATH = DATA_DIR / "catalog.parquet"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
EMBEDDINGS_PATH = ARTIFACTS_DIR / "embeddings.npy"
IDS_PATH = ARTIFACTS_DIR / "ids.json"
LIMITED_EMBEDDINGS_TEMPLATE = "embeddings.limit{limit}.npy"
LIMITED_IDS_TEMPLATE = "ids.limit{limit}.json"

# --- Model / encoding parameters --------------------------------------------
MODEL_ID = "openai/clip-vit-base-patch32"
EMBED_DIM = 512
BATCH_SIZE = 256
NUM_WORKERS = 4
MIN_BATCH_SIZE = 1  # OOM-halving floor; below this the error is real

# Mirrors CLIPImageProcessor's config for this checkpoint. Kept explicit so the
# transform can run inside DataLoader workers instead of on the main process.
RESIZE_SHORT_SIDE = 224
CROP_SIZE = 224
IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)


# ---------------------------------------------------------------------------
# Dataset (runs in worker processes)
# ---------------------------------------------------------------------------
class CenterCropFloor:
    """Center crop using HF's offset convention: (size - crop) // 2.

    torchvision's CenterCrop rounds instead of flooring, so whenever
    size - crop is odd the two disagree by one pixel. Most images here are
    224x299 (difference 75, odd), so using CenterCrop would shift nearly every
    image one pixel off what CLIPImageProcessor feeds the model.
    """

    def __init__(self, size: int) -> None:
        self.size = size

    def __call__(self, img: Image.Image) -> Image.Image:
        left = (img.width - self.size) // 2
        top = (img.height - self.size) // 2
        return img.crop((left, top, left + self.size, top + self.size))


def build_transform() -> transforms.Compose:
    """Bicubic short-side resize -> center crop -> rescale -> normalize."""
    return transforms.Compose(
        [
            transforms.Resize(
                RESIZE_SHORT_SIDE, interpolation=InterpolationMode.BICUBIC
            ),
            CenterCropFloor(CROP_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
        ]
    )


class ImageDataset(Dataset):
    """Yields (row_index, pixel_values, ok) so batches stay aligned.

    A decode failure returns zeros with ok=False rather than raising: dropping
    the item here would silently shift every later row out of alignment with
    ids.json. Failed rows are removed after encoding instead.
    """

    def __init__(self, paths: list[Path]) -> None:
        self.paths = paths
        self.transform = build_transform()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[int, torch.Tensor, bool]:
        try:
            with Image.open(self.paths[idx]) as img:
                pixel_values = self.transform(img.convert("RGB"))
        except Exception:
            return idx, torch.zeros(3, CROP_SIZE, CROP_SIZE), False
        return idx, pixel_values, True


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------
def encode_batch(model: CLIPModel, pixel_values: torch.Tensor) -> torch.Tensor:
    """Project one batch to float32 image features, halving the batch on OOM.

    transformers v5 returns a BaseModelOutputWithPooling whose pooler_output has
    already been passed through the visual projection, so that field is the
    512-d embedding (v4 returned a bare tensor here).
    """
    try:
        with torch.autocast("cuda", dtype=torch.float16):
            outputs = model.get_image_features(pixel_values=pixel_values)
        return outputs.pooler_output.float()
    except torch.OutOfMemoryError:
        n = pixel_values.shape[0]
        if n <= MIN_BATCH_SIZE:
            raise
        torch.cuda.empty_cache()
        half = n // 2
        print(
            f"\nCUDA OOM at batch {n}; retrying as {half} + {n - half}",
            file=sys.stderr,
        )
        return torch.cat(
            [
                encode_batch(model, pixel_values[:half]),
                encode_batch(model, pixel_values[half:]),
            ]
        )


def encode_all(
    model: CLIPModel, loader: DataLoader, n_rows: int, device: torch.device
) -> tuple[np.ndarray, np.ndarray, float]:
    """Encode every batch. Returns (embeddings, ok_mask, elapsed_seconds)."""
    embeddings = np.zeros((n_rows, EMBED_DIM), dtype=np.float32)
    ok_mask = np.zeros(n_rows, dtype=bool)

    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()

    with torch.inference_mode():
        for indices, pixel_values, ok in tqdm(loader, unit="batch"):
            pixel_values = pixel_values.to(device, non_blocking=True)
            features = encode_batch(model, pixel_values)
            # L2-normalize in float32 so similarity is a plain dot product.
            features = torch.nn.functional.normalize(features, p=2, dim=1)
            rows = indices.numpy()
            embeddings[rows] = features.cpu().numpy()
            ok_mask[rows] = ok.numpy()

    torch.cuda.synchronize(device)
    return embeddings, ok_mask, time.perf_counter() - start


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
            "encode only the first N catalog rows (smoke test); artifacts go to "
            "artifacts/embeddings.limit{N}.npy and artifacts/ids.limit{N}.json"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="images per forward pass (default: %(default)s)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=NUM_WORKERS,
        help="DataLoader worker processes (default: %(default)s)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")
    if not CATALOG_PATH.exists():
        raise SystemExit(f"missing input: {CATALOG_PATH} (run python -m src.prepare)")
    if not IMAGES_DIR.exists():
        raise SystemExit(f"missing input: {IMAGES_DIR} (run python -m src.prepare)")
    # Encoding 44k images on CPU is not a fallback, it is a different job.
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; refusing to encode on CPU")

    embeddings_path = (
        EMBEDDINGS_PATH
        if args.limit is None
        else ARTIFACTS_DIR / LIMITED_EMBEDDINGS_TEMPLATE.format(limit=args.limit)
    )
    ids_path = (
        IDS_PATH
        if args.limit is None
        else ARTIFACTS_DIR / LIMITED_IDS_TEMPLATE.format(limit=args.limit)
    )

    df = pd.read_parquet(CATALOG_PATH, columns=["id"])
    catalog_rows = len(df)
    if args.limit is not None:
        df = df.head(args.limit).reset_index(drop=True)

    paths = [IMAGES_DIR / f"{pid}.jpg" for pid in df["id"]]
    on_disk = np.fromiter((p.exists() for p in paths), dtype=bool, count=len(paths))
    missing = int((~on_disk).sum())
    if missing:
        df = df[on_disk].reset_index(drop=True)
        paths = [p for p, keep in zip(paths, on_disk) if keep]
    if not paths:
        raise SystemExit("no images to encode")

    device = torch.device("cuda")
    print(f"catalog rows:                    {catalog_rows}")
    print(f"  dropped (image not on disk):   {missing}")
    print(f"images to encode:                {len(paths)}")
    print(f"device:                          {torch.cuda.get_device_name(device)}")
    print(f"model:                           {MODEL_ID}")
    print(f"batch size:                      {args.batch_size}")
    print(f"dataloader workers:              {args.workers}")
    print("autocast:                        cuda fp16")

    model = CLIPModel.from_pretrained(MODEL_ID).to(device).eval()

    loader = DataLoader(
        ImageDataset(paths),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )

    print(f"\nencoding {len(paths)} images -> {embeddings_path}")
    embeddings, ok_mask, elapsed = encode_all(model, loader, len(paths), device)

    peak_allocated = torch.cuda.max_memory_allocated(device) / 1024**2
    peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**2

    failed = int((~ok_mask).sum())
    if failed:
        print(f"images failed to decode (dropped): {failed}", file=sys.stderr)
        for path in [p for p, ok in zip(paths, ok_mask) if not ok][:20]:
            print(f"  {path.name}", file=sys.stderr)
        embeddings = embeddings[ok_mask]
        df = df[ok_mask].reset_index(drop=True)

    ids = [int(pid) for pid in df["id"]]
    if embeddings.shape[0] != len(ids):
        raise SystemExit(
            f"embeddings/ids length mismatch: {embeddings.shape[0]} vs {len(ids)}"
        )

    norms = np.linalg.norm(embeddings, axis=1)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_path, embeddings)
    ids_path.write_text(json.dumps(ids), encoding="utf-8")

    print(f"\nencoded:                         {embeddings.shape[0]} images")
    print(f"elapsed:                         {elapsed:.1f} s")
    print(
        f"throughput:                      {embeddings.shape[0] / elapsed:.1f} img/s"
        " (includes image loading)"
    )
    print(f"peak VRAM allocated:             {peak_allocated:.0f} MiB")
    print(f"peak VRAM reserved:              {peak_reserved:.0f} MiB")
    print(f"embeddings:                      {embeddings.shape} {embeddings.dtype}")
    print(f"L2 norm min/max:                 {norms.min():.6f} / {norms.max():.6f}")
    print(f"embeddings written:              {embeddings_path}")
    print(f"ids written:                     {ids_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Measure the cost of getting the center-crop offset wrong.

CLIPImageProcessor crops at floor offsets, (size - crop) // 2. torchvision's
CenterCrop rounds instead, so the two disagree by one pixel whenever
size - crop is odd -- which is nearly every image here, because a short-side-224
resize of this dataset yields 224x299 (299 - 224 = 75, odd). The shift is
invisible in the images and silent in the pipeline: nothing errors, every
embedding is simply built from a slightly different crop than the one the
checkpoint's own processor would produce.

This script quantifies that, on the same fp16 encode path src.embed uses:

    parity  src.embed.build_transform vs the real CLIPImageProcessor
            (expected: pixel-identical, cosine 1.0000)
    cost    torchvision CenterCrop vs build_transform, per-image cosine
            distribution over a fixed random sample

Usage:
    python -m src.crop_parity
    python -m src.crop_parity --limit 64
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from transformers import AutoImageProcessor, CLIPModel

from src.embed import (
    CATALOG_PATH,
    CROP_SIZE,
    IMAGE_MEAN,
    IMAGE_STD,
    IMAGES_DIR,
    MODEL_ID,
    RESIZE_SHORT_SIDE,
    build_transform,
)

# --- Sampling ---------------------------------------------------------------
SAMPLE_SIZE = 1000
SEED = 42
PARITY_SAMPLE = 256  # CLIPImageProcessor is slow; parity needs fewer images
ENCODE_BATCH = 64
PERCENTILES = (0, 0.5, 1, 5, 25, 50)


def round_offset_transform() -> transforms.Compose:
    """build_transform() with torchvision's rounding CenterCrop -- the bug."""
    return transforms.Compose(
        [
            transforms.Resize(
                RESIZE_SHORT_SIDE, interpolation=InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(CROP_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
        ]
    )


def resolve_device(device: str | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    print("warning: CUDA not available; encoding on CPU", file=sys.stderr)
    return torch.device("cpu")


def sample_paths(sample_size: int) -> list[Path]:
    catalog = pd.read_parquet(CATALOG_PATH)
    rng = np.random.default_rng(SEED)
    size = min(sample_size, len(catalog))
    rows = rng.choice(len(catalog), size=size, replace=False)
    return [IMAGES_DIR / f"{pid}.jpg" for pid in catalog["id"].to_numpy()[rows]]


def load_rgb(paths: list[Path]) -> list[Image.Image]:
    images = []
    for path in paths:
        with Image.open(path) as img:
            images.append(img.convert("RGB"))
    return images


def to_pixels(images: list[Image.Image], transform) -> torch.Tensor:
    return torch.stack([transform(img) for img in images])


def encode_pixels(
    model: CLIPModel, device: torch.device, pixel_values: torch.Tensor
) -> np.ndarray:
    """Encode a pixel tensor to L2-normalized float32, as src.embed does."""
    out = []
    with torch.inference_mode():
        for start in range(0, len(pixel_values), ENCODE_BATCH):
            batch = pixel_values[start : start + ENCODE_BATCH].to(device)
            with torch.autocast(
                "cuda", dtype=torch.float16, enabled=device.type == "cuda"
            ):
                features = model.get_image_features(pixel_values=batch).pooler_output
            features = torch.nn.functional.normalize(features.float(), p=2, dim=1)
            out.append(features.cpu().numpy())
    return np.concatenate(out)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="use only N images (smoke test)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=SAMPLE_SIZE,
        help="images to sample (default: %(default)s)",
    )
    parser.add_argument(
        "--parity-sample",
        type=int,
        default=PARITY_SAMPLE,
        help="images for the CLIPImageProcessor comparison (default: %(default)s)",
    )
    parser.add_argument("--device", default=None, help="cuda or cpu (default: auto)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    for path in (CATALOG_PATH, IMAGES_DIR):
        if not path.exists():
            raise SystemExit(f"missing input: {path} (run python -m src.prepare)")

    sample_size = args.sample_size if args.limit is None else args.limit
    paths = sample_paths(sample_size)
    device = resolve_device(args.device)
    model = CLIPModel.from_pretrained(MODEL_ID).to(device).eval()

    floor = build_transform()
    rounded = round_offset_transform()
    images = load_rgb(paths)
    odd = sum(1 for img in images if (max(img.size) - CROP_SIZE) % 2 == 1)

    print(f"sample:   {len(paths)} images, seed {SEED}, device {device.type}")
    print(f"model:    {MODEL_ID}")
    print(
        f"offsets:  {odd} of {len(paths)} images have an odd size-crop "
        "difference, so floor and round pick different pixels"
    )

    # --- parity: our transform vs the checkpoint's own processor -------------
    parity_images = images[: min(args.parity_sample, len(images))]
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    hf_pixels = processor(images=parity_images, return_tensors="pt")["pixel_values"]
    hf_emb = encode_pixels(model, device, hf_pixels)

    print(f"\n{'=' * 74}")
    print(
        f"PARITY  |  transform vs CLIPImageProcessor ({len(parity_images)} images)"
    )
    print(f"{'=' * 74}")
    print(f"{'transform':<28}{'max |pixel diff|':>18}{'mean cosine':>14}{'min':>10}")
    for name, transform in (
        ("build_transform (floor)", floor),
        ("CenterCrop (round)", rounded),
    ):
        pixels = to_pixels(parity_images, transform)
        cos = (encode_pixels(model, device, pixels) * hf_emb).sum(axis=1)
        delta = (pixels - hf_pixels).abs().max().item()
        print(f"{name:<28}{delta:>18.4f}{cos.mean():>14.4f}{cos.min():>10.4f}")

    # --- cost: the same images through both transforms ----------------------
    floor_emb = encode_pixels(model, device, to_pixels(images, floor))
    round_emb = encode_pixels(model, device, to_pixels(images, rounded))
    cos = (floor_emb * round_emb).sum(axis=1)

    print(f"\n{'=' * 74}")
    print(
        f"COST  |  round-offset vs floor-offset embeddings ({len(images)} images)"
    )
    print(f"{'=' * 74}")
    print("per-image cosine between the two embeddings of the same image")
    print(f"  mean     {cos.mean():.4f}")
    for q in PERCENTILES:
        print(f"  p{q:<7} {np.percentile(cos, q):.4f}")
    print(f"  below 0.99   {int((cos < 0.99).sum())} of {len(cos)} images")
    print(
        "\nthe mean is close to 1 because the shift is one pixel; the tail is what\n"
        "matters, and it is a silent difference from the crop the checkpoint expects."
    )
    print("reproduce: python -m src.crop_parity")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

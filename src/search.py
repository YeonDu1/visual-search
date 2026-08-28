"""Bidirectional CLIP search over the precomputed catalog embeddings.

SearchEngine loads artifacts/embeddings.npy, artifacts/ids.json and
data/catalog.parquet once at construction and answers three queries:

    text_to_image(query, k)                    text  -> catalog products
    image_to_image(path_or_pil, k)             image -> catalog products
    image_to_text(path_or_pil, candidates, k)  image -> candidate captions

Every score is a plain dot product between L2-normalized 512-d vectors, i.e.
cosine similarity in [-1, 1]. No FAISS: 44k x 512 float32 is ~87MB, so a numpy
matmul over the whole index is the entire search.

Image preprocessing is imported from src.embed rather than reimplemented, so a
query image goes through exactly the transform the index was built with
(bicubic short-side resize -> floor-offset center crop -> CLIP normalize).
Text encoding uses the same checkpoint and the same fp16 autocast.

Usage:
    python -m src.search --text "red running shoes for women" -k 5
    python -m src.search --image data/images_224/15970.jpg -k 5
    python -m src.search --image data/images_224/15970.jpg --captions "a shirt" "a watch"
    python -m src.search --text "black handbag" --limit 2000
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoTokenizer, CLIPModel

from src.embed import EMBED_DIM, MODEL_ID, build_transform

# --- Paths ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CATALOG_PATH = DATA_DIR / "catalog.parquet"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
EMBEDDINGS_PATH = ARTIFACTS_DIR / "embeddings.npy"
IDS_PATH = ARTIFACTS_DIR / "ids.json"

# --- Encoding parameters ----------------------------------------------------
TEXT_CONTEXT_LENGTH = 77  # CLIP's fixed text context; longer queries truncate
TEXT_BATCH_SIZE = 256
IMAGE_BATCH_SIZE = 64
NORM_TOLERANCE = 1e-3  # embeddings.npy must already be L2-normalized

ImageLike = str | Path | Image.Image


class SearchEngine:
    """In-memory CLIP index over the catalog.

    Args:
        embeddings_path/ids_path/catalog_path: artifacts to load.
        device: "cuda", "cpu", or None to auto-select. Auto-select prefers CUDA
            and warns on stderr when it falls back, so a CPU run is never
            silent.
        limit: keep only the first N index rows (smoke testing).
        load_model: set False to work with the vectors and catalog only; the
            model then loads on the first encode call.
    """

    def __init__(
        self,
        embeddings_path: Path = EMBEDDINGS_PATH,
        ids_path: Path = IDS_PATH,
        catalog_path: Path = CATALOG_PATH,
        device: str | torch.device | None = None,
        limit: int | None = None,
        load_model: bool = True,
    ) -> None:
        for path in (embeddings_path, ids_path, catalog_path):
            if not path.exists():
                raise SystemExit(
                    f"missing input: {path} "
                    "(run python -m src.prepare and python -m src.embed)"
                )

        self.embeddings: np.ndarray = np.load(embeddings_path)
        self.ids: list[int] = [int(i) for i in json.loads(ids_path.read_text("utf-8"))]
        catalog = pd.read_parquet(catalog_path)

        if self.embeddings.ndim != 2 or self.embeddings.shape[1] != EMBED_DIM:
            raise SystemExit(
                f"{embeddings_path} has shape {self.embeddings.shape}, "
                f"expected (N, {EMBED_DIM})"
            )
        if self.embeddings.shape[0] != len(self.ids):
            raise SystemExit(
                f"embeddings/ids length mismatch: "
                f"{self.embeddings.shape[0]} vs {len(self.ids)}"
            )

        if limit is not None:
            self.embeddings = self.embeddings[:limit]
            self.ids = self.ids[:limit]

        # Row i of embeddings.npy is ids.json[i]; reindex the catalog onto that
        # order instead of assuming the parquet still matches it.
        catalog = catalog.set_index("id")
        unknown = [i for i in self.ids if i not in catalog.index]
        if unknown:
            raise SystemExit(
                f"{len(unknown)} ids from {ids_path} are absent from "
                f"{catalog_path} (first: {unknown[:5]})"
            )
        self.catalog: pd.DataFrame = catalog.loc[self.ids].reset_index()

        self.embeddings = np.ascontiguousarray(self.embeddings, dtype=np.float32)
        norms = np.linalg.norm(self.embeddings, axis=1)
        if np.abs(norms - 1.0).max() > NORM_TOLERANCE:
            raise SystemExit(
                f"{embeddings_path} is not L2-normalized "
                f"(norm range {norms.min():.6f}..{norms.max():.6f}); "
                "similarity would not be a dot product"
            )

        self._row_of_id = {pid: row for row, pid in enumerate(self.ids)}
        self.device = self._resolve_device(device)
        self._model: CLIPModel | None = None
        self._tokenizer = None
        self._transform = build_transform()
        if load_model:
            self._ensure_model()

    # -- setup ---------------------------------------------------------------
    @staticmethod
    def _resolve_device(device: str | torch.device | None) -> torch.device:
        if device is not None:
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        print(
            "warning: CUDA not available; SearchEngine is encoding on CPU",
            file=sys.stderr,
        )
        return torch.device("cpu")

    def _ensure_model(self) -> CLIPModel:
        if self._model is None:
            self._model = CLIPModel.from_pretrained(MODEL_ID).to(self.device).eval()
            self._tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        return self._model

    def _autocast(self):
        """fp16 autocast on CUDA, matching how the index was encoded."""
        return torch.autocast(
            "cuda", dtype=torch.float16, enabled=self.device.type == "cuda"
        )

    def __len__(self) -> int:
        return len(self.ids)

    # -- encoding ------------------------------------------------------------
    def encode_text(
        self, texts: str | Sequence[str], batch_size: int = TEXT_BATCH_SIZE
    ) -> np.ndarray:
        """Encode text to L2-normalized float32 (n, 512)."""
        if isinstance(texts, str):
            texts = [texts]
        texts = list(texts)
        if not texts:
            return np.zeros((0, EMBED_DIM), dtype=np.float32)

        model = self._ensure_model()
        out = np.zeros((len(texts), EMBED_DIM), dtype=np.float32)
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                chunk = texts[start : start + batch_size]
                tokens = self._tokenizer(
                    chunk,
                    padding=True,
                    truncation=True,
                    max_length=TEXT_CONTEXT_LENGTH,
                    return_tensors="pt",
                ).to(self.device)
                with self._autocast():
                    # v5 returns BaseModelOutputWithPooling whose pooler_output
                    # is already through the text projection.
                    features = model.get_text_features(**tokens).pooler_output
                features = torch.nn.functional.normalize(features.float(), p=2, dim=1)
                out[start : start + len(chunk)] = features.cpu().numpy()
        return out

    def encode_image(
        self,
        images: ImageLike | Sequence[ImageLike],
        batch_size: int = IMAGE_BATCH_SIZE,
    ) -> np.ndarray:
        """Encode images to L2-normalized float32 (n, 512).

        Accepts paths or PIL images. Decode failures raise: unlike the bulk
        encode in src.embed there is no row alignment to preserve, and a query
        that silently became a zero vector would return nonsense.
        """
        if isinstance(images, (str, Path, Image.Image)):
            images = [images]
        images = list(images)
        if not images:
            return np.zeros((0, EMBED_DIM), dtype=np.float32)

        model = self._ensure_model()
        out = np.zeros((len(images), EMBED_DIM), dtype=np.float32)
        with torch.inference_mode():
            for start in range(0, len(images), batch_size):
                chunk = images[start : start + batch_size]
                pixel_values = torch.stack(
                    [self._transform(self._as_rgb(item)) for item in chunk]
                ).to(self.device)
                with self._autocast():
                    features = model.get_image_features(
                        pixel_values=pixel_values
                    ).pooler_output
                features = torch.nn.functional.normalize(features.float(), p=2, dim=1)
                out[start : start + len(chunk)] = features.cpu().numpy()
        return out

    @staticmethod
    def _as_rgb(image: ImageLike) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        with Image.open(image) as opened:
            return opened.convert("RGB")

    # -- ranking -------------------------------------------------------------
    def score(
        self, queries: np.ndarray, exclude_ids: Iterable[int] | None = None
    ) -> np.ndarray:
        """(n_queries, 512) -> (n_queries, n_index) cosine similarities."""
        queries = np.atleast_2d(np.asarray(queries, dtype=np.float32))
        scores = queries @ self.embeddings.T
        rows = self._rows_for_ids(exclude_ids)
        if rows.size:
            scores[:, rows] = -np.inf
        return scores

    def _rows_for_ids(self, ids: Iterable[int] | None) -> np.ndarray:
        if ids is None:
            return np.empty(0, dtype=np.int64)
        rows = [self._row_of_id[int(i)] for i in ids if int(i) in self._row_of_id]
        return np.asarray(rows, dtype=np.int64)

    def top_k(self, scores: np.ndarray, k: int) -> list[list[tuple[int, float]]]:
        """Top k (id, score) per row of a (n_queries, n_index) score matrix."""
        k = max(1, min(int(k), self.embeddings.shape[0]))
        results = []
        for row in np.atleast_2d(scores):
            candidates = np.argpartition(-row, k - 1)[:k]
            candidates = candidates[np.argsort(-row[candidates], kind="stable")]
            results.append([(self.ids[i], float(row[i])) for i in candidates])
        return results

    # -- public search API ---------------------------------------------------
    def text_to_image(
        self, query: str, k: int = 10, exclude_ids: Iterable[int] | None = None
    ) -> list[tuple[int, float]]:
        """Rank catalog products by similarity to a text query."""
        return self.top_k(self.score(self.encode_text(query), exclude_ids), k)[0]

    def image_to_image(
        self,
        path_or_pil: ImageLike,
        k: int = 10,
        exclude_ids: Iterable[int] | None = None,
    ) -> list[tuple[int, float]]:
        """Rank catalog products by similarity to a query image.

        A query image that is itself in the index will come back first with
        score ~1.0; pass its id in exclude_ids to drop it.
        """
        return self.top_k(self.score(self.encode_image(path_or_pil), exclude_ids), k)[0]

    def image_to_text(
        self, path_or_pil: ImageLike, candidate_texts: Sequence[str], k: int = 10
    ) -> list[tuple[str, float]]:
        """Rank caller-supplied captions by similarity to a query image.

        Returns (candidate_text, score): here the candidate string is its own
        identifier, since the candidates do not live in the catalog. Captions
        are encoded verbatim, so any "a photo of a ..." templating is the
        caller's choice.
        """
        candidates = list(candidate_texts)
        if not candidates:
            return []
        image_vec = self.encode_image(path_or_pil)
        scores = (self.encode_text(candidates) @ image_vec[0]).astype(np.float32)
        k = max(1, min(int(k), len(candidates)))
        order = np.argsort(-scores, kind="stable")[:k]
        return [(candidates[i], float(scores[i])) for i in order]

    # -- convenience ---------------------------------------------------------
    def describe(self, product_id: int) -> dict:
        """Catalog row for a product id, as a plain dict."""
        row = self.catalog.iloc[self._row_of_id[int(product_id)]]
        return {c: (None if pd.isna(v) else v) for c, v in row.items()}


# ---------------------------------------------------------------------------
# CLI (smoke test)
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--text", help="text query -> images")
    parser.add_argument("--image", help="image query -> images")
    parser.add_argument(
        "--captions",
        nargs="+",
        help="with --image, rank these captions instead (image -> text)",
    )
    parser.add_argument("-k", type=int, default=5, help="results (default: %(default)s)")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="index only the first N catalog rows (smoke test)",
    )
    parser.add_argument("--device", default=None, help="cuda or cpu (default: auto)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.text and not args.image:
        raise SystemExit("pass --text and/or --image")

    engine = SearchEngine(limit=args.limit, device=args.device)
    print(f"index: {len(engine)} products, device {engine.device.type}")

    def show(results: list[tuple[int, float]]) -> None:
        for rank, (pid, score) in enumerate(results, 1):
            row = engine.describe(pid)
            print(
                f"  {rank}. {score:+.4f}  {pid}  "
                f"{row['articleType']} / {row['baseColour']} / {row['gender']}"
                f"  {row['productDisplayName']}"
            )

    if args.text:
        print(f"\ntext -> image: {args.text!r}")
        show(engine.text_to_image(args.text, args.k))

    if args.image and args.captions:
        print(f"\nimage -> text: {args.image}")
        for rank, (text, score) in enumerate(
            engine.image_to_text(args.image, args.captions, args.k), 1
        ):
            print(f"  {rank}. {score:+.4f}  {text}")
    elif args.image:
        print(f"\nimage -> image: {args.image}")
        query_stem = Path(args.image).stem
        exclude = [int(query_stem)] if query_stem.isdigit() else None
        show(engine.image_to_image(args.image, args.k, exclude_ids=exclude))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

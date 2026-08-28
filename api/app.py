"""Flask REST API over the precomputed CLIP catalog index.

Endpoints
    GET  /health                index size, device, label set
    POST /search/text           {"query": str, "k": int}          -> products
    POST /search/image          multipart file field "file"       -> products
    POST /classify              multipart file field "file"       -> subCategories
    GET  /images/{id}.jpg       static image from data/images_224

The SearchEngine and the zero-shot label vectors are built once at import time,
not per request: loading artifacts/embeddings.npy (87MB), the catalog and the
CLIP checkpoint takes seconds, and encoding the label prompts is a GPU round
trip that would otherwise repeat on every /classify call. A request therefore
costs one CLIP encode of the query plus one numpy matmul.

Every response body is JSON, including errors ({"error": ..., "status": ...}),
and every product hit has the same shape: id, score, image_url and the catalog
metadata in METADATA_FIELDS.

Concurrency: the dev server is threaded, so encode calls are serialized behind
_ENCODE_LOCK. One CLIP model on one GPU is not a shared-nothing resource, and
interleaved autocast batches are not worth the throughput.

Note: `flask run --debug` imports this module in both the reloader parent and
the child, so the index loads twice. Pass --no-reload, or use `python -m
api.app`, to load it once.

Usage:
    flask --app api.app run --debug
    python -m api.app --limit 2000        # smoke test on a small index
    python -m api.app --port 5001 --device cpu

Environment (for the flask CLI, which cannot take our flags):
    VISUAL_SEARCH_LIMIT=2000    index only the first N catalog rows
    VISUAL_SEARCH_DEVICE=cpu    override device auto-selection
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import threading

import numpy as np
import pandas as pd
import torch
from flask import Flask, abort, jsonify, request, send_from_directory, url_for
from flask_cors import CORS
from PIL import Image, UnidentifiedImageError
from werkzeug.exceptions import HTTPException

from src.embed import IMAGES_DIR, MODEL_ID
from src.search import SearchEngine
from src.zeroshot import LABEL_COLUMN, build_label_vectors

# --- Response shape ---------------------------------------------------------
# Catalog columns returned with every product hit.
METADATA_FIELDS = (
    "articleType",
    "subCategory",
    "baseColour",
    "gender",
    "productDisplayName",
)

# --- Request limits ---------------------------------------------------------
MIN_K = 1
MAX_K = 50
DEFAULT_K = 10
CLASSIFY_DEFAULT_K = 5
MAX_QUERY_CHARS = 500  # CLIP truncates at 77 tokens anyway; this bounds the body
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
IMAGE_MAX_AGE = 60 * 60 * 24  # catalog images are immutable once prepared

# --- Zero-shot classification ----------------------------------------------
# The expanded surface forms are the better variant (0.6835 overall / 0.6238
# macro, vs 0.6504 / 0.5149 plain); see CLAUDE.md.
CLASSIFY_EXPANDED = True
# CLIP's learned temperature, exp(logit_scale) ~= 100. Cosines sit in a narrow
# band, so a softmax is only meaningful after this scaling. `score` below stays
# the raw cosine; `probability` is the scaled softmax over all labels.
LOGIT_SCALE = 100.0

# --- CORS -------------------------------------------------------------------
# Any localhost / 127.0.0.1 port, so the Vite dev server works without pinning
# its port number here.
CORS_ORIGINS = (r"http://localhost:\d+", r"http://127\.0\.0\.1:\d+")


# ---------------------------------------------------------------------------
# Startup: index, metadata, label vectors
# ---------------------------------------------------------------------------
ENGINE: SearchEngine
METADATA: dict[int, dict]
LABELS: list[str]
LABEL_VECTORS: np.ndarray
N_LABEL_PROMPTS: int

_ENCODE_LOCK = threading.Lock()


def _build_metadata(engine: SearchEngine) -> dict[int, dict]:
    """id -> the METADATA_FIELDS of its catalog row, NaN normalized to None.

    Precomputed so a hit is a dict lookup rather than a DataFrame .iloc, and so
    a missing column fails at startup instead of inside a request.
    """
    missing = [c for c in METADATA_FIELDS if c not in engine.catalog.columns]
    if missing:
        raise SystemExit(f"catalog is missing required columns: {missing}")
    frame = engine.catalog[["id", *METADATA_FIELDS]]
    return {
        int(row["id"]): {
            field: (None if pd.isna(row[field]) else row[field])
            for field in METADATA_FIELDS
        }
        for row in frame.to_dict("records")
    }


def load_index(limit: int | None = None, device: str | None = None) -> None:
    """Build the engine, metadata table and label vectors into module globals."""
    global ENGINE, METADATA, LABELS, LABEL_VECTORS, N_LABEL_PROMPTS

    ENGINE = SearchEngine(device=device, limit=limit)
    METADATA = _build_metadata(ENGINE)

    if ENGINE.catalog[LABEL_COLUMN].isna().any():
        raise SystemExit(f"catalog has null {LABEL_COLUMN} values; labels ill-defined")
    LABELS = sorted(ENGINE.catalog[LABEL_COLUMN].unique())
    LABEL_VECTORS, prompts = build_label_vectors(
        ENGINE, LABELS, expanded=CLASSIFY_EXPANDED
    )
    N_LABEL_PROMPTS = sum(len(group) for group in prompts)

    print(
        f"index:  {len(ENGINE):,} products, device {ENGINE.device.type}\n"
        f"labels: {len(LABELS)} {LABEL_COLUMN} values from {N_LABEL_PROMPTS} "
        f"prompts ({'expanded' if CLASSIFY_EXPANDED else 'plain'} surface forms)",
        flush=True,
    )


def _env_int(name: str) -> int | None:
    raw = os.environ.get(name)
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        raise SystemExit(f"{name}={raw!r} is not an integer")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument(
        "--limit",
        type=int,
        default=_env_int("VISUAL_SEARCH_LIMIT"),
        help="index only the first N catalog rows (smoke test)",
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("VISUAL_SEARCH_DEVICE") or None,
        help="cuda or cpu (default: auto)",
    )
    parser.add_argument("--debug", action="store_true", help="debug, no reloader")
    return parser.parse_args(argv)


# Parsed here, before the index loads, so `python -m api.app --limit N` loads the
# limited index once. Deferring to main() would load the full index at import and
# then throw it away, which on CUDA means two CLIP models resident at once.
CLI_ARGS = parse_args(sys.argv[1:]) if __name__ == "__main__" else parse_args([])

load_index(limit=CLI_ARGS.limit, device=CLI_ARGS.device)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
app.json.sort_keys = False
CORS(app, origins=list(CORS_ORIGINS))


# ---------------------------------------------------------------------------
# Request parsing and validation
# ---------------------------------------------------------------------------
def _sources() -> tuple[dict, ...]:
    """JSON body, form fields and query string, in precedence order."""
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        payload = {}
    return (payload, request.form, request.args)


def _first(field: str):
    for source in _sources():
        if field in source:
            return source[field]
    return None


def _request_k(default: int, ceiling: int = MAX_K) -> int:
    """k from the request, validated to [MIN_K, MAX_K] then capped at ceiling."""
    raw = _first("k")
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return min(default, ceiling)
    if isinstance(raw, bool):
        abort(400, f"k must be an integer between {MIN_K} and {MAX_K}, got {raw!r}")
    try:
        # via str so 3.7 and "3.7" are both rejected, not silently floored
        k = int(str(raw).strip())
    except ValueError:
        abort(400, f"k must be an integer between {MIN_K} and {MAX_K}, got {raw!r}")
    if not MIN_K <= k <= MAX_K:
        abort(400, f"k must be between {MIN_K} and {MAX_K}, got {k}")
    return min(k, ceiling)


def _request_query() -> str:
    raw = _first("query")
    if raw is None:
        abort(400, "missing required field 'query'")
    if not isinstance(raw, str):
        abort(400, f"'query' must be a string, got {type(raw).__name__}")
    query = raw.strip()
    if not query:
        abort(400, "'query' must not be empty")
    if len(query) > MAX_QUERY_CHARS:
        abort(400, f"'query' must be at most {MAX_QUERY_CHARS} characters")
    return query


def _uploaded_image() -> tuple[Image.Image, str]:
    """The uploaded file decoded to RGB, or 400.

    Validation is by decode, not by filename or Content-Type: both are
    client-supplied, and a .jpg that is not a JPEG would otherwise fail inside
    the encode path as a 500.
    """
    upload = request.files.get("file") or request.files.get("image")
    if upload is None or not upload.filename:
        abort(
            400,
            "send multipart/form-data with an image in the 'file' field "
            f"(max {MAX_UPLOAD_BYTES // (1024 * 1024)}MB)",
        )

    raw = upload.read()
    if not raw:
        abort(400, f"uploaded file {upload.filename!r} is empty")
    try:
        with Image.open(io.BytesIO(raw)) as opened:
            opened.load()
            image = opened.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        abort(
            400,
            f"{upload.filename!r} is not a readable image "
            f"({type(exc).__name__}: {exc})",
        )
    return image, upload.filename


# ---------------------------------------------------------------------------
# Response building
# ---------------------------------------------------------------------------
def _product(product_id: int, score: float) -> dict:
    pid = int(product_id)
    return {
        "id": pid,
        "score": round(float(score), 6),
        "image_url": url_for("serve_image", product_id=pid, _external=True),
        **METADATA[pid],
    }


def _products(hits: list[tuple[int, float]]) -> list[dict]:
    return [_product(pid, score) for pid, score in hits]


def _softmax(scores: np.ndarray, scale: float = LOGIT_SCALE) -> np.ndarray:
    logits = scale * scores.astype(np.float64)
    shifted = np.exp(logits - logits.max())
    return shifted / shifted.sum()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    device = ENGINE.device.type
    return jsonify(
        {
            "status": "ok",
            "index_size": len(ENGINE),
            "embedding_dim": int(ENGINE.embeddings.shape[1]),
            "device": device,
            "device_name": (
                torch.cuda.get_device_name(ENGINE.device) if device == "cuda" else None
            ),
            "model": MODEL_ID,
            "label_column": LABEL_COLUMN,
            "n_labels": len(LABELS),
            "n_label_prompts": N_LABEL_PROMPTS,
            "images_dir_present": IMAGES_DIR.is_dir(),
            "limits": {
                "min_k": MIN_K,
                "max_k": MAX_K,
                "max_upload_bytes": MAX_UPLOAD_BYTES,
            },
        }
    )


@app.post("/search/text")
def search_text():
    query = _request_query()
    k = _request_k(DEFAULT_K, ceiling=len(ENGINE))
    with _ENCODE_LOCK:
        hits = ENGINE.text_to_image(query, k)
    return jsonify(
        {"query": query, "k": k, "count": len(hits), "results": _products(hits)}
    )


@app.post("/search/image")
def search_image():
    image, filename = _uploaded_image()
    k = _request_k(DEFAULT_K, ceiling=len(ENGINE))
    with _ENCODE_LOCK:
        hits = ENGINE.image_to_image(image, k)
    return jsonify(
        {"filename": filename, "k": k, "count": len(hits), "results": _products(hits)}
    )


@app.post("/classify")
def classify():
    image, filename = _uploaded_image()
    k = _request_k(CLASSIFY_DEFAULT_K, ceiling=len(LABELS))
    with _ENCODE_LOCK:
        vector = ENGINE.encode_image(image)[0]

    # Both sides are L2-normalized, so this dot product is cosine similarity.
    scores = LABEL_VECTORS @ vector
    probabilities = _softmax(scores)
    order = np.argsort(-scores, kind="stable")[:k]
    return jsonify(
        {
            "filename": filename,
            "label_column": LABEL_COLUMN,
            "k": k,
            "predictions": [
                {
                    "label": LABELS[i],
                    "score": round(float(scores[i]), 6),
                    "probability": round(float(probabilities[i]), 6),
                }
                for i in order
            ],
        }
    )


@app.get("/images/<int:product_id>.jpg")
def serve_image(product_id: int):
    return send_from_directory(IMAGES_DIR, f"{product_id}.jpg", max_age=IMAGE_MAX_AGE)


# ---------------------------------------------------------------------------
# Errors: JSON for everything, so a client never has to parse Flask's HTML
# ---------------------------------------------------------------------------
@app.errorhandler(HTTPException)
def handle_http_error(exc: HTTPException):
    return jsonify({"error": exc.description, "status": exc.code}), exc.code


@app.errorhandler(Exception)
def handle_unexpected_error(exc: Exception):
    app.logger.exception("unhandled error on %s %s", request.method, request.path)
    return jsonify({"error": f"{type(exc).__name__}: {exc}", "status": 500}), 500


# ---------------------------------------------------------------------------
# CLI (dev server)
# ---------------------------------------------------------------------------
def main() -> int:
    args = CLI_ARGS  # already parsed at import; the index reflects it
    print(f"serving on http://{args.host}:{args.port}", flush=True)
    # use_reloader=False: reloading would load the index and model a second time
    app.run(host=args.host, port=args.port, debug=args.debug, use_reloader=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

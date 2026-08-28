"""Retrieval metrics for the CLIP catalog index, in two modes.

Both modes draw the same fixed sample (numpy default_rng(seed=42), 1000
products) so their numbers are directly comparable, and both print their metric
definitions next to the numbers.

--mode articletype (default)  category-level retrieval, per CLAUDE.md
    query text   space-joined gender, baseColour, articleType, usage
    relevant     any result sharing the query product's articleType
    top-k hit    at least one of the top k results is relevant
    ranked over  the index with the query product excluded from its own
                 results (its own image would otherwise be a free hit)
    reported     top-1, top-5, MRR

--mode strict                 instance-level retrieval, both directions
    text -> image
        query text   the product's productDisplayName, verbatim
        relevant     only the query product itself (exact id match)
        ranked over  all index products, query product included
        reported     R@1, R@5, R@10, MRR
    image -> text
        query        the product's image
        candidates   the 1000 sampled productDisplayNames
        relevant     only the caption of the query product itself
        reported     R@1, R@5, MRR

--mode both runs everything, so one command reproduces every number.

Every table carries the exact random-ranking baseline: the closed-form
expectation of the same metric under a uniformly random ranking, using each
query's true number of relevant items. It is not a simulation.

Ranks are exact without sorting the index: the rank of the first relevant
result is 1 + the number of candidates scoring strictly above the best-scoring
relevant candidate.

Usage:
    python -m src.eval
    python -m src.eval --mode both
    python -m src.eval --mode strict --limit 50
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from src.embed import IMAGES_DIR
from src.search import SearchEngine

# --- Metric parameters (do not change silently; see CLAUDE.md) ---------------
SEED = 42
SAMPLE_SIZE = 1000
QUERY_ATTRIBUTES = ("gender", "baseColour", "articleType", "usage")
ARTICLETYPE_TOP_KS = (1, 5)
STRICT_TOP_KS = (1, 5, 10)
IMAGE_TO_TEXT_TOP_KS = (1, 5)

# The dataset writes unknown categoricals as the literal string "NA" as well as
# leaving them empty, so both spellings mean "absent" when building query text.
MISSING_TOKENS = {"NA", "N/A", "None", "none", "nan", ""}

QUERY_CHUNK = 250  # queries per score matmul; bounds the (chunk, n_index) buffer


# ---------------------------------------------------------------------------
# Query construction
# ---------------------------------------------------------------------------
def is_missing(value) -> bool:
    """True for NA/NaN and for the dataset's literal placeholder strings."""
    if value is None or pd.isna(value):
        return True
    return str(value).strip() in MISSING_TOKENS


def build_attribute_query(row: pd.Series) -> str:
    """Space-join the present QUERY_ATTRIBUTES in their defined order."""
    parts = [str(row[a]).strip() for a in QUERY_ATTRIBUTES if not is_missing(row[a])]
    return " ".join(parts)


def missing_attribute_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {
        attribute: int(frame[attribute].map(is_missing).sum())
        for attribute in QUERY_ATTRIBUTES
    }


# ---------------------------------------------------------------------------
# Random-ranking baseline (exact, closed form)
# ---------------------------------------------------------------------------
def random_rank_survival(n_items: int, n_relevant: int) -> np.ndarray:
    """P(first relevant result is ranked after r), for r = 0, 1, ... , max.

    Under a uniformly random ranking of n_items of which n_relevant are
    relevant, P(R > r) = C(n_items - n_relevant, r) / C(n_items, r), computed
    in log space as a cumulative product. Index r of the result is P(R > r),
    with the final entry exactly 0.
    """
    if n_relevant <= 0:
        return np.ones(1)  # R is never reached: P(R > 0) = 1, no mass anywhere
    if n_relevant >= n_items:
        return np.array([1.0, 0.0])  # rank 1 is always relevant

    i = np.arange(n_items - n_relevant, dtype=np.float64)
    log_terms = np.log(n_items - n_relevant - i) - np.log(n_items - i)
    survival = np.exp(np.cumsum(log_terms))
    return np.concatenate(([1.0], survival, [0.0]))


def random_baseline(
    n_items: int, relevant_counts: np.ndarray, top_ks: tuple[int, ...]
) -> dict[str, float]:
    """Expected top-k accuracy and MRR of a random ranking, averaged over queries."""
    totals = {f"top{k}": 0.0 for k in top_ks}
    totals["mrr"] = 0.0
    cache: dict[int, dict[str, float]] = {}

    for n_relevant in relevant_counts:
        n_relevant = int(n_relevant)
        if n_relevant not in cache:
            survival = random_rank_survival(n_items, n_relevant)
            per_query = {
                f"top{k}": float(1.0 - survival[min(k, len(survival) - 1)])
                for k in top_ks
            }
            ranks = np.arange(1, len(survival), dtype=np.float64)
            pmf = survival[:-1] - survival[1:]
            per_query["mrr"] = float((pmf / ranks).sum())
            cache[n_relevant] = per_query
        for key, value in cache[n_relevant].items():
            totals[key] += value

    n_queries = max(1, len(relevant_counts))
    return {key: value / n_queries for key, value in totals.items()}


# ---------------------------------------------------------------------------
# Ranking core
# ---------------------------------------------------------------------------
def rank_metrics(
    query_vectors: np.ndarray,
    candidate_vectors: np.ndarray,
    relevance: dict[str, tuple[np.ndarray, np.ndarray]],
    top_ks: tuple[int, ...],
    exclude: np.ndarray | None = None,
) -> dict[str, dict]:
    """Score every query against every candidate and accumulate metrics.

    Every relevance definition in this file has the same shape -- a candidate
    is relevant when its code equals the query's code -- so `relevance` maps a
    label to (query_codes, candidate_codes) and all definitions are scored from
    one pass over the score matrix.

    `exclude[i]` is a candidate index masked out of both the scores and the
    relevant set for query i (used to drop a query product from its own
    results). Pass None to rank over every candidate.
    """
    n_queries, n_candidates = len(query_vectors), len(candidate_vectors)
    max_k = min(max(top_ks), n_candidates)
    accumulators = {
        label: {
            "hits": dict.fromkeys(top_ks, 0),
            "reciprocal_ranks": np.zeros(n_queries),
            "first_ranks": np.full(n_queries, np.inf),
            "relevant_counts": np.zeros(n_queries, dtype=np.int64),
        }
        for label in relevance
    }

    for start in range(0, n_queries, QUERY_CHUNK):
        stop = min(start + QUERY_CHUNK, n_queries)
        scores = query_vectors[start:stop] @ candidate_vectors.T

        for offset in range(stop - start):
            index = start + offset
            score_row = scores[offset]
            if exclude is not None:
                score_row[exclude[index]] = -np.inf

            top = np.argpartition(-score_row, max_k - 1)[:max_k]
            top = top[np.argsort(-score_row[top], kind="stable")]

            for label, (query_codes, candidate_codes) in relevance.items():
                relevant = candidate_codes == query_codes[index]
                if exclude is not None:
                    relevant[exclude[index]] = False
                accumulator = accumulators[label]
                accumulator["relevant_counts"][index] = int(relevant.sum())
                if not relevant.any():
                    continue
                for k in top_ks:
                    if relevant[top[:k]].any():
                        accumulator["hits"][k] += 1
                # Rank of the first relevant candidate, without sorting.
                best_relevant = score_row[relevant].max()
                rank = 1 + int((score_row > best_relevant).sum())
                accumulator["first_ranks"][index] = rank
                accumulator["reciprocal_ranks"][index] = 1.0 / rank

    n_items = n_candidates - (0 if exclude is None else 1)
    out = {}
    for label, accumulator in accumulators.items():
        counts = accumulator["relevant_counts"]
        metrics = {f"top{k}": accumulator["hits"][k] / n_queries for k in top_ks}
        metrics["mrr"] = float(accumulator["reciprocal_ranks"].mean())
        metrics["median_rank"] = float(np.median(accumulator["first_ranks"]))
        metrics["relevant_counts"] = counts
        metrics["no_relevant"] = int((counts == 0).sum())
        metrics["n_items"] = n_items
        metrics["n_queries"] = n_queries
        metrics["random"] = random_baseline(n_items, counts, top_ks)
        out[label] = metrics
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def fmt_value(value: float) -> str:
    if value == 0:
        return "0"
    return f"{value:.2e}" if value < 1e-3 else f"{value:.4f}"


def fmt_lift(value: float, base: float) -> str:
    if base <= 0:
        return "n/a"
    ratio = value / base
    if ratio >= 10000:
        return f"{ratio:,.0f}x"
    return f"{ratio:.0f}x" if ratio >= 100 else f"{ratio:.1f}x"


def print_header(mode: str, direction: str) -> None:
    print("\n" + "=" * 72)
    print(f"MODE {mode}  |  {direction}")
    print("=" * 72)


def print_definitions(lines: list[tuple[str, str]]) -> None:
    print("definitions")
    for name, text in lines:
        print(f"  {name:<14}{text}")


def print_table(
    metrics: dict, top_ks: tuple[int, ...], label: str, rank_name: str
) -> None:
    print(f"\n{'metric':<10}{'CLIP':>11}{'random':>11}{'lift':>10}")
    rows = [(f"{label}{k}", f"top{k}") for k in top_ks] + [("MRR", "mrr")]
    for name, key in rows:
        value, base = metrics[key], metrics["random"][key]
        print(
            f"{name:<10}{fmt_value(value):>11}{fmt_value(base):>11}"
            f"{fmt_lift(value, base):>10}"
        )
    print(f"median rank of {rank_name}: {metrics['median_rank']:.0f}")


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------
def run_articletype(
    engine: SearchEngine, sample_rows: np.ndarray, include_self: bool, examples: int
) -> None:
    sample = engine.catalog.iloc[sample_rows]
    queries = [build_attribute_query(row) for _, row in sample.iterrows()]

    print("\nattributes treated as absent across the full catalog")
    print('  (NA/empty or the literal string "NA"):')
    for attribute, count in missing_attribute_counts(engine.catalog).items():
        print(f"  {attribute}: {count}")
    print(f"\nexample articletype queries ({min(examples, len(queries))}):")
    for query in queries[:examples]:
        print(f"  {query!r}")

    codes, _ = pd.factorize(engine.catalog["articleType"].to_numpy())
    results = rank_metrics(
        engine.encode_text(queries),
        engine.embeddings,
        {"articleType": (codes[sample_rows], codes)},
        ARTICLETYPE_TOP_KS,
        exclude=None if include_self else sample_rows,
    )["articleType"]

    print_header("articletype", "TEXT -> IMAGE")
    print_definitions(
        [
            ("query text", f"space-joined {', '.join(QUERY_ATTRIBUTES)} of the"),
            ("", "query product, skipping absent attributes"),
            ("relevant", "any result sharing the query product's articleType"),
            ("top-k hit", "at least one of the top k results is relevant"),
            ("MRR", "mean of 1/(rank of first relevant result); a query"),
            ("", "with no relevant item contributes 0"),
            (
                "ranked over",
                f"{results['n_items']:,} products"
                f"{'' if include_self else ', query product excluded'}",
            ),
            ("sample", f"{results['n_queries']} products, default_rng(seed)"),
        ]
    )
    counts = results["relevant_counts"]
    print(
        f"\nrelevant items per query: min {counts.min()}, median "
        f"{int(np.median(counts))}, max {counts.max()} (mean {counts.mean():.1f} "
        f"of {results['n_items']:,})"
    )
    if results["no_relevant"]:
        print(f"queries with 0 relevant items: {results['no_relevant']}")
    print_table(results, ARTICLETYPE_TOP_KS, "top-", "first relevant result")


def run_strict(
    engine: SearchEngine, sample_rows: np.ndarray, examples: int, verify: bool
) -> None:
    names = engine.catalog["productDisplayName"].to_numpy()
    queries = [str(names[row]) for row in sample_rows]
    name_codes, _ = pd.factorize(names)
    # Exact-id relevance: every product is its own code, so only the query
    # product itself can be relevant.
    id_codes = np.arange(len(engine), dtype=np.int64)

    shared_names = int(pd.Series(names).duplicated(keep=False).sum())
    print(f"\nexample strict queries ({min(examples, len(queries))}):")
    for query in queries[:examples]:
        print(f"  {query!r}")

    # --- text -> image ----------------------------------------------------
    results = rank_metrics(
        engine.encode_text(queries),
        engine.embeddings,
        {
            "exact id": (sample_rows, id_codes),
            "identical name": (name_codes[sample_rows], name_codes),
        },
        STRICT_TOP_KS,
        exclude=None,
    )
    exact, by_name = results["exact id"], results["identical name"]

    print_header("strict", "TEXT -> IMAGE")
    print_definitions(
        [
            ("query text", "the product's productDisplayName, verbatim"),
            ("relevant", "only the query product itself (exact id match)"),
            ("R@k", "the query product appears in the top k results"),
            ("MRR", "mean of 1/(rank of the query product)"),
            (
                "ranked over",
                f"all {exact['n_items']:,} products, query product included",
            ),
            ("sample", f"{exact['n_queries']} products, default_rng(seed)"),
        ]
    )
    print_table(exact, STRICT_TOP_KS, "R@", "the query product")

    print(
        f"\nceiling: {shared_names:,} of {len(engine):,} catalog rows share their\n"
        "productDisplayName with another row, so exact-id relevance is partly\n"
        "unreachable from text alone. Widening relevance to any product with an\n"
        "identical productDisplayName (diagnostic, NOT the strict metric):"
    )
    print_table(by_name, STRICT_TOP_KS, "R@", "the first identical-name product")

    # --- image -> text ----------------------------------------------------
    pool = queries  # the 1000 sampled productDisplayNames, in sample order
    paths = [IMAGES_DIR / f"{pid}.jpg" for pid in engine.catalog["id"].to_numpy()[sample_rows]]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"{len(missing)} sampled images absent (first: {missing[0]})")

    image_vectors = engine.encode_image(paths)
    text_vectors = engine.encode_text(pool)
    position = np.arange(len(pool), dtype=np.int64)
    pool_name_codes = name_codes[sample_rows]

    reverse = rank_metrics(
        image_vectors,
        text_vectors,
        {
            "exact id": (position, position),
            "identical name": (pool_name_codes, pool_name_codes),
        },
        IMAGE_TO_TEXT_TOP_KS,
        exclude=None,
    )

    print_header("strict", "IMAGE -> TEXT")
    print_definitions(
        [
            ("query", "the product's image, preprocessed exactly as in"),
            ("", "src.embed (SearchEngine.encode_image)"),
            (
                "candidates",
                f"the {len(pool)} sampled productDisplayNames "
                f"({len(set(pool))} distinct)",
            ),
            ("relevant", "only the caption of the query product itself"),
            ("R@k", "that caption appears in the top k"),
            ("MRR", "mean of 1/(rank of that caption)"),
            ("ranked over", f"the {len(pool)}-caption pool"),
            ("sample", f"{len(pool)} products, default_rng(seed)"),
        ]
    )
    print_table(reverse["exact id"], IMAGE_TO_TEXT_TOP_KS, "R@", "the query caption")
    print(
        f"\ndiagnostic: {len(pool) - len(set(pool))} of {len(pool)} pool captions are "
        "a duplicate of\nanother pool caption; counting any identical caption as "
        "correct:"
    )
    print_table(
        reverse["identical name"], IMAGE_TO_TEXT_TOP_KS, "R@", "the first match"
    )

    if verify:
        # The tables above batch the matmul that image_to_text() does per call.
        # Compare per candidate index, not rank for rank: fp16 encoding is
        # mildly batch-size sensitive, so a rank-wise comparison could pair up
        # two different captions and hide a real disagreement.
        batched_scores = image_vectors[0] @ text_vectors.T
        solo_scores = engine.encode_image(paths[0])[0] @ text_vectors.T
        top_batched = pool[int(np.argmax(batched_scores))]
        top_single = engine.image_to_text(paths[0], pool, k=1)[0][0]
        delta = float(np.abs(solo_scores - batched_scores).max())
        print(
            f"\nimage_to_text() vs batched matmul on {paths[0].name}: top-1 caption "
            f"{'agrees' if top_single == top_batched else 'DIFFERS'}, "
            f"max per-candidate score delta {delta:.2e}\n"
            "(a nonzero delta is fp16 batch-size sensitivity in the image encoder,"
            " not a preprocessing difference)"
        )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--mode",
        choices=("articletype", "strict", "both"),
        default="articletype",
        help="which metric definition to run (default: %(default)s)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="cap the sample at N products (smoke test)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=SAMPLE_SIZE,
        help="products to sample (default: %(default)s)",
    )
    parser.add_argument(
        "--seed", type=int, default=SEED, help="sampling seed (default: %(default)s)"
    )
    parser.add_argument(
        "--include-self",
        action="store_true",
        help=(
            "articletype mode only: do not exclude the query product from its "
            "own results (strict mode always includes it, it is the target)"
        ),
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=5,
        help="query texts to print as a sanity check (default: %(default)s)",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="skip the image_to_text() vs batched-matmul agreement check",
    )
    parser.add_argument("--device", default=None, help="cuda or cpu (default: auto)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    engine = SearchEngine(device=args.device)
    n_items = len(engine)

    sample_size = args.sample_size
    if args.limit is not None:
        sample_size = min(sample_size, args.limit)
    if sample_size > n_items:
        raise SystemExit(f"--sample-size {sample_size} exceeds index size {n_items}")

    # One draw shared by both modes so their numbers are comparable.
    rng = np.random.default_rng(args.seed)
    sample_rows = rng.choice(n_items, size=sample_size, replace=False)

    print(f"index:                 {n_items:,} products, device {engine.device.type}")
    print(f"sample:                {sample_size} products, seed {args.seed}")
    print(f"mode:                  {args.mode}")

    start = time.perf_counter()
    if args.mode in ("articletype", "both"):
        run_articletype(engine, sample_rows, args.include_self, args.examples)
    if args.mode in ("strict", "both"):
        run_strict(engine, sample_rows, args.examples, not args.no_verify)
    elapsed = time.perf_counter() - start

    print(
        f"\nmode {args.mode}: {sample_size} queries scored in {elapsed:.1f} s\n"
        f"reproduce: python -m src.eval --mode {args.mode}"
        + (f" --seed {args.seed}" if args.seed != SEED else "")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

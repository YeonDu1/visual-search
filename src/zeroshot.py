"""Zero-shot subCategory classification with prompt ensembling.

Every catalog product is assigned one of the 45 subCategory labels by nearest
label vector. A label vector is built by averaging the encodings of several
prompts for that label and re-normalizing, which is the standard CLIP
zero-shot recipe: the mean of unit vectors is not a unit vector, so the
renormalize step is what keeps the score a plain dot product.

    label_vector(L) = normalize( mean over prompts p of encode_text(p) )
    prediction(i)   = argmax over L of  label_vector(L) . image_embedding(i)

Prompts are the cross product of TEMPLATES and the label's surface forms. Two
templates are fixed by spec ("a photo of a {}" and "a product photo of {},
fashion e-commerce"); the rest describe the e-commerce framing of this dataset.

Surface forms exist because several subCategory names are warehouse jargon
rather than things CLIP has seen captioned: "Topwear", "Innerwear", "Lips",
"Eyes", "Skin". LABEL_SURFACES maps those to natural noun phrases derived from
the articleType values actually in each class (Lips is 315 lipsticks and 144
lip glosses, so "lipstick" is a truer prompt than "lips"). Two variants are
reported so the effect is measured rather than asserted:

    plain      the label string itself, lowercased, through the templates
    expanded   LABEL_SURFACES phrasings through the same templates

Text encoding goes through SearchEngine.encode_text, so label prompts and the
image index cannot drift apart: same checkpoint, same tokenizer, same fp16
autocast, same dot product.

Near-synonym labels are handled and reported, not hidden. The script prints the
label pairs whose ensembled vectors are closest in CLIP space, alongside the
Jaccard overlap of their articleType sets, which separates the two causes: a
high cosine with low articleType overlap is CLIP conflating distinct classes
(Sandal vs Flip Flops), while a high cosine with high overlap means the label
pair is degenerate in the data itself and no prompt can fix it (Fragrance and
Perfumes are both "Perfume and Body Mist"). Classes that lose all their items
to such a neighbour, or that are never predicted at all, are listed as
collapsed.

Usage:
    python -m src.zeroshot
    python -m src.zeroshot --variant expanded
    python -m src.zeroshot --limit 2000
    python -m src.zeroshot --worst 15 --show-prompts
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from src.search import SearchEngine

# --- Label set --------------------------------------------------------------
LABEL_COLUMN = "subCategory"

# --- Prompt ensembling ------------------------------------------------------
# The first two are required; the others add e-commerce framing. Every template
# takes one {} and is applied to every surface form of every label.
TEMPLATES = (
    "a photo of a {}",
    "a product photo of {}, fashion e-commerce",
    "a product photo of a {} on a plain white background",
    "an online store catalogue image of a {}",
    "a close up product shot of a {}",
)

# Natural-language phrasings for labels whose dataset name is jargon, ambiguous,
# or a bare body part. Derived from the articleType composition of each class,
# not invented: see the module docstring. Labels absent from this map fall back
# to their lowercased name, which is already a fine prompt ("belts", "watches").
LABEL_SURFACES: dict[str, tuple[str, ...]] = {
    # apparel jargon
    "Topwear": ("t-shirt or shirt", "top worn on the upper body"),
    "Bottomwear": ("jeans or trousers", "clothing worn on the lower body"),
    "Innerwear": ("underwear", "briefs or bra"),
    "Loungewear and Nightwear": ("nightwear", "pyjamas or night suit", "bath robe"),
    "Apparel Set": ("matching clothing set", "kurta set"),
    "Saree": ("saree", "sari"),
    # footwear
    "Shoes": ("shoes", "pair of shoes"),
    "Sandal": ("sandals", "pair of sandals"),
    "Flip Flops": ("flip flops", "pair of rubber slippers"),
    "Shoe Accessories": ("shoe laces", "shoe care accessory"),
    # bags and hard goods
    "Bags": ("bag", "handbag or backpack"),
    "Watches": ("wristwatch", "watch"),
    "Jewellery": ("jewellery", "necklace or earrings"),
    "Eyewear": ("sunglasses", "pair of eyeglasses"),
    "Headwear": ("cap", "cap or hat"),
    "Ties": ("necktie", "tie"),
    "Cufflinks": ("cufflinks", "pair of cufflinks"),
    "Socks": ("socks", "pair of socks"),
    "Gloves": ("gloves", "pair of gloves"),
    "Stoles": ("stole", "shawl"),
    "Mufflers": ("muffler", "winter neck scarf"),
    # beauty: the label is a body part, the product is what is photographed
    "Lips": ("lipstick", "lip gloss", "lip makeup product"),
    "Eyes": ("eyeliner or mascara", "eye makeup product"),
    "Nails": ("nail polish bottle", "nail polish"),
    "Makeup": ("makeup compact or foundation", "cosmetics product"),
    "Skin": ("face moisturiser jar", "face cream"),
    "Skin Care": ("face wash or sunscreen bottle", "skin care product"),
    "Bath and Body": ("body lotion or body wash bottle", "bath and body product"),
    "Beauty Accessories": ("beauty accessory", "makeup applicator"),
    "Hair": ("hair colour kit", "hair care product"),
    "Fragrance": ("perfume bottle or deodorant", "body mist bottle"),
    "Perfumes": ("perfume bottle",),
    # long tail
    "Accessories": ("accessory gift set", "gift set box"),
    "Sports Accessories": ("sports wristband", "wristband"),
    "Wristbands": ("wristband",),
    "Sports Equipment": ("football or basketball", "sports ball"),
    "Water Bottle": ("water bottle",),
    "Umbrellas": ("umbrella",),
    "Free Gifts": ("free gift item", "promotional gift item"),
    "Vouchers": ("gift voucher",),
    "Home Furnishing": ("cushion cover", "home furnishing item"),
}

# --- Reporting thresholds ---------------------------------------------------
SYNONYM_COSINE = 0.90  # label pairs at or above this are flagged as near-synonyms
SYNONYM_PAIRS = 12  # how many of the closest pairs to print
COLLAPSE_RECALL = 0.02  # a class at or below this accuracy has lost its items
COLLAPSE_PREDICTED = 0.05  # ... or is predicted for < 5% as many items as its support
WORST_CLASSES = 10
SCORE_CHUNK = 8192  # products per matmul; bounds the (n_labels, chunk) buffer


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
def surfaces_for(label: str, expanded: bool) -> tuple[str, ...]:
    """Surface forms to render into the templates for one label."""
    if not expanded:
        return (label.lower(),)
    return LABEL_SURFACES.get(label, (label.lower(),))


def prompts_for(label: str, expanded: bool) -> list[str]:
    return [t.format(s) for s in surfaces_for(label, expanded) for t in TEMPLATES]


def build_label_vectors(
    engine: SearchEngine, labels: list[str], expanded: bool
) -> tuple[np.ndarray, list[list[str]]]:
    """Encode every prompt once, then average per label and re-normalize.

    All prompts go through a single encode_text call so the batching matches
    the rest of the project; the per-label mean is taken afterwards over the
    rows belonging to that label.
    """
    per_label = [prompts_for(label, expanded) for label in labels]
    flat = [p for group in per_label for p in group]
    encoded = engine.encode_text(flat)

    vectors = np.zeros((len(labels), encoded.shape[1]), dtype=np.float32)
    start = 0
    for i, group in enumerate(per_label):
        mean = encoded[start : start + len(group)].mean(axis=0)
        norm = float(np.linalg.norm(mean))
        if norm == 0.0:  # only reachable if the prompts cancel exactly
            raise SystemExit(f"label {labels[i]!r} averaged to a zero vector")
        vectors[i] = mean / norm
        start += len(group)
    return vectors, per_label


# ---------------------------------------------------------------------------
# Classification and metrics
# ---------------------------------------------------------------------------
def classify(engine: SearchEngine, label_vectors: np.ndarray) -> np.ndarray:
    """Predicted label index for every indexed product.

    Scoring the whole index at once would allocate (n_labels, n_products),
    only 8MB at 45 x 44k; the chunk keeps that flat if the label set grows.
    """
    n_products = len(engine)
    predictions = np.zeros(n_products, dtype=np.int64)
    for start in range(0, n_products, SCORE_CHUNK):
        stop = min(start + SCORE_CHUNK, n_products)
        scores = label_vectors @ engine.embeddings[start:stop].T
        predictions[start:stop] = np.argmax(scores, axis=0)
    return predictions


def evaluate(truth: np.ndarray, predicted: np.ndarray, n_labels: int) -> dict:
    """Overall accuracy, macro accuracy, and the per-class confusion rows."""
    confusion = np.zeros((n_labels, n_labels), dtype=np.int64)
    np.add.at(confusion, (truth, predicted), 1)

    support = confusion.sum(axis=1)
    predicted_counts = confusion.sum(axis=0)
    correct = np.diag(confusion)
    present = support > 0
    recall = np.where(present, correct / np.maximum(support, 1), np.nan)

    return {
        "confusion": confusion,
        "support": support,
        "predicted_counts": predicted_counts,
        "correct": correct,
        "recall": recall,
        "overall": float(correct.sum() / support.sum()),
        "macro": float(np.nanmean(recall[present])),
        "n_present": int(present.sum()),
    }


def worst_classes(result: dict, n: int) -> list[int]:
    """Indices of the n weakest present classes: accuracy ascending, then
    support descending, so a big class and a 1-item class do not tie at 0."""
    present = [i for i in range(len(result["support"])) if result["support"][i] > 0]
    present.sort(key=lambda i: (result["recall"][i], -result["support"][i]))
    return present[:n]


def top_confusion(result: dict, index: int) -> tuple[int, int]:
    """(label index, count) of the most common wrong prediction for a class."""
    row = result["confusion"][index].copy()
    row[index] = -1
    target = int(np.argmax(row))
    return target, int(row[target])


# ---------------------------------------------------------------------------
# Near-synonym analysis
# ---------------------------------------------------------------------------
def articletype_sets(catalog: pd.DataFrame, labels: list[str]) -> list[set[str]]:
    grouped = catalog.groupby(LABEL_COLUMN)["articleType"].apply(set)
    return [set(grouped.get(label, set())) for label in labels]


def jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def near_synonyms(
    label_vectors: np.ndarray, labels: list[str], catalog: pd.DataFrame, top_n: int
) -> list[tuple[str, str, float, float]]:
    """Closest label pairs as (label a, label b, cosine, articleType Jaccard)."""
    similarity = label_vectors @ label_vectors.T
    rows, cols = np.triu_indices(len(labels), k=1)
    order = np.argsort(-similarity[rows, cols])[:top_n]
    type_sets = articletype_sets(catalog, labels)
    return [
        (
            labels[rows[i]],
            labels[cols[i]],
            float(similarity[rows[i], cols[i]]),
            jaccard(type_sets[rows[i]], type_sets[cols[i]]),
        )
        for i in order
    ]


def collapsed_classes(result: dict, labels: list[str]) -> list[tuple[str, str, str]]:
    """Classes that lost their items, as (label, reason, where the mass went)."""
    out = []
    for i, label in enumerate(labels):
        support = int(result["support"][i])
        if support == 0:
            continue
        recall = float(result["recall"][i])
        predicted = int(result["predicted_counts"][i])
        absorbed = recall <= COLLAPSE_RECALL
        unused = predicted < COLLAPSE_PREDICTED * support
        if not (absorbed or unused):
            continue
        target, count = top_confusion(result, i)
        reason = "absorbed" if absorbed else "rarely predicted"
        if absorbed and unused:
            reason = "absorbed, never wins"
        detail = (
            f"{count}/{support} of its items -> {labels[target]!r}, "
            f"predicted for {predicted} products overall"
        )
        out.append((label, reason, detail))
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_header(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def print_definitions(lines: list[tuple[str, str]]) -> None:
    print("definitions")
    for name, text in lines:
        print(f"  {name:<20}{text}")


def report_variant(
    variant: str,
    result: dict,
    labels: list[str],
    label_vectors: np.ndarray,
    per_label_prompts: list[list[str]],
    catalog: pd.DataFrame,
    n_worst: int,
    show_prompts: bool,
) -> None:
    support = result["support"]
    majority = int(np.argmax(support))
    n_products = int(support.sum())
    n_prompts = sum(len(group) for group in per_label_prompts)

    print_header(f"VARIANT {variant}  |  zero-shot {LABEL_COLUMN} classification")
    print_definitions(
        [
            (
                "task",
                f"assign each product one of {result['n_present']} "
                f"{LABEL_COLUMN} labels",
            ),
            ("prediction", "argmax over labels of (label vector . image"),
            ("", "embedding); both are L2-normalized, so it is cosine"),
            ("label vector", "mean of its prompt embeddings, re-normalized"),
            (
                "prompts",
                f"{n_prompts} total, {len(TEMPLATES)} templates x each "
                "label's surface forms",
            ),
            (
                "surface forms",
                "the label lowercased"
                if variant == "plain"
                else "LABEL_SURFACES phrasings (see module docstring)",
            ),
            ("overall accuracy", "correct / all products"),
            ("macro accuracy", "unweighted mean of per-class accuracy, where"),
            ("", "per-class accuracy = recall = correct_c / support_c"),
            (
                "majority baseline",
                f"always predict {labels[majority]!r} "
                f"({int(support[majority]):,} of {n_products:,})",
            ),
        ]
    )

    if show_prompts:
        print(f"\nprompt ensemble for the majority class {labels[majority]!r}:")
        for prompt in per_label_prompts[majority]:
            print(f"  {prompt!r}")

    majority_overall = float(support[majority] / n_products)
    majority_macro = 1.0 / result["n_present"]
    print(f"\n{'metric':<20}{'CLIP':>10}{'majority':>11}{'lift':>9}")
    print(
        f"{'overall accuracy':<20}{result['overall']:>10.4f}"
        f"{majority_overall:>11.4f}{result['overall'] / majority_overall:>8.2f}x"
    )
    print(
        f"{'macro accuracy':<20}{result['macro']:>10.4f}"
        f"{majority_macro:>11.4f}{result['macro'] / majority_macro:>8.2f}x"
    )
    print(
        f"\nproducts {n_products:,}   correct {int(result['correct'].sum()):,}"
        f"   classes with support {result['n_present']}"
    )

    print(
        f"\n{n_worst} worst classes by per-class accuracy "
        "(ties broken by larger support first)"
    )
    print(
        f"  {'class':<26}{'support':>8}{'acc':>8}{'pred':>7}  "
        "most common wrong prediction"
    )
    for i in worst_classes(result, n_worst):
        target, count = top_confusion(result, i)
        share = count / max(int(support[i]), 1)
        print(
            f"  {labels[i]:<26}{int(support[i]):>8}{result['recall'][i]:>8.4f}"
            f"{int(result['predicted_counts'][i]):>7}  "
            f"{labels[target]} ({share:.0%})"
        )

    print("\nclosest label pairs in CLIP space (cosine of the ensembled vectors).")
    print("articleType Jaccard says whether the pair is also degenerate in the data:")
    print("high cosine + high Jaccard means no prompt can separate them.")
    print(f"  {'pair':<48}{'cosine':>8}{'jaccard':>9}")
    for a, b, cosine, overlap in near_synonyms(
        label_vectors, labels, catalog, SYNONYM_PAIRS
    ):
        flag = "  <- near-synonym" if cosine >= SYNONYM_COSINE else ""
        print(f"  {a + ' / ' + b:<48}{cosine:>8.4f}{overlap:>9.2f}{flag}")

    collapses = collapsed_classes(result, labels)
    print(
        f"\ncollapsed classes: accuracy <= {COLLAPSE_RECALL:.2f}, or predicted for "
        f"< {COLLAPSE_PREDICTED:.0%} as many\nproducts as their support "
        f"({len(collapses)} of {result['n_present']})"
    )
    if not collapses:
        print("  none")
    for label, reason, detail in collapses:
        print(f"  {label:<26}{reason:<22}{detail}")


def compare_variants(results: dict[str, dict], labels: list[str], n: int) -> None:
    plain, expanded = results["plain"], results["expanded"]
    print_header("plain vs expanded  |  effect of the surface-form handling")
    print(f"{'metric':<20}{'plain':>10}{'expanded':>11}{'delta':>10}")
    for name, key in (("overall accuracy", "overall"), ("macro accuracy", "macro")):
        print(
            f"{name:<20}{plain[key]:>10.4f}{expanded[key]:>11.4f}"
            f"{expanded[key] - plain[key]:>+10.4f}"
        )

    delta = np.where(
        plain["support"] > 0, expanded["recall"] - plain["recall"], np.nan
    )
    order = np.argsort(-np.abs(np.nan_to_num(delta)))[:n]
    print(f"\n{n} classes most changed by the surface forms")
    print(f"  {'class':<26}{'support':>8}{'plain':>9}{'expanded':>10}{'delta':>9}")
    for i in order:
        print(
            f"  {labels[i]:<26}{int(plain['support'][i]):>8}"
            f"{plain['recall'][i]:>9.4f}{expanded['recall'][i]:>10.4f}"
            f"{delta[i]:>+9.4f}"
        )

    plain_collapsed = {c[0] for c in collapsed_classes(plain, labels)}
    expanded_collapsed = {c[0] for c in collapsed_classes(expanded, labels)}
    print(
        "\ncollapsed under plain only (recovered): "
        f"{', '.join(sorted(plain_collapsed - expanded_collapsed)) or 'none'}"
    )
    print(
        "collapsed under expanded only (new):    "
        f"{', '.join(sorted(expanded_collapsed - plain_collapsed)) or 'none'}"
    )
    print(
        "collapsed under both:                   "
        f"{', '.join(sorted(plain_collapsed & expanded_collapsed)) or 'none'}"
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--variant",
        choices=("plain", "expanded", "both"),
        default="both",
        help="prompt surface forms to use (default: %(default)s)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="classify only the first N indexed products (smoke test)",
    )
    parser.add_argument(
        "--worst",
        type=int,
        default=WORST_CLASSES,
        help="worst classes to list (default: %(default)s)",
    )
    parser.add_argument(
        "--show-prompts",
        action="store_true",
        help="print the prompt ensemble of the majority class",
    )
    parser.add_argument("--device", default=None, help="cuda or cpu (default: auto)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    engine = SearchEngine(limit=args.limit, device=args.device)
    catalog = engine.catalog

    if catalog[LABEL_COLUMN].isna().any():
        raise SystemExit(
            f"{int(catalog[LABEL_COLUMN].isna().sum())} rows have a null "
            f"{LABEL_COLUMN}; the label set would be ill-defined"
        )

    labels = sorted(catalog[LABEL_COLUMN].unique())
    truth = np.asarray(
        pd.Categorical(catalog[LABEL_COLUMN], categories=labels).codes, dtype=np.int64
    )

    unknown = sorted(set(LABEL_SURFACES) - set(labels))
    if unknown and args.limit is None:
        print(f"warning: LABEL_SURFACES keys absent from the catalog: {unknown}")
    fallback = [label for label in labels if label not in LABEL_SURFACES]

    print(f"index:    {len(engine):,} products, device {engine.device.type}")
    print(f"labels:   {len(labels)} {LABEL_COLUMN} values")
    print(
        f"surfaces: {len(labels) - len(fallback)} labels have custom surface forms, "
        f"{len(fallback)} fall back to the lowercased label"
    )
    if fallback:
        print(f"          fallback: {', '.join(fallback)}")

    variants = ("plain", "expanded") if args.variant == "both" else (args.variant,)
    results: dict[str, dict] = {}
    start = time.perf_counter()
    for variant in variants:
        label_vectors, per_label_prompts = build_label_vectors(
            engine, labels, expanded=variant == "expanded"
        )
        predicted = classify(engine, label_vectors)
        results[variant] = evaluate(truth, predicted, len(labels))
        report_variant(
            variant,
            results[variant],
            labels,
            label_vectors,
            per_label_prompts,
            catalog,
            args.worst,
            args.show_prompts,
        )
    elapsed = time.perf_counter() - start

    if len(variants) == 2:
        compare_variants(results, labels, args.worst)

    print(
        f"\n{len(variants)} variant(s) over {len(engine):,} products in {elapsed:.1f} s"
    )
    print(f"reproduce: python -m src.zeroshot --variant {args.variant}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

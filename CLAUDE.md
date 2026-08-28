\# Visual-Language E-commerce Search



CLIP-based bidirectional image/text search over the Kaggle Fashion Product

Images dataset (44,441 products).



Dataset: https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset

Model:   https://huggingface.co/openai/clip-vit-base-patch32



\## Environment

\- Python 3.11 in .venv. Activate: .\\.venv\\Scripts\\Activate.ps1

\- torch 2.13.0+cu126, CUDA available, RTX 4060 Laptop, 8GB VRAM.

\- transformers 5.16.1 (v5, NOT v4). Do not use deprecated v4 arguments such

&#x20; as use\_auth\_token or feature\_extractor. Check the installed API before

&#x20; writing code rather than assuming v4 signatures.

\- pandas 3.0.5, numpy 2.4.6, Pillow 12.3.0.



\## Data

\- Processed:  data/images\_224/       44,441 JPEGs, short side 224, \~419MB

\- Catalog:    data/catalog.parquet   44,441 rows

&#x20;             45 subCategories, 142 articleTypes

\- Embeddings: artifacts/embeddings.npy  (44,441, 512) float32, L2-normalized, \~87MB

\- IDs:        artifacts/ids.json        44,441 ints. Row i of embeddings.npy is

&#x20;             entry i of ids.json; both follow catalog row order.

\- Full encode: 44,441 images in 45.8s (970 img/s incl. image loading) at

&#x20; batch 256, peak VRAM 1114MB allocated / 1314MB reserved on the 4060.

\- Raw dataset has been deleted after preprocessing. Re-download from Kaggle

&#x20; only if src/prepare.py needs to change.

\- Everything under data/ and artifacts/ is gitignored. Never commit it.

\## Layout

\- src/prepare.py   resize + build catalog

\- src/embed.py     CLIP encoding -> artifacts/

\- src/search.py    SearchEngine class

\- src/zeroshot.py  category classification

\- src/eval.py      retrieval metrics

\- api/app.py       Flask REST API

\- web/             React (Vite)



\## Rules

\- Embeddings are L2-normalized float32. Similarity is a plain dot product.

\- No FAISS. 44k x 512 fits in RAM; use numpy matmul.

\- Encode under torch.autocast fp16, batch size 256. On CUDA OOM, halve the

&#x20; batch. Never silently fall back to CPU.

\- DataLoader with num\_workers=4, persistent\_workers=True.

\- The DataLoader image transform must center-crop with floor offsets,

&#x20; (size-crop)//2, to match CLIPImageProcessor. torchvision CenterCrop

&#x20; rounds instead, which shifts every 224x299 image by one pixel and

&#x20; silently degrades every embedding.

\- Use pathlib for all paths. No hardcoded absolute paths, no forward-slash

&#x20; string concatenation. Paths come from constants at the top of each module.

\- styles.csv has malformed rows with extra commas in productDisplayName.

&#x20; Handle them and report how many rows were dropped.

\- Every script takes --limit for smoke testing before a full run.

\- Run the code after writing it and report the real output, not the expected

&#x20; output. If it fails, show the traceback.

\- Ask before adding a dependency. Update requirements.txt when you do.



\## Metric definitions (do not change these silently)

\- Query text is built from a product's catalog attributes

&#x20; (gender, baseColour, articleType, usage).

\- top-k hit: at least one of the top k results shares the query product's

&#x20; articleType.

\- MRR: mean of 1/(rank of first relevant result), over a 1000-product sample

&#x20; with a fixed random seed.

\- Zero-shot classification is over the subCategory column.

\- eval.py must print the metric definition next to the numbers.

\- Retrieval numbers are for reference only. They are reproduced, not trusted:

&#x20; re-run the command below rather than quoting these from memory.

\## Measured metrics

\- Command: python -m src.eval --mode both   (seed 42, 1000-product sample)

\- articletype mode, text -> image, ranked over 44,440 products with the

&#x20; query product excluded from its own results:

&#x20; top-1 0.7340, top-5 0.9300, MRR 0.8187.

&#x20; Random baseline 0.0530 / 0.2163 / 0.1453, i.e. 4.3x lift at top-5.

&#x20; The baseline is high because articleType is coarse: the mean query has

&#x20; 2,355 relevant items of 44,440 and the median rank of the first relevant

&#x20; result is 1. Passing --include-self moves each number by about +0.002.

\- strict mode, text -> image, query = productDisplayName, relevant = the query

&#x20; product only (exact id), ranked over all 44,441 products:

&#x20; R@1 0.0440, R@5 0.1190, R@10 0.1820, MRR 0.0925.

&#x20; Random baseline 2.25e-05 / 1.13e-04 / 2.25e-04 / 2.54e-04.

&#x20; Median rank of the query product is 82 of 44,441.

&#x20; Ceiling: 18,413 of 44,441 rows share a productDisplayName with another

&#x20; row, so exact-id relevance is partly unreachable from text. eval.py also

&#x20; prints an identical-name diagnostic (R@1 0.0750, MRR 0.1397) to separate

&#x20; label ambiguity from genuine retrieval failure. Do not quote the

&#x20; diagnostic as the strict metric.

\- strict mode, image -> text, candidate pool = the 1000 sampled

&#x20; productDisplayNames (979 distinct):

&#x20; R@1 0.3470, R@5 0.6630, MRR 0.4929.

&#x20; Random baseline 1.00e-03 / 5.00e-03 / 7.50e-03. Median rank 3.

\- Both directions are one command: python -m src.eval --mode both, 3.4 s.



\## Commands

\- Prepare:  python -m src.prepare

\- Embed:    python -m src.embed

\- Evaluate: python -m src.eval --mode both

\- API:      flask --app api.app run --debug

\- Frontend: cd web; npm run dev


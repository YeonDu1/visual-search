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

\- Raw images:  data/fashion-dataset/images/     <-- VERIFY THIS PATH

\- Metadata:    data/fashion-dataset/styles.csv  <-- VERIFY THIS PATH

\- Processed:   data/images\_224/    (created by src/prepare.py)

\- Catalog:     data/catalog.parquet (created by src/prepare.py)

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



\## Commands

\- Prepare:  python -m src.prepare

\- Embed:    python -m src.embed

\- Evaluate: python -m src.eval

\- API:      flask --app api.app run --debug

\- Frontend: cd web; npm run dev


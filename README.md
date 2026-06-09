# Animal Re-Identification — ML Core (Part 1)

Identify *which individual* animal is in a photo (not the species) using
**patch-bag Multiple Instance Learning (MIL)**.

Each image is treated as a *bag* of patch *instances*. A **frozen** pretrained
backbone embeds every patch; a small **gated-attention** head (the only trained
part) learns which patches carry the animal's identity — fur/scale patterns,
scars, notches — with **no patch-level labels**. The pooled, L2-normalised
embedding drives **open-set retrieval**: a query is matched against a gallery of
enrolled individuals by cosine similarity, and rejected as a *new* individual
when even the best match is too weak. The attention weights double as an
**explainability heatmap** showing where the model looked.


API layer should import only from [`ml/inference/`](ml/inference/) — that is the
stable, model-internals-free seam.

## Why this design
- **Frozen backbone + tiny head** → trains on CPU, data-efficient (reuses
  pretrained representations instead of relearning them).
- **Attention MIL** → focuses on the few discriminative patches automatically and
  yields a visual explanation as a by-product.
- **Open-set retrieval** → new individuals are enrolled by appending a vector; no
  retraining, unlike a closed-set classifier.

## Layout
```
ml/
  config.py        # Pydantic settings + enums (dataset / backbone / pooling)
  data/            # dataset access, open-set split, patch tiling (bag creation)
  features/        # frozen backbone + on-disk patch-embedding cache
  models/          # GatedAttentionMIL head + (train-only) ArcFace head
  training/        # cached-feature dataset, loss, train loop
  inference/       # embedder, gallery, identifier, explain  <-- API imports here
  eval/            # rank-1/5, mAP, open-set AUROC + driver
scripts/           # CLI entrypoints
```

## Setup

### Docker (recommended for reviewers)
Reproducible on any machine with Docker — no Python/torch/lzma setup required.
```bash
docker compose build      # build the CPU image (one time)
docker compose up         # runs the smoke test → "ALL 9 CORE-LOGIC TESTS PASSED"
```
Run any pipeline step in the container (datasets + weights download into the
mounted `data/` and `artifacts/` and persist across runs):
```bash
docker compose run --rm reid python -m scripts.download_data
docker compose run --rm reid python -m scripts.precompute_features
docker compose run --rm reid python -m scripts.train
docker compose run --rm reid python -m scripts.evaluate
docker compose run --rm reid python -m scripts.demo_identify
```
`ml/`, `scripts/`, `tests/`, `api/`, and `web/` are bind-mounted, so **editing the
code on the host takes effect immediately — no rebuild**. Rebuild only when
dependencies ([requirements.txt](requirements.txt)) change.

### Web UI + API
The browser client is served same-origin by the FastAPI app:
```bash
docker compose up api            # API + UI on http://localhost:8000
```
Then open **http://localhost:8000/** for the demo UI (Identify / Enroll / Gallery),
or **http://localhost:8000/docs** for the interactive OpenAPI schema.

Typical flow: train a model first (`docker compose run --rm reid python -m scripts.train`),
restart `api`, click **Seed from dataset** in the Gallery tab, then **Identify** a
photo to see the matched individual and the attention heatmap. Until a model is
trained the UI shows a red "no trained model" banner and the API's inference
endpoints return `503`.

Sample API calls:
```bash
curl localhost:8000/health
curl -F "individual_id=turtle_17" -F "files=@a.jpg" -F "files=@b.jpg" localhost:8000/enroll
curl -F "file=@query.jpg" localhost:8000/identify
curl -F "file=@query.jpg" localhost:8000/explain --output heatmap.png
```

### Native (Intel macOS)
The container is the supported path. For a native Intel-mac install, use the
mac-specific pins (torch comes from the PyTorch CPU index, not PyPI):
```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install -r requirements-mac.txt
```
See [requirements-mac.txt](requirements-mac.txt) for why the caps exist
(Intel-mac torch is NumPy-1.x only).

Configuration is centralised in [`ml/config.py`](ml/config.py) and overridable
via `REID_*` environment variables, e.g. `REID_DATASET=IPanda50`,
`REID_POOLING=MEAN`, `REID_EPOCHS=50` (works the same with `docker compose run -e REID_...`).

## Run order
```bash
python -m scripts.download_data        # 1. fetch dataset via WildlifeDatasets
python -m scripts.precompute_features  # 2. cache frozen-backbone patch embeddings (slow, once)
python -m scripts.train                # 3. train the MIL head (CPU-friendly, reads cache)
python -m scripts.evaluate             # 4. rank-1/5, mAP, open-set AUROC report
python -m scripts.demo_identify        # 5. enroll + identify known/unknown + save heatmap
```
(In Docker, prefix each with `docker compose run --rm reid`.)

## Patch resolution (fidelity vs speed)
Each patch must reach the backbone's input size (224). `REID_PATCH_RESOLUTION`
controls *when* that happens:
- `UPSAMPLED` (default) — resize the animal to 224, tile into 56px patches, and
  let the backbone upsample each. Cheap; patches are low-detail.
- `NATIVE` — resize the animal to `patch_grid × 224` (e.g. 896) so each tile is
  natively 224 with genuine fine detail (fur/scale texture). Higher fidelity, more
  compute per image.
```bash
docker compose run --rm reid -e REID_PATCH_RESOLUTION=NATIVE python -m scripts.precompute_features
```
Re-run `precompute_features` after changing this — the cached embeddings differ
between modes.

## Ablation (quantifying what attention buys)
Train and evaluate with mean pooling to get the baseline the paper compares
against:
```bash
REID_POOLING=MEAN python -m scripts.train
REID_POOLING=MEAN python -m scripts.evaluate
```
Gated-attention MIL should beat the mean-pooling baseline on rank-1/mAP.

## Notes
- We depend on `wildlife-datasets` only (for dataset download + metadata). We do
  **not** use `wildlife-tools` — it hard-requires `faiss-gpu` (no CPU/macOS wheel)
  and similarity/retrieval/metrics are implemented natively here anyway.
- `wildlife-datasets` class names occasionally shift between versions; every call
  into it is isolated in [`ml/data/dataset.py`](ml/data/dataset.py) so any breakage
  is fixed in one place.
- The default backbone is MegaDescriptor (pulled from the HuggingFace hub); set
  `REID_BACKBONE=resnet50` for a fully offline timm fallback.

# Animal Re-Identification

Identify **which individual** animal is in a photo — not the species — using
**patch-bag Multiple Instance Learning (MIL)** with **open-set retrieval**.

This is a full demo in three Dockerised parts:

1. **ML core** (`ml/`) — the model: tiling, frozen backbone, attention-MIL head, training, retrieval, evaluation.
2. **FastAPI API** (`api/`) — an HTTP layer over the ML core's inference seam.
3. **Web UI** (`web/`) — a plain HTML/CSS/JS client served same-origin by the API.

---

## Table of contents
- [How it works](#how-it-works)
- [Project layout](#project-layout)
- [Quick start (Docker)](#quick-start-docker)
- [The workflow](#the-workflow-download--precompute--train--identify)
- [Web UI](#web-ui)
- [Multiple models / species](#multiple-models--species)
- [API reference](#api-reference)
- [Background jobs & SSE](#background-jobs--sse)
- [Configuration](#configuration)
- [Development](#development)
- [Design notes & caveats](#design-notes--caveats)

---

## How it works

Each image is treated as a **bag of patch instances**. A **frozen** pretrained
backbone embeds every patch; a small **gated-attention** head — the *only* trained
part — learns which patches carry the animal's identity (fur/scale patterns, scars,
notches) with **no patch-level labels**. The pooled, L2-normalised embedding drives
**open-set retrieval**: a query is matched against a gallery of enrolled individuals
by cosine similarity, and rejected as a *new* individual when even the best match is
too weak. The attention weights double as an **explainability heatmap**.

```
image ─► crop+resize ─► tile into N×N patches (the BAG)
                              │
                  frozen backbone (MegaDescriptor)   ← NOT trained
                              │  per-patch embeddings
                              ▼
              gated-attention MIL pooling + projection ← the ONLY trained part
                              │  L2-normalised identity embedding
            ┌──────────── TRAIN ────────────┐   ┌────────── INFERENCE ──────────┐
            │ ArcFace margin loss over the  │   │ gallery = enrolled embeddings  │
            │ training identities           │   │ cosine top-k; max<τ ⇒ unknown  │
            └───────────────────────────────┘   │ attention weights ⇒ heatmap    │
                                                 └────────────────────────────────┘
```

**Why this design**
- **Frozen backbone + tiny head** → trains on CPU, data-efficient (reuses pretrained representations).
- **Attention MIL** → focuses on the few discriminative patches automatically, and yields a visual explanation for free.
- **Open-set retrieval** → enroll new individuals by appending a vector; no retraining, unlike a closed-set classifier.

> **Training** teaches the model *how to turn an image into a discriminative embedding* for a domain (e.g. turtles). **Enrolling** registers the *specific individuals* you want to recognise. You can enroll/identify individuals the model never saw in training — but only within the **domain it was trained on**.

---

## Project layout
```
ml/                         # Part 1 — ML core
  config.py                 # Pydantic settings + enums (backbone / pooling / patch resolution)
  data/                     # dataset access + catalogue, open-set split, patch tiling
  features/                 # frozen backbone + on-disk patch-embedding cache
  models/                   # GatedAttentionMIL head + (train-only) ArcFace head
  training/                 # cached-feature dataset, loss, train loop
  inference/                # embedder, gallery, identifier, explain, model registry
  eval/                     # rank-1/5, mAP, open-set AUROC + driver
api/                        # Part 2 — FastAPI
  main.py                   # app, lifespan, CORS, static-UI mount, exception handlers
  service.py                # ReidService — multi-model façade over ml.inference
  jobs.py                   # background JobManager (single-flight, cancel, SSE events)
  routers/                  # health, models, datasets, training, jobs, enroll, identify, gallery
  schemas.py                # Pydantic request/response DTOs
web/                        # Part 3 — browser client (index.html, css/, js/)
scripts/                    # CLI entrypoints (download / precompute / train / evaluate / demo)
tests/                      # test_core_logic.py, test_api.py (runnable, no pytest needed)
Dockerfile, docker-compose.yml, requirements.txt, requirements-mac.txt, sample.env
```
The API depends **only** on `ml/inference/` — the stable, model-internals-free seam.

---

## Quick start (Docker)

Reproducible on any machine with Docker — no Python/torch/CUDA/lzma setup.

```bash
docker compose build            # build the CPU image (one time)
docker compose up api           # API + Web UI on http://localhost:8000
```

Open **http://localhost:8000/** for the UI, or **http://localhost:8000/docs** for the
interactive OpenAPI schema.

On a fresh checkout there's no trained model yet, so the UI shows a "no trained model"
banner. Use the **Settings** tab (or the CLI below) to download a dataset → precompute
→ train, then **Seed** and **Identify**.

---

## The workflow (download → precompute → train → identify)

Every model is produced by the same pipeline. You can run it from the **Settings UI**
(background jobs, progress streamed live) or the **CLI**.

| Step | What it does | UI | CLI |
|------|--------------|----|-----|
| 1. Download | Fetch a dataset via WildlifeDatasets | Settings → **Download** | `scripts.download_data` |
| 2. Precompute | Cache frozen-backbone patch embeddings (slow, once) | Settings → **Precompute** | `scripts.precompute_features` |
| 3. Train | Train the MIL head on the cached features | Settings → **Train** | `scripts.train` |
| 4. Enroll / Seed | Register individuals into the gallery | Enroll tab / Gallery → **Seed** | `scripts.demo_identify` |
| 5. Identify | Match a query → individual or "unknown" | Identify tab | `scripts.demo_identify` |
| (eval) | rank-1/5, mAP, open-set AUROC report | — | `scripts.evaluate` |

CLI form (prefix any step with `docker compose run --rm reid`):
```bash
docker compose run --rm reid python -m scripts.download_data
docker compose run --rm reid python -m scripts.precompute_features
docker compose run --rm reid python -m scripts.train
docker compose run --rm reid python -m scripts.evaluate
docker compose run --rm reid python -m scripts.demo_identify
```
Datasets and trained artifacts land in the bind-mounted `data/` and `artifacts/` and
persist across runs. Training **requires** a precomputed dataset — the API returns
`422` (and the UI only lists precomputed datasets) otherwise.

Sample API calls:
```bash
curl localhost:8000/health
curl -F "individual_id=turtle_17" -F "files=@a.jpg" -F "files=@b.jpg" localhost:8000/enroll
curl -F "file=@query.jpg" localhost:8000/identify
curl -F "file=@query.jpg" localhost:8000/explain --output heatmap.png
```

---

## Web UI

A single page (plain HTML/CSS/JS, no build step) with four tabs plus a model picker
and a live job indicator:

- **Identify** — upload a query image → ranked candidates, the `unknown` verdict, and the attention heatmap overlay. Has a frontend **Reset**.
- **Enroll** — register an individual from one or more reference images (with thumbnails). Has a frontend **Reset**.
- **Gallery** — list enrolled individuals (latest first), **Delete** one, **Reset** the gallery, or **Seed** from the trained dataset.
- **Settings** — download datasets (with downloaded/precomputed badges), precompute features, and train new models.

The **model picker** scopes every action to the selected model; the **job indicator**
shows the running background job's progress with a **Cancel** button.

---

## Multiple models / species

One running app can serve several trained models (one per species/domain) — a model
only re-identifies the domain it was trained on. Each model is namespaced by
`REID_MODEL_NAME` under `artifacts/models/<name>/` with its **own checkpoint and
gallery**:

```bash
REID_MODEL_NAME=turtles REID_DATASET=SeaTurtleIDHeads docker compose run --rm reid python -m scripts.train
REID_MODEL_NAME=pandas  REID_DATASET=IPanda50         docker compose run --rm reid python -m scripts.train
```
(each dataset must be downloaded + precomputed first). Then `GET /models` lists both,
the UI shows a model dropdown, and every API call takes an optional `model` field
(defaults to `REID_MODEL_NAME`). **Galleries are isolated** per model; models load
lazily on first use and stay resident. A legacy single-model checkpoint at
`artifacts/mil_head.pt` is surfaced as model `default`.

---

## API reference

| Method & path | Purpose |
|---|---|
| `GET /health` | Liveness + how many models are trained |
| `GET /models` | List models (name, ready, dataset, backbone, enrolled count) |
| `POST /identify` | Query image → ranked candidates + `is_unknown` (+`model`) |
| `POST /explain` | Query image → attention heatmap **PNG** |
| `POST /enroll` | Enroll an individual from images (multipart, `+model`) |
| `GET /individuals` | List enrolled individuals for a model |
| `DELETE /individuals/{id}` | Remove one enrolled individual |
| `POST /gallery/reset` | Clear a model's gallery |
| `POST /gallery/seed` | **Job** — enroll the dataset's gallery split |
| `GET /datasets` | Curated datasets + `downloaded`/`precomputed` flags |
| `POST /datasets/{name}/download` | **Job** — download a dataset |
| `POST /datasets/{name}/precompute` | **Job** — build the feature cache |
| `POST /train` | **Job** — train `{model_name, dataset}` |
| `GET /jobs/active` | The running job (busy indicator) or null |
| `GET /jobs/{id}` | Job status (poll) |
| `POST /jobs/{id}/cancel` | Request cooperative cancellation |
| `GET /jobs/{id}/events` | **SSE** progress stream |
| `GET /` , `GET /docs` | Web UI · OpenAPI docs |

Status codes: `404` unknown model/dataset/job · `409` a job is already running or cancel-after-finish · `422` dataset not downloaded/precomputed · `503` selected model not trained/loadable · `415`/`413` bad/oversized upload.

---

## Background jobs & SSE

Download, precompute, train, and seed are slow, so they run as **background jobs**:
the API returns `202 Accepted` with a `job_id`, and the browser streams progress over
**Server-Sent Events** (`GET /jobs/{id}/events`).

- **Single-flight** — only one job runs at a time; a second start is rejected with `409` (not queued). The UI disables job buttons and shows the running job while one is active.
- **Cancellable** — jobs check a flag at loop checkpoints and stop cleanly (`cancelled`); no partial model/gallery is persisted. A dataset download already in flight can't be interrupted.
- **In-memory, single worker** — the job registry lives in one uvicorn process; a restart forgets jobs (fine for a demo).

---

## Configuration

All ML settings live in [`ml/config.py`](ml/config.py) and are overridable via
`REID_*` environment variables; API settings use `REID_API_*`. Copy
[`sample.env`](sample.env) → `.env` (auto-loaded by compose) or pass
`docker compose run -e REID_...`.

Common knobs:

| Variable | Default | Notes |
|---|---|---|
| `REID_MODEL_NAME` | `default` | Namespaces the model's checkpoint + gallery |
| `REID_DATASET` | `SeaTurtleIDHeads` | Any WildlifeDatasets class (UI offers a curated subset) |
| `REID_BACKBONE` | `hf-hub:BVRA/MegaDescriptor-T-224` | or `vit_small_patch16_224`, `resnet50` (offline) |
| `REID_POOLING` | `GATED_ATTENTION` | or `MEAN` (ablation baseline) |
| `REID_PATCH_RESOLUTION` | `UPSAMPLED` | or `NATIVE` (see below) |
| `REID_EPOCHS` / `REID_BATCH_SIZE` | `30` / `32` | training |
| `REID_UNKNOWN_THRESHOLD` | `0.5` | cosine score below which a query is "unknown" |
| `REID_API_CORS_ORIGINS` / `REID_API_MAX_UPLOAD_MB` | `*` / `15` | API |

**Curated UI datasets:** SeaTurtleIDHeads, IPanda50, MacaqueFaces, AAUZebraFish,
CatIndividualImages, DogFaceNet. (The CLI accepts any WildlifeDatasets class name.)

**Patch resolution** — each patch must reach the backbone's input size (224):
- `UPSAMPLED` (default) — resize the animal to 224, tile into 56px patches, let the backbone upsample. Cheap; low-detail patches.
- `NATIVE` — resize to `patch_grid × 224` so each tile is natively 224 with genuine fine detail. Higher fidelity, more compute.

Re-run **precompute** after changing the backbone or any patch setting — cached
embeddings differ, and the precompute marker invalidates automatically.

**Ablation** (what attention buys): train + evaluate with `REID_POOLING=MEAN` and
compare rank-1/mAP against the default gated-attention model.

---

## Development

- **Hot reload** — `ml/`, `api/`, `web/`, `scripts/`, `tests/` are bind-mounted into the container. Python changes reload via uvicorn `--reload`; web edits just need a browser refresh. **Rebuild only when dependencies change.**
- **Tests** (no `pytest` dependency — runnable modules):
  ```bash
  docker compose run --rm reid python -m tests.test_api          # API layer (stubs, fast)
  docker compose run --rm reid python -m tests.test_core_logic   # ML core logic (synthetic data)
  ```
- **Native (Intel macOS)** — the container is the supported path; for a native install use the mac pins (torch from the PyTorch CPU index, not PyPI):
  ```bash
  pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
  pip install -r requirements-mac.txt
  ```
  Two requirements files exist on purpose: [`requirements.txt`](requirements.txt) is the Linux/CPU container set (modern torch + NumPy 2, exact pins); [`requirements-mac.txt`](requirements-mac.txt) carries Intel-mac caps (its last torch is NumPy-1.x only).

---

## Design notes & caveats

- **Hugging Face** is used only to download the **MegaDescriptor** backbone weights (cached under `artifacts/hf_cache`); set `REID_BACKBONE=resnet50` for a fully offline run.
- **`wildlife-datasets` only** — we do *not* use `wildlife-tools` (it hard-requires `faiss-gpu`); similarity/retrieval/metrics are implemented natively. All SDK calls are isolated in [`ml/data/dataset.py`](ml/data/dataset.py).
- **Domain match matters** — a turtle model will not reliably tell two human faces apart; train a model per domain and pick it in the UI.
- **Feature cache** is keyed by `image_id` and shared across models trained on the same dataset + patch settings; a per-dataset marker records precompute completion for the training guard.

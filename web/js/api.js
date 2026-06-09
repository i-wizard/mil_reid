/*
 * api.js — the single place that knows the API's endpoints and shapes.
 *
 * Every call uses a relative URL because the page is served from the same origin
 * as the API (FastAPI StaticFiles), so no base URL or CORS handling is needed.
 * Errors are normalised into a thrown Error carrying the server's `detail`
 * message, so the UI layer (app.js) can show one consistent toast regardless of
 * which endpoint failed.
 */

const Api = (() => {
  /**
   * Parse a failed Response into a thrown Error using the API's `detail` field.
   * Centralised so a 503/413/415/422 from any endpoint surfaces the same way.
   */
  async function _raise(response) {
    let detail = `Request failed (${response.status})`;
    try {
      const body = await response.json();
      if (body && body.detail) detail = body.detail;
    } catch (_) {
      /* non-JSON error body — keep the generic message */
    }
    throw new Error(detail);
  }

  /** Append a model name to a FormData body when one is selected (omit ⇒ server default). */
  function _withModel(form, model) {
    if (model) form.append("model", model);
    return form;
  }

  /** Build a `?model=` query suffix for GET/DELETE when a model is selected. */
  function _modelQuery(model) {
    return model ? `?model=${encodeURIComponent(model)}` : "";
  }

  /** GET /health — global liveness (models_available, default_model). */
  async function health() {
    const res = await fetch("/health");
    if (!res.ok) await _raise(res);
    return res.json();
  }

  /** GET /models — the catalogue backing the model picker. */
  async function listModels() {
    const res = await fetch("/models");
    if (!res.ok) await _raise(res);
    return res.json();
  }

  /** POST /identify — query image (+model) → {model, is_unknown, candidates, grid, attention_grid}. */
  async function identify(file, model) {
    const form = _withModel(new FormData(), model);
    form.append("file", file);
    const res = await fetch("/identify", { method: "POST", body: form });
    if (!res.ok) await _raise(res);
    return res.json();
  }

  /** POST /explain — query image (+model) → heatmap PNG, returned as an object URL for <img>. */
  async function explainObjectUrl(file, model) {
    const form = _withModel(new FormData(), model);
    form.append("file", file);
    const res = await fetch("/explain", { method: "POST", body: form });
    if (!res.ok) await _raise(res);
    const blob = await res.blob();
    return URL.createObjectURL(blob);
  }

  /** POST /enroll — individual_id + files (+model) → EnrollResponse. */
  async function enroll(individualId, files, model) {
    const form = _withModel(new FormData(), model);
    form.append("individual_id", individualId);
    for (const f of files) form.append("files", f);
    const res = await fetch("/enroll", { method: "POST", body: form });
    if (!res.ok) await _raise(res);
    return res.json();
  }

  /** GET /individuals?model= — current roster for the selected model. */
  async function listIndividuals(model) {
    const res = await fetch(`/individuals${_modelQuery(model)}`);
    if (!res.ok) await _raise(res);
    return res.json();
  }

  /** DELETE /individuals/{id}?model= — remove one enrolled individual from the selected model. */
  async function deleteIndividual(individualId, model) {
    const url = `/individuals/${encodeURIComponent(individualId)}${_modelQuery(model)}`;
    const res = await fetch(url, { method: "DELETE" });
    if (!res.ok) await _raise(res);
    return res.json();
  }

  /** POST /gallery/reset — clear the selected model's gallery. */
  async function resetGallery(model) {
    const res = await fetch("/gallery/reset", { method: "POST", body: _withModel(new FormData(), model) });
    if (!res.ok) await _raise(res);
    return res.json();
  }

  /** POST /gallery/seed — start a background seed job → {job_id, kind}. */
  async function seed(model) {
    const res = await fetch("/gallery/seed", { method: "POST", body: _withModel(new FormData(), model) });
    if (!res.ok) await _raise(res);
    return res.json();
  }

  // --- Settings: datasets + background jobs ---

  /** GET /datasets — curated catalogue + downloaded flags. */
  async function listDatasets() {
    const res = await fetch("/datasets");
    if (!res.ok) await _raise(res);
    return res.json();
  }

  /** POST /datasets/{name}/download — start a download job → {job_id}. */
  async function downloadDataset(name) {
    const res = await fetch(`/datasets/${encodeURIComponent(name)}/download`, { method: "POST" });
    if (!res.ok) await _raise(res);
    return res.json();
  }

  /** POST /datasets/{name}/precompute — start a precompute job → {job_id}. */
  async function precomputeDataset(name) {
    const res = await fetch(`/datasets/${encodeURIComponent(name)}/precompute`, { method: "POST" });
    if (!res.ok) await _raise(res);
    return res.json();
  }

  /** POST /train — start a training job → {job_id}. */
  async function train(modelName, dataset) {
    const res = await fetch("/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model_name: modelName, dataset: dataset }),
    });
    if (!res.ok) await _raise(res);
    return res.json();
  }

  /** GET /jobs/active — the running job (for the busy indicator) or null. */
  async function activeJob() {
    const res = await fetch("/jobs/active");
    if (!res.ok) await _raise(res);
    return res.json();
  }

  /** POST /jobs/{id}/cancel — request cooperative cancellation. */
  async function cancelJob(jobId) {
    const res = await fetch(`/jobs/${encodeURIComponent(jobId)}/cancel`, { method: "POST" });
    if (!res.ok) await _raise(res);
    return res.json();
  }

  /**
   * Subscribe to a job's SSE progress stream.
   *
   * Wraps EventSource (GET, same-origin). Calls onEvent(parsedEvent) for each
   * progress event and closes automatically on a terminal status. Returns the
   * EventSource so the caller can close it early if needed.
   */
  function subscribeJob(jobId, onEvent) {
    const source = new EventSource(`/jobs/${encodeURIComponent(jobId)}/events`);
    source.onmessage = (msg) => {
      let event;
      try {
        event = JSON.parse(msg.data);
      } catch (_) {
        return;
      }
      onEvent(event);
      if (["succeeded", "failed", "cancelled"].includes(event.status)) source.close();
    };
    source.onerror = () => source.close(); // stream ends → browser fires error; just close
    return source;
  }

  return {
    health, listModels, identify, explainObjectUrl, enroll,
    listIndividuals, deleteIndividual, resetGallery, seed,
    listDatasets, downloadDataset, precomputeDataset, train, activeJob, cancelJob, subscribeJob,
  };
})();

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

  /** GET /health — readiness + gallery size for the status banner. */
  async function health() {
    const res = await fetch("/health");
    if (!res.ok) await _raise(res);
    return res.json();
  }

  /** POST /identify — query image → {is_unknown, candidates, grid, attention_grid}. */
  async function identify(file) {
    const form = new FormData();
    form.append("file", file);
    const res = await fetch("/identify", { method: "POST", body: form });
    if (!res.ok) await _raise(res);
    return res.json();
  }

  /** POST /explain — query image → heatmap PNG, returned as an object URL for <img>. */
  async function explainObjectUrl(file) {
    const form = new FormData();
    form.append("file", file);
    const res = await fetch("/explain", { method: "POST", body: form });
    if (!res.ok) await _raise(res);
    const blob = await res.blob();
    return URL.createObjectURL(blob);
  }

  /** POST /enroll — individual_id + one or more files → EnrollResponse. */
  async function enroll(individualId, files) {
    const form = new FormData();
    form.append("individual_id", individualId);
    for (const f of files) form.append("files", f);
    const res = await fetch("/enroll", { method: "POST", body: form });
    if (!res.ok) await _raise(res);
    return res.json();
  }

  /** GET /individuals — current roster. */
  async function listIndividuals() {
    const res = await fetch("/individuals");
    if (!res.ok) await _raise(res);
    return res.json();
  }

  /** DELETE /individuals/{id} — remove one enrolled individual. */
  async function deleteIndividual(individualId) {
    const res = await fetch(`/individuals/${encodeURIComponent(individualId)}`, { method: "DELETE" });
    if (!res.ok) await _raise(res);
    return res.json();
  }

  /** POST /gallery/reset — clear the gallery. */
  async function resetGallery() {
    const res = await fetch("/gallery/reset", { method: "POST" });
    if (!res.ok) await _raise(res);
    return res.json();
  }

  /** POST /gallery/seed — enroll the trained dataset's gallery split. */
  async function seed() {
    const res = await fetch("/gallery/seed", { method: "POST" });
    if (!res.ok) await _raise(res);
    return res.json();
  }

  return { health, identify, explainObjectUrl, enroll, listIndividuals, deleteIndividual, resetGallery, seed };
})();

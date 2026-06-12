/*
 * app.js — DOM wiring for the re-identification demo.
 *
 * Talks to the backend only through Api (api.js) and keeps the interaction model
 * simple: tabs toggle panels; each action runs through `withBusy` so buttons
 * disable + spin during a request and errors always become a toast. The status
 * banner (from /health) gates Identify/Enroll so the user gets a clear "train the
 * model first" instead of a raw 503.
 */

(() => {
  "use strict";

  const $ = (sel) => document.querySelector(sel);
  let modelReady = false;
  let selectedModel = null; // the model every action targets (null until /models loads)
  let modelsById = {}; // name -> ModelInfo, for banner/readiness lookups

  // --- Toast -------------------------------------------------------------
  let toastTimer = null;
  function toast(message, kind = "success") {
    const el = $("#toast");
    el.textContent = message;
    el.className = `toast show toast--${kind}`;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => (el.className = "toast"), 3500);
  }

  /**
   * Run an async action with button busy-state + uniform error handling.
   * Every endpoint call goes through here so the UX (spinner, disable, error
   * toast) is identical everywhere and never duplicated per handler.
   */
  async function withBusy(button, fn) {
    if (button) {
      button.disabled = true;
      button.classList.add("is-busy");
    }
    try {
      return await fn();
    } catch (err) {
      toast(err.message || "Something went wrong", "error");
      return null;
    } finally {
      if (button) {
        button.disabled = false;
        button.classList.remove("is-busy");
      }
    }
  }

  // --- Background jobs (single-flight) + SSE -----------------------------
  let jobRunning = false;
  let currentJobId = null;

  /** Enable/disable all job-start buttons; the running job's own Cancel stays active. */
  function applyJobGating() {
    document.querySelectorAll(".js-job-btn").forEach((b) => (b.disabled = jobRunning));
  }

  function beginJobUI(jobId, label) {
    jobRunning = true;
    currentJobId = jobId;
    $("#job-indicator").hidden = false;
    $("#job-indicator-text").textContent = label + "…";
    $("#job-bar-fill").style.width = "0%";
    applyJobGating();
  }

  function updateJobUI(event) {
    if (event.message) $("#job-indicator-text").textContent = event.message;
    if (typeof event.progress === "number") $("#job-bar-fill").style.width = `${Math.round(event.progress * 100)}%`;
  }

  function endJobUI() {
    jobRunning = false;
    currentJobId = null;
    $("#job-indicator").hidden = true;
    applyJobGating();
  }

  /**
   * Start a background job and drive the UI from its SSE stream.
   * `startFn` returns the 202 body ({job_id}); a 409 (busy) surfaces as a toast.
   * `onSuccess` runs on the terminal success event (to refresh the affected view).
   */
  async function runJob(label, startFn, onSuccess) {
    let started;
    try {
      started = await startFn();
    } catch (err) {
      toast(err.message || "Could not start job", "error"); // includes 409 "a job is running"
      return;
    }
    beginJobUI(started.job_id, label);
    Api.subscribeJob(started.job_id, (event) => {
      updateJobUI(event);
      if (event.status === "succeeded") {
        endJobUI();
        toast(`${label} complete`);
        if (onSuccess) onSuccess(event);
      } else if (event.status === "failed") {
        endJobUI();
        toast(`${label} failed: ${event.error || "unknown error"}`, "error");
      } else if (event.status === "cancelled") {
        endJobUI();
        toast(`${label} cancelled`);
      }
    });
  }

  function initJobIndicator() {
    $("#job-cancel-btn").addEventListener("click", () => {
      if (currentJobId) Api.cancelJob(currentJobId).catch((err) => toast(err.message, "error"));
    });
  }

  /** On load, re-attach to a job already running (e.g. after a page refresh). */
  async function resumeActiveJob() {
    const job = await Api.activeJob().catch(() => null);
    if (!job) return;
    beginJobUI(job.id, job.kind);
    Api.subscribeJob(job.id, (event) => {
      updateJobUI(event);
      if (["succeeded", "failed", "cancelled"].includes(event.status)) {
        endJobUI();
        refreshModels();
        refreshDatasets();
      }
    });
  }

  // --- Tabs --------------------------------------------------------------
  function initTabs() {
    document.querySelectorAll(".tab").forEach((tab) => {
      tab.addEventListener("click", () => {
        document.querySelectorAll(".tab").forEach((t) => t.classList.remove("is-active"));
        document.querySelectorAll(".panel").forEach((p) => p.classList.remove("is-active"));
        tab.classList.add("is-active");
        $(`#panel-${tab.dataset.tab}`).classList.add("is-active");
        if (tab.dataset.tab === "gallery") refreshGallery();
        if (tab.dataset.tab === "settings") refreshDatasets();
      });
    });
  }

  // --- Model picker + status banner --------------------------------------
  function setBanner(kind, text) {
    const banner = $("#status-banner");
    banner.className = `banner banner--${kind}`;
    banner.textContent = text;
  }

  /** Enable/disable model-dependent actions based on the selected model's readiness. */
  function gateActions() {
    document.querySelectorAll('[data-action="identify"], [data-action="enroll"], #seed-btn').forEach((b) => {
      b.disabled = !modelReady;
    });
  }

  /**
   * Fetch /models, (re)populate the picker, and reflect the selection's readiness.
   * Called on load and after any gallery change so counts/readiness stay current.
   */
  async function refreshModels() {
    const select = $("#model-select");
    let data;
    try {
      data = await Api.listModels();
    } catch (err) {
      setBanner("notready", `API unreachable: ${err.message}`);
      modelReady = false;
      gateActions();
      return;
    }

    modelsById = {};
    data.models.forEach((m) => (modelsById[m.name] = m));
    const names = data.models.map((m) => m.name);

    // Keep the current selection if it still exists; else fall back to the default.
    if (!selectedModel || !names.includes(selectedModel)) {
      selectedModel = data.default_model || names[0] || null;
    }

    select.innerHTML = "";
    if (names.length === 0) {
      const opt = document.createElement("option");
      opt.value = "";
      opt.textContent = "(no models trained)";
      select.appendChild(opt);
      select.disabled = true;
    } else {
      select.disabled = false;
      data.models.forEach((m) => {
        const opt = document.createElement("option");
        opt.value = m.name;
        opt.textContent = m.ready ? m.name : `${m.name} (not ready)`;
        if (m.name === selectedModel) opt.selected = true;
        select.appendChild(opt);
      });
    }
    applySelectedModel();
  }

  /** Update the banner + action gating for whichever model is currently selected. */
  function applySelectedModel() {
    const m = selectedModel ? modelsById[selectedModel] : null;
    modelReady = !!(m && m.ready);
    if (!m) {
      setBanner("notready", "No trained model — run training (scripts.train); it appears here automatically.");
    } else if (m.ready) {
      setBanner("ready", `Model '${m.name}' ready · ${m.dataset || "?"} · ${m.num_individuals} enrolled`);
    } else {
      setBanner("notready", `Model '${m.name}' is not ready.`);
    }
    gateActions();
  }

  function initModelPicker() {
    $("#model-select").addEventListener("change", (e) => {
      selectedModel = e.target.value || null;
      applySelectedModel();
      refreshGallery();
    });
  }

  // --- Identify ----------------------------------------------------------
  function initIdentify() {
    const form = $("#identify-form");
    const fileInput = $("#identify-file");
    const preview = $("#identify-preview");

    fileInput.addEventListener("change", () => {
      const file = fileInput.files[0];
      if (file) {
        preview.src = URL.createObjectURL(file);
        preview.classList.add("has-image");
      }
    });

    form.addEventListener("submit", (e) => {
      e.preventDefault();
      const file = fileInput.files[0];
      if (!file) return;
      const button = form.querySelector('[data-action="identify"]');

      withBusy(button, async () => {
        // Identification and heatmap run together so the result and the "why"
        // appear at once; both are independent calls on the same image.
        const [result, heatmapUrl] = await Promise.all([
          Api.identify(file, selectedModel),
          Api.explainObjectUrl(file, selectedModel).catch(() => null), // heatmap is best-effort
        ]);
        if (!result) return;
        renderIdentify(result, heatmapUrl);
      });
    });
  }

  function renderIdentify(result, heatmapUrl) {
    const verdict = $("#identify-verdict");
    const list = $("#identify-candidates");
    const heatmap = $("#identify-heatmap");

    if (heatmapUrl) {
      heatmap.src = heatmapUrl;
      heatmap.classList.add("has-image");
    }

    const top = result.candidates[0];
    if (result.is_unknown || !top) {
      verdict.className = "verdict verdict--unknown";
      verdict.textContent = "❓ Unknown individual (no confident match)";
    } else {
      verdict.className = "verdict verdict--known";
      verdict.textContent = `✅ ${top.individual_id} (score ${top.score.toFixed(3)})`;
    }

    list.innerHTML = "";
    result.candidates.forEach((c, i) => {
      const pct = Math.max(0, Math.min(100, c.score * 100));
      const li = document.createElement("li");
      li.className = "candidate" + (i === 0 ? " is-top" : "");
      li.innerHTML =
        `<div class="candidate__row"><span>${escapeHtml(c.individual_id)}</span><span>${c.score.toFixed(3)}</span></div>` +
        `<div class="candidate__bar"><div class="candidate__fill" style="width:${pct}%"></div></div>`;
      list.appendChild(li);
    });
  }

  // --- Enroll ------------------------------------------------------------
  function initEnroll() {
    const form = $("#enroll-form");
    const idInput = $("#enroll-id");
    const filesInput = $("#enroll-files");
    const thumbs = $("#enroll-thumbs");

    filesInput.addEventListener("change", () => {
      thumbs.innerHTML = "";
      [...filesInput.files].forEach((f) => {
        const img = document.createElement("img");
        img.src = URL.createObjectURL(f);
        thumbs.appendChild(img);
      });
    });

    form.addEventListener("submit", (e) => {
      e.preventDefault();
      const id = idInput.value.trim();
      const files = filesInput.files;
      if (!id || files.length === 0) return;
      const button = form.querySelector('[data-action="enroll"]');

      withBusy(button, async () => {
        const res = await Api.enroll(id, files, selectedModel);
        if (!res) return;
        toast(`Enrolled '${res.individual_id}' into '${res.model}' (${res.images_enrolled} image(s)); ${res.total_individuals} total`);
        form.reset();
        thumbs.innerHTML = "";
        refreshModels();
      });
    });
  }

  // --- Gallery -----------------------------------------------------------
  async function refreshGallery() {
    const list = $("#gallery-list");
    const data = await Api.listIndividuals(selectedModel).catch((err) => {
      toast(err.message, "error");
      return null;
    });
    if (!data) return;

    list.innerHTML = "";
    if (data.count === 0) {
      const li = document.createElement("li");
      li.className = "empty";
      li.textContent = "No individuals enrolled yet.";
      list.appendChild(li);
      return;
    }
    // Latest-first: the gallery stores ids in enrollment order, so reverse it.
    data.individuals.slice().reverse().forEach((id) => {
      const li = document.createElement("li");
      const name = document.createElement("span");
      name.textContent = id;
      const del = document.createElement("button");
      del.className = "btn btn--danger";
      del.textContent = "Delete";
      del.addEventListener("click", () =>
        withBusy(del, async () => {
          await Api.deleteIndividual(id, selectedModel);
          toast(`Removed '${id}'`);
          await refreshGallery();
          refreshModels();
        })
      );
      li.appendChild(name);
      li.appendChild(del);
      list.appendChild(li);
    });
  }

  function initGalleryActions() {
    $("#refresh-btn").addEventListener("click", (e) => withBusy(e.target, refreshGallery));

    // Seeding embeds every reference image, so it runs as a background job (SSE).
    $("#seed-btn").addEventListener("click", () =>
      runJob("Seed", () => Api.seed(selectedModel), () => {
        refreshGallery();
        refreshModels();
      })
    );

    $("#reset-btn").addEventListener("click", (e) => {
      if (!confirm("Remove ALL enrolled individuals?")) return;
      withBusy(e.target, async () => {
        await Api.resetGallery(selectedModel);
        toast("Gallery cleared");
        await refreshGallery();
        refreshModels();
      });
    });
  }

  // --- Settings: datasets + training -------------------------------------
  /** Render the dataset catalogue (download/precompute) and populate the train select. */
  async function refreshDatasets() {
    const listEl = $("#dataset-list");
    const trainSelect = $("#train-dataset");
    const data = await Api.listDatasets().catch((err) => {
      toast(err.message, "error");
      return null;
    });
    if (!data) return;

    listEl.innerHTML = "";
    const trainable = []; // downloaded AND precomputed → eligible to train
    data.datasets.forEach((d) => {
      const li = document.createElement("li");
      const name = document.createElement("span");
      name.textContent = d.name;

      const badges = document.createElement("span");
      badges.className = "settings-badges";
      const dlBadge = document.createElement("span");
      dlBadge.className = "badge " + (d.downloaded ? "badge--ok" : "badge--muted");
      dlBadge.textContent = d.downloaded ? "downloaded" : "not downloaded";
      badges.appendChild(dlBadge);
      if (d.downloaded) {
        const pcBadge = document.createElement("span");
        pcBadge.className = "badge " + (d.precomputed ? "badge--ok" : "badge--muted");
        pcBadge.textContent = d.precomputed ? "precomputed" : "not precomputed";
        badges.appendChild(pcBadge);
      }

      const actions = document.createElement("span");
      actions.className = "settings-actions";
      const dl = document.createElement("button");
      dl.className = "btn js-job-btn";
      dl.textContent = "Download";
      dl.disabled = jobRunning;
      dl.addEventListener("click", () =>
        runJob(`Download ${d.name}`, () => Api.downloadDataset(d.name), () => refreshDatasets())
      );
      const pc = document.createElement("button");
      pc.className = "btn js-job-btn";
      pc.textContent = "Precompute";
      pc.disabled = jobRunning || !d.downloaded;
      // Refresh the list on success so the 'precomputed' badge flips on.
      pc.addEventListener("click", () => runJob(`Precompute ${d.name}`, () => Api.precomputeDataset(d.name), () => refreshDatasets()));
      actions.append(dl, pc);

      li.append(name, badges, actions);
      listEl.appendChild(li);
      if (d.downloaded && d.precomputed) trainable.push(d.name);
    });

    // Only precomputed datasets can be trained (the server enforces this too).
    trainSelect.innerHTML = "";
    if (trainable.length === 0) {
      const opt = document.createElement("option");
      opt.value = "";
      opt.textContent = "(precompute a dataset first)";
      trainSelect.appendChild(opt);
      trainSelect.disabled = true;
    } else {
      trainSelect.disabled = false;
      trainable.forEach((n) => {
        const opt = document.createElement("option");
        opt.value = n;
        opt.textContent = n;
        trainSelect.appendChild(opt);
      });
    }
  }

  function initTrainForm() {
    $("#train-form").addEventListener("submit", (e) => {
      e.preventDefault();
      const name = $("#train-model-name").value.trim();
      const dataset = $("#train-dataset").value;
      if (!name || !dataset) return;
      runJob(`Train ${name}`, () => Api.train(name, dataset), () => {
        $("#train-form").reset();
        refreshModels(); // the new model now appears in the picker
      });
    });
  }

  // --- Frontend-only Reset buttons ---------------------------------------
  function initResetButtons() {
    $("#identify-reset").addEventListener("click", () => {
      $("#identify-form").reset();
      ["#identify-preview", "#identify-heatmap"].forEach((sel) => {
        const img = $(sel);
        img.removeAttribute("src");
        img.classList.remove("has-image");
      });
      const verdict = $("#identify-verdict");
      verdict.textContent = "";
      verdict.className = "verdict";
      $("#identify-candidates").innerHTML = "";
    });
    $("#enroll-reset").addEventListener("click", () => {
      $("#enroll-form").reset();
      $("#enroll-thumbs").innerHTML = "";
    });
  }

  /** Escape user-supplied ids before injecting into innerHTML (defensive XSS guard). */
  function escapeHtml(str) {
    return String(str).replace(/[&<>"']/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
    );
  }

  // --- Bootstrap ---------------------------------------------------------
  document.addEventListener("DOMContentLoaded", () => {
    initTabs();
    initModelPicker();
    initJobIndicator();
    initIdentify();
    initEnroll();
    initGalleryActions();
    initTrainForm();
    initResetButtons();
    refreshModels();
    resumeActiveJob(); // re-attach the indicator if a job is already running
  });
})();

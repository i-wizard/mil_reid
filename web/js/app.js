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

  // --- Tabs --------------------------------------------------------------
  function initTabs() {
    document.querySelectorAll(".tab").forEach((tab) => {
      tab.addEventListener("click", () => {
        document.querySelectorAll(".tab").forEach((t) => t.classList.remove("is-active"));
        document.querySelectorAll(".panel").forEach((p) => p.classList.remove("is-active"));
        tab.classList.add("is-active");
        $(`#panel-${tab.dataset.tab}`).classList.add("is-active");
        if (tab.dataset.tab === "gallery") refreshGallery();
      });
    });
  }

  // --- Status banner -----------------------------------------------------
  async function refreshStatus() {
    const banner = $("#status-banner");
    try {
      const h = await Api.health();
      modelReady = h.model_ready;
      if (h.model_ready) {
        banner.className = "banner banner--ready";
        banner.textContent = `Model ready · ${h.dataset} · ${h.num_individuals} enrolled`;
      } else {
        banner.className = "banner banner--notready";
        banner.textContent = "No trained model — run training (scripts.train), then restart the API.";
      }
    } catch (err) {
      banner.className = "banner banner--notready";
      banner.textContent = `API unreachable: ${err.message}`;
      modelReady = false;
    }
    // Gate the model-dependent actions on readiness.
    document.querySelectorAll('[data-action="identify"], [data-action="enroll"], #seed-btn').forEach((b) => {
      b.disabled = !modelReady;
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
          Api.identify(file),
          Api.explainObjectUrl(file).catch(() => null), // heatmap is best-effort
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
        const res = await Api.enroll(id, files);
        if (!res) return;
        toast(`Enrolled '${res.individual_id}' (${res.images_enrolled} image(s)); ${res.total_individuals} total`);
        form.reset();
        thumbs.innerHTML = "";
        refreshStatus();
      });
    });
  }

  // --- Gallery -----------------------------------------------------------
  async function refreshGallery() {
    const list = $("#gallery-list");
    const data = await Api.listIndividuals().catch((err) => {
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
    data.individuals.forEach((id) => {
      const li = document.createElement("li");
      const name = document.createElement("span");
      name.textContent = id;
      const del = document.createElement("button");
      del.className = "btn btn--danger";
      del.textContent = "Delete";
      del.addEventListener("click", () =>
        withBusy(del, async () => {
          await Api.deleteIndividual(id);
          toast(`Removed '${id}'`);
          await refreshGallery();
          refreshStatus();
        })
      );
      li.appendChild(name);
      li.appendChild(del);
      list.appendChild(li);
    });
  }

  function initGalleryActions() {
    $("#refresh-btn").addEventListener("click", (e) => withBusy(e.target, refreshGallery));

    $("#seed-btn").addEventListener("click", (e) =>
      withBusy(e.target, async () => {
        const res = await Api.seed();
        if (!res) return;
        toast(`Seeded ${res.individuals_enrolled} individuals from the dataset`);
        await refreshGallery();
        refreshStatus();
      })
    );

    $("#reset-btn").addEventListener("click", (e) => {
      if (!confirm("Remove ALL enrolled individuals?")) return;
      withBusy(e.target, async () => {
        await Api.resetGallery();
        toast("Gallery cleared");
        await refreshGallery();
        refreshStatus();
      });
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
    initIdentify();
    initEnroll();
    initGalleryActions();
    refreshStatus();
  });
})();

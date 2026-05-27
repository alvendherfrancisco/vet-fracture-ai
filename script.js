// ── CONFIG ──────────────────────────────────────────────────────────────
const API_URL = "https://alvendherfrancisco-vetfractureai.hf.space"; // http://localhost:7860

// ── STATE ───────────────────────────────────────────────────────────────
let currentFile = null;
let lastResultImage = null;

// ── FILE SELECTED ────────────────────────────────────────────────────────
function onFileSelected(e) {
  const file = e.target.files[0];
  if (!file) return;
  currentFile = file;

  const reader = new FileReader();
  reader.onload = (ev) => {
    const img = document.getElementById("preview-img");
    img.src = ev.target.result;
    img.style.display = "block";
    img.style.animation = "none";
    requestAnimationFrame(() => {
      img.style.animation = "";
    });
    document.getElementById("upload-hint").style.display = "none";
    document.getElementById("upload-zone").classList.add("has-image");
  };
  reader.readAsDataURL(file);

  document.getElementById("clahe-btn").disabled = false;
  document.getElementById("detect-btn").disabled = false;
  setStatus("Image loaded — ready to detect.", "");
}

// ── SHARED FETCH HELPER ───────────────────────────────────────────────────
async function apiFetch(path, formData) {
  const res = await fetch(`${API_URL}${path}`, {
    method: "POST",
    body: formData,
  });

  if (res.status === 503) {
    return {
      error: "Model is still loading — please wait ~60 s and try again.",
    };
  }
  if (res.status === 502 || res.status === 504) {
    return {
      error: `Server unavailable (HTTP ${res.status}). The service may be waking up — please try again in a moment.`,
    };
  }

  const text = await res.text();
  if (!text) {
    return {
      error:
        "Server returned an empty response. The service may still be starting up — please try again in ~60 s.",
    };
  }

  try {
    return JSON.parse(text);
  } catch {
    return { error: `Could not parse server response: ${text.slice(0, 120)}` };
  }
}

// ── APPLY CLAHE ───────────────────────────────────────────────────────────
async function applyClahe() {
  if (!currentFile) return;
  setStatus(
    '<span class="spinner"></span>Applying CLAHE enhancement…',
    "loading",
  );

  const form = new FormData();
  form.append("file", currentFile);

  try {
    const data = await apiFetch("/clahe", form);
    if (data.error) {
      setStatus("Error: " + data.error, "error");
      return;
    }

    const img = document.getElementById("preview-img");
    img.style.opacity = "0";
    img.src = data.image;
    img.onload = () => {
      img.style.transition = "opacity 0.4s ease";
      img.style.opacity = "1";
    };

    setStatus(data.message, "success");

    const blob = await (await fetch(data.image)).blob();
    currentFile = new File([blob], "clahe_enhanced.jpg", {
      type: "image/jpeg",
    });
  } catch (err) {
    setStatus("Error applying CLAHE: " + err.message, "error");
  }
}

// ── DETECT ────────────────────────────────────────────────────────────────
async function runDetection() {
  if (!currentFile) return;
  setStatus(
    '<span class="spinner"></span>Running fracture detection…',
    "loading",
  );

  // Show skeleton loader
  const resultZone = document.getElementById("result-zone");
  resultZone.classList.remove("has-result");
  resultZone.innerHTML = `<div class="skeleton" style="width:100%;height:300px;"></div>`;

  const form = new FormData();
  form.append("file", currentFile);
  form.append("conf_32a", document.getElementById("conf-a").value);
  form.append("conf_32b", document.getElementById("conf-b").value);
  form.append("conf_32c", document.getElementById("conf-c").value);

  try {
    const data = await apiFetch("/predict", form);
    if (data.error) {
      resultZone.innerHTML = `<div class="result-placeholder"><div class="result-placeholder-icon"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18M9 21V9"/></svg></div>Detection results will appear here after analysis.</div>`;
      setStatus("Error: " + data.error, "error");
      return;
    }

    const imgEl = document.createElement("img");
    imgEl.src = data.image;
    imgEl.alt = "Result";
    imgEl.style.cssText =
      "max-width:100%;max-height:340px;object-fit:contain;border-radius:8px;";
    resultZone.innerHTML = "";
    resultZone.appendChild(imgEl);
    resultZone.classList.add("has-result");

    lastResultImage = data.image;
    const dlBtn = document.getElementById("download-btn");
    dlBtn.style.display = "flex";
    dlBtn.style.animation = "fadeUp 0.4s ease 0.3s both";

    renderDetections(data.detections, data.summary);
    setStatus(
      `Analysis complete — ${data.total} fracture(s) detected.`,
      "success",
    );
  } catch (err) {
    resultZone.innerHTML = `<div class="result-placeholder">Detection results will appear here after analysis.</div>`;
    setStatus("Error running detection: " + err.message, "error");
  }
}

// ── RENDER DETECTIONS ─────────────────────────────────────────────────────
function renderDetections(detections, summary) {
  const el = document.getElementById("detections-list");
  if (!detections || detections.length === 0) {
    el.innerHTML = `<div class="no-detections">No fractures detected above the confidence threshold.</div>`;
    return;
  }

  const badgeClass = (cls) =>
    cls.includes("A") ? "badge-a" : cls.includes("B") ? "badge-b" : "badge-c";

  const confBarColor = (cls) =>
    cls.includes("A") ? "#ffb830" : cls.includes("B") ? "#ff8042" : "#ff5757";

  el.innerHTML = `
    <div class="detections-header">Detected Fractures (${detections.length})</div>
    ${detections
      .map(
        (d, i) => `
      <div class="detection-item" style="animation-delay:${i * 0.08}s">
        <div class="detection-left">
          <span class="${badgeClass(d.class)}">${d.class}</span>
          <div>
            <div class="detection-confidence">${(d.confidence * 100).toFixed(1)}% conf.</div>
            <div class="conf-bar-wrap">
              <div class="conf-bar" style="width:${(d.confidence * 100).toFixed(1)}%;background:${confBarColor(d.class)};"></div>
            </div>
          </div>
        </div>
      </div>
    `,
      )
      .join("")}
  `;
}

// ── CLEAR ─────────────────────────────────────────────────────────────────
function clearAll() {
  currentFile = null;
  document.getElementById("file-input").value = "";

  const img = document.getElementById("preview-img");
  img.style.transition = "opacity 0.3s ease";
  img.style.opacity = "0";
  setTimeout(() => {
    img.style.display = "none";
    img.style.opacity = "1";
    img.style.transition = "";
  }, 300);

  document.getElementById("upload-hint").style.display = "block";
  document.getElementById("upload-zone").classList.remove("has-image");

  lastResultImage = null;
  const dlBtn = document.getElementById("download-btn");
  dlBtn.style.display = "none";

  document.getElementById("result-zone").classList.remove("has-result");
  document.getElementById("result-zone").innerHTML = `
    <div class="result-placeholder">
      <div class="result-placeholder-icon">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18M9 21V9"/></svg>
      </div>
      Detection results will appear here after analysis.
    </div>`;

  document.getElementById("detections-list").innerHTML = "";
  document.getElementById("clahe-btn").disabled = true;
  document.getElementById("detect-btn").disabled = true;
  setStatus("Upload an X-ray image to begin.", "");
}

// ── STATUS HELPER ─────────────────────────────────────────────────────────
function setStatus(msg, type) {
  const el = document.getElementById("status-box");
  const icons = {
    success: `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" style="flex-shrink:0"><polyline points="20 6 9 17 4 12"/></svg>`,
    error: `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" style="flex-shrink:0"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>`,
    loading: "",
    "": `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" style="flex-shrink:0;opacity:0.5"><circle cx="12" cy="12" r="10"/><path d="M12 8v4M12 16h.01"/></svg>`,
  };
  el.innerHTML = (icons[type] || "") + msg;
  el.className = "status-box" + (type ? " " + type : "");
}

// ── DOWNLOAD RESULT ───────────────────────────────────────────────────────
function downloadResult() {
  if (!lastResultImage) return;
  const a = document.createElement("a");
  a.href = lastResultImage;
  a.download = "vetfractureai_result.jpg";
  a.click();
}

// ── DRAG & DROP ───────────────────────────────────────────────────────────
const zone = document.getElementById("upload-zone");

zone.addEventListener("dragover", (e) => {
  e.preventDefault();
  zone.classList.add("drag-over");
});

zone.addEventListener("dragleave", (e) => {
  if (!zone.contains(e.relatedTarget)) {
    zone.classList.remove("drag-over");
  }
});

zone.addEventListener("drop", (e) => {
  e.preventDefault();
  zone.classList.remove("drag-over");
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith("image/")) {
    const dt = new DataTransfer();
    dt.items.add(file);
    document.getElementById("file-input").files = dt.files;
    onFileSelected({ target: { files: [file] } });
  }
});

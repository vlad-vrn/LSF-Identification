/* LSF Direct — frontend
 * MediaPipe Holistic (navigateur) → extraction 318 features/frame (identique au
 * preprocessing Python) → cadence PREPARE/CAPTURE/RESULT/REST → API backend.
 */

// ── Constantes (DOIVENT correspondre au Python) ──────────────────────────────
// src/inference/mediapipe_extractor.py : FACE_LANDMARK_INDICES
const FACE_LANDMARK_INDICES = [46, 105, 334, 276, 33, 263, 1, 4, 61, 291, 0, 17, 117, 346, 152];

// src/inference/pipeline.py : durées de phase (frames ≈ 30 fps)
const DUR = { prepare: 60, capture: 64, result: 45, rest: 18 };
const CAPTURE_FRAMES = DUR.capture; // 64 = longueur d'entraînement

// ── DOM ──────────────────────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);
const videoEl = $("input_video");
const canvasEl = $("output_canvas");
const ctx = canvasEl.getContext("2d");
canvasEl.width = 640; canvasEl.height = 480;

// ── État ──────────────────────────────────────────────────────────────────────
const S = {
  running: false,
  paused: false,
  phase: "idle",
  frameInPhase: 0,
  capture: [],
  result: null,
  phrase: [],
  threshold: 0.10,
  fps: 0, _t: performance.now(), _n: 0,
};

// ── Extraction des features (318) ────────────────────────────────────────────
function extractFeatures(r) {
  const pose = new Array(132).fill(0);
  if (r.poseLandmarks) {
    r.poseLandmarks.forEach((lm, i) => {
      pose[i * 4] = lm.x; pose[i * 4 + 1] = lm.y; pose[i * 4 + 2] = lm.z;
      pose[i * 4 + 3] = lm.visibility ?? 0;
    });
  }
  const left = new Array(63).fill(0);
  if (r.leftHandLandmarks) {
    r.leftHandLandmarks.forEach((lm, i) => { left[i * 3] = lm.x; left[i * 3 + 1] = lm.y; left[i * 3 + 2] = lm.z; });
  }
  const right = new Array(63).fill(0);
  if (r.rightHandLandmarks) {
    r.rightHandLandmarks.forEach((lm, i) => { right[i * 3] = lm.x; right[i * 3 + 1] = lm.y; right[i * 3 + 2] = lm.z; });
  }
  const face = new Array(60).fill(0);
  if (r.faceLandmarks) {
    FACE_LANDMARK_INDICES.forEach((idx, k) => {
      const lm = r.faceLandmarks[idx];
      if (lm) { face[k * 4] = lm.x; face[k * 4 + 1] = lm.y; face[k * 4 + 2] = lm.z; face[k * 4 + 3] = 0.0; }
    });
  }
  return pose.concat(left, right, face); // 318
}

// ── Machine à états cadencée ─────────────────────────────────────────────────
function enterPhase(phase) {
  S.phase = phase;
  S.frameInPhase = 0;
  if (phase === "capture") S.capture = [];
  if (phase === "prepare") S.result = null;
}

function advance(vec) {
  S.frameInPhase += 1;
  if (S.phase === "capture") S.capture.push(vec);

  if (S.frameInPhase >= DUR[S.phase]) {
    if (S.phase === "capture") {
      const frames = S.capture.slice();
      enterPhase("result");          // affiche « analyse… » en attendant l'API
      runPrediction(frames);
      return;
    }
    enterPhase({ prepare: "capture", result: "rest", rest: "prepare" }[S.phase]);
  }
}

async function runPrediction(frames) {
  S.result = { pending: true };
  try {
    const res = await fetch("/api/predict", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ frames }),
    });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    const accepted = data.confidence >= S.threshold;
    S.result = { ...data, accepted, pending: false };
    if (accepted) { S.phrase.push(data.label); renderChips(); }
  } catch (e) {
    console.error("predict:", e);
    S.result = { pending: false, accepted: false, label: "erreur" };
  }
}

// ── Boucle MediaPipe ─────────────────────────────────────────────────────────
function onResults(r) {
  drawScene(r);
  // FPS
  S._n++; const now = performance.now();
  if (now - S._t > 1000) { S.fps = S._n; S._n = 0; S._t = now; $("fps").textContent = S.fps + " fps"; }

  if (S.running && !S.paused) advance(extractFeatures(r));
  renderUI();
}

function drawScene(r) {
  ctx.save();
  ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
  if (r.image) ctx.drawImage(r.image, 0, 0, canvasEl.width, canvasEl.height);
  const pc = "rgba(0,212,200,.7)", hc = "#6d8bff";
  if (window.drawConnectors) {
    if (r.poseLandmarks) drawConnectors(ctx, r.poseLandmarks, POSE_CONNECTIONS, { color: pc, lineWidth: 2 });
    if (r.leftHandLandmarks) drawConnectors(ctx, r.leftHandLandmarks, HAND_CONNECTIONS, { color: hc, lineWidth: 3 });
    if (r.rightHandLandmarks) drawConnectors(ctx, r.rightHandLandmarks, HAND_CONNECTIONS, { color: hc, lineWidth: 3 });
  }
  ctx.restore();
}

// ── Rendu UI ──────────────────────────────────────────────────────────────────
function renderUI() {
  const banner = $("phaseBanner"), cue = $("cue"), cd = $("cueCountdown"), txt = $("cueText");
  banner.className = "phase-banner " + S.phase;
  const dur = DUR[S.phase] || 1;
  $("phaseFill").style.width = Math.min(100, (S.frameInPhase / dur) * 100) + "%";

  if (S.paused) {
    banner.classList.add("paused"); banner.textContent = "EN PAUSE";
    cd.textContent = ""; txt.textContent = "Reprendre pour continuer";
    return;
  }

  if (!S.running) { banner.textContent = "EN ATTENTE"; cd.textContent = ""; txt.textContent = "Appuyez sur Démarrer"; return; }

  if (S.phase === "prepare") {
    banner.textContent = "PRÉPAREZ-VOUS";
    const remaining = DUR.prepare - S.frameInPhase;
    cd.textContent = Math.max(1, Math.ceil(remaining / (DUR.prepare / 3)));
    cd.style.color = "var(--orange)"; txt.textContent = "";
  } else if (S.phase === "capture") {
    banner.textContent = "SIGNEZ !";
    cd.textContent = ""; txt.textContent = "SIGNEZ"; txt.style.color = "var(--green)";
  } else if (S.phase === "result") {
    banner.textContent = "RÉSULTAT";
    renderResultCue(cd, txt);
  } else { // rest
    banner.textContent = "…"; cd.textContent = ""; txt.textContent = "";
  }
  renderResultBox();
}

function renderResultCue(cd, txt) {
  cd.textContent = "";
  if (!S.result || S.result.pending) { txt.textContent = "analyse…"; txt.style.color = "var(--muted)"; return; }
  if (S.result.accepted) { txt.textContent = S.result.label.toUpperCase(); txt.style.color = "var(--green)"; }
  else { txt.textContent = "NON RECONNU"; txt.style.color = "var(--red)"; }
}

function renderResultBox() {
  const lbl = $("resultLabel"), conf = $("resultConf"), alts = $("resultAlts");
  if (!S.result || S.result.pending) {
    if (S.result && S.result.pending) { lbl.textContent = "…"; lbl.className = "result-label"; conf.textContent = "analyse en cours"; alts.textContent = ""; }
    return;
  }
  if (S.result.accepted) {
    lbl.textContent = S.result.label; lbl.className = "result-label ok";
    conf.textContent = `confiance ${(S.result.confidence * 100).toFixed(0)}%`;
  } else {
    lbl.textContent = "non reconnu"; lbl.className = "result-label no";
    conf.textContent = S.result.label ? `meilleur : ${S.result.label} (${(S.result.confidence * 100).toFixed(0)}%)` : "";
  }
  alts.textContent = (S.result.top3 || []).slice(1).map(t => `${t.label} ${(t.prob * 100).toFixed(0)}%`).join("   ·   ");
}

function renderChips() {
  const box = $("chips");
  if (!S.phrase.length) { box.innerHTML = '<span class="empty">aucun signe pour l\'instant</span>'; return; }
  box.innerHTML = "";
  S.phrase.slice(-12).forEach(s => {
    const c = document.createElement("span"); c.className = "chip"; c.textContent = s.replace(/_/g, " "); box.appendChild(c);
  });
}

// ── Actions ───────────────────────────────────────────────────────────────────
async function generateSentence() {
  if (!S.phrase.length) return;
  S.paused = true;
  $("generateBtn").disabled = true; $("generateBtn").hidden = true;
  $("resumeBtn").hidden = false; $("resumeBtn").disabled = false;
  const out = $("sentence"); out.className = "sentence loading"; out.textContent = "Construction de la phrase…";
  try {
    const res = await fetch("/api/sentence", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ signs: S.phrase }),
    });
    const data = await res.json();
    out.className = "sentence"; out.textContent = data.sentence || "(vide)";
  } catch (e) {
    out.className = "sentence"; out.textContent = "Erreur de génération."; console.error(e);
  }
}

function resumeDetection() {
  S.paused = false;
  enterPhase("prepare");
  $("resumeBtn").hidden = true; $("generateBtn").hidden = false; $("generateBtn").disabled = false;
}

function clearAll() {
  S.phrase = []; renderChips();
  const out = $("sentence"); out.className = "sentence";
  out.innerHTML = '<span class="empty">— appuyez sur « Générer la phrase »</span>';
}

// ── Démarrage caméra + MediaPipe ──────────────────────────────────────────────
let holistic, camera;
async function startCamera() {
  $("startBtn").disabled = true; $("startBtn").textContent = "Initialisation…";
  holistic = new Holistic({ locateFile: (f) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${f}` });
  holistic.setOptions({
    modelComplexity: 1, smoothLandmarks: true, refineFaceLandmarks: false,
    minDetectionConfidence: 0.5, minTrackingConfidence: 0.5,
  });
  holistic.onResults(onResults);

  camera = new Camera(videoEl, {
    onFrame: async () => { await holistic.send({ image: videoEl }); },
    width: 640, height: 480,
  });
  await camera.start();

  S.running = true; enterPhase("prepare");
  $("startBtn").textContent = "● Caméra active"; $("generateBtn").disabled = false;
}

// ── Init ────────────────────────────────────────────────────────────────────
async function init() {
  $("startBtn").addEventListener("click", startCamera);
  $("generateBtn").addEventListener("click", generateSentence);
  $("resumeBtn").addEventListener("click", resumeDetection);
  $("clearBtn").addEventListener("click", clearAll);
  const thr = $("thr");
  thr.addEventListener("input", () => { S.threshold = thr.value / 100; $("thrVal").textContent = thr.value + "%"; });

  try {
    const h = await (await fetch("/api/health")).json();
    $("connStatus").className = "status-pill ok";
    $("connStatus").innerHTML = `<span class="dot"></span> Modèle prêt · ${h.n_classes} signes`;
    $("nClasses").textContent = h.n_classes;
    $("legendList").innerHTML = h.classes.map(c => `<span>${c.replace(/_/g, " ")}</span>`).join("");
  } catch (e) {
    $("connStatus").className = "status-pill err";
    $("connStatus").innerHTML = `<span class="dot"></span> Backend injoignable`;
  }
}
init();

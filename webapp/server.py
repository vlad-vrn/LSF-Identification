"""
Backend web FastAPI pour la reconnaissance LSF.

Architecture : la webcam et MediaPipe tournent dans le NAVIGATEUR (JS), qui
envoie une fenêtre de 64 frames × 318 features. Ce serveur réutilise le code
Python d'inférence (build_feature_vector + modèle 1D-CNN) et le NLP (Nous).

Endpoints :
    GET  /                → page web (static/index.html)
    GET  /api/health      → statut + liste des classes
    POST /api/predict     → {frames: [[318]×64]}  → {label, confidence, top3}
    POST /api/sentence    → {signs: [str]}        → {sentence}

Lancement local :
    uvicorn webapp.server:app --host 0.0.0.0 --port 8000
"""

import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from data.preprocessing import build_feature_vector  # noqa: E402
from model.classifier import make_classifier  # noqa: E402
from nlp.sentence_builder import build_sentence  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("lsf.server")

SEQUENCE_LENGTH = 64
RAW_FEATURE_DIM = 318
MODEL_PATH = os.environ.get("LSF_MODEL_PATH", str(_ROOT / "models" / "lsf_v1.pt"))


def _load_dotenv(path: Path) -> None:
    """Charge un .env local (KEY=VALUE) sans écraser l'environnement existant."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv(_ROOT / ".env")

# --- Chargement du modèle au démarrage ---
_device = torch.device("cpu")
logger.info("Chargement du modèle : %s", MODEL_PATH)
_ckpt = torch.load(MODEL_PATH, map_location=_device, weights_only=False)
_model = make_classifier(n_classes=_ckpt["n_classes"])
_model.load_state_dict(_ckpt["model_state_dict"])
_model.to(_device).eval()
_label_map: dict[str, int] = _ckpt["label_map"]
_idx_to_label = {v: k for k, v in _label_map.items()}
logger.info("Modèle chargé — %d classes : %s", len(_label_map), list(_label_map.keys()))

app = FastAPI(title="LSF Recognition", version="1.0")


# ── Schémas ────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    frames: list[list[float]]  # 64 × 318


class SentenceRequest(BaseModel):
    signs: list[str]


# ── Endpoints API ───────────────────────────────────────────────────────────

@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "classes": list(_label_map.keys()), "n_classes": len(_label_map)}


@app.post("/api/predict")
def predict(req: PredictRequest) -> dict:
    """Reçoit une fenêtre de keypoints bruts, retourne la prédiction."""
    arr = np.asarray(req.frames, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != RAW_FEATURE_DIM:
        raise HTTPException(400, f"Attendu (N, {RAW_FEATURE_DIM}), reçu {arr.shape}")
    if arr.shape[0] < 8:
        raise HTTPException(400, "Trop peu de frames (minimum 8).")

    try:
        feat = build_feature_vector(arr)                      # (64, 320)
        x = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(_model(x), dim=1)[0]
    except Exception as exc:  # noqa: BLE001
        logger.exception("Erreur d'inférence")
        raise HTTPException(500, f"Erreur d'inférence : {exc}") from None

    sorted_probs, sorted_idx = probs.sort(descending=True)
    top3 = [
        {"label": _idx_to_label[int(sorted_idx[i])], "prob": float(sorted_probs[i])}
        for i in range(min(3, len(sorted_idx)))
    ]
    return {"label": top3[0]["label"], "confidence": top3[0]["prob"], "top3": top3}


@app.post("/api/sentence")
def sentence(req: SentenceRequest) -> dict:
    """Reconstruit une phrase française à partir des signes (NLP Nous/Anthropic)."""
    if not req.signs:
        return {"sentence": ""}
    text = build_sentence(req.signs)   # provider/clé lus depuis l'environnement
    return {"sentence": text}


# ── Fichiers statiques (frontend) — monté en dernier pour ne pas masquer /api ─
app.mount("/", StaticFiles(directory=str(Path(__file__).parent / "static"), html=True), name="static")

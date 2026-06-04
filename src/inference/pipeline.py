"""
Pipeline d'inférence temps réel CADENCÉE pour la reconnaissance LSF.

Remplace l'ancienne détection continue « signe / pas de signe » (gate de
mouvement + déduplication + cooldown) par une capture rythmée type métronome,
qui synchronise l'utilisateur sur la fenêtre exacte attendue par le modèle
(SEQUENCE_LENGTH frames, comme à l'entraînement).

Boucle à 4 phases, répétée indéfiniment :

    PREPARE  → décompte 3-2-1, l'utilisateur se met en position
    CAPTURE  → on enregistre exactement CAPTURE_FRAMES frames pendant que
               l'utilisateur réalise UN signe
    RESULT   → une inférence est lancée, le résultat est affiché
    REST     → courte pause avant le battement suivant

Une seule prédiction par battement. Si la confiance est sous le seuil, le
résultat est « non reconnu » et rien n'est ajouté à la phrase.

Toutes les durées sont des constantes nommées, surchargeables à la création de
l'état (make_cadence_state) — elles-mêmes exposées en arguments CLI par run.py.
"""

import logging
import math
from collections import deque

import numpy as np
import torch
import torch.nn as nn

from data.preprocessing import build_feature_vector

logger = logging.getLogger(__name__)

SEQUENCE_LENGTH: int = 64

# --- Durées des phases, en frames (≈ valeurs à 30 fps) ---
PREPARE_FRAMES: int = 60          # ~2.0 s — décompte 3-2-1
CAPTURE_FRAMES: int = SEQUENCE_LENGTH  # ~2.1 s — fenêtre du signe (= entraînement)
RESULT_FRAMES: int = 45           # ~1.5 s — affichage du résultat
REST_FRAMES: int = 18             # ~0.6 s — pause entre deux battements

# Seuil de confiance softmax sous lequel le signe est déclaré « non reconnu ».
# Volontairement bas : on accepte le meilleur label dès qu'il dépasse 10 %,
# pour afficher quasi systématiquement une prédiction plutôt que « non reconnu ».
CONFIDENCE_THRESHOLD: float = 0.10

# --- Phases ---
PHASE_PREPARE: str = "prepare"
PHASE_CAPTURE: str = "capture"
PHASE_RESULT: str = "result"
PHASE_REST: str = "rest"

_NEXT_PHASE: dict[str, str] = {
    PHASE_PREPARE: PHASE_CAPTURE,
    PHASE_CAPTURE: PHASE_RESULT,
    PHASE_RESULT: PHASE_REST,
    PHASE_REST: PHASE_PREPARE,
}


def make_cadence_state(
    prepare_frames: int = PREPARE_FRAMES,
    capture_frames: int = CAPTURE_FRAMES,
    result_frames: int = RESULT_FRAMES,
    rest_frames: int = REST_FRAMES,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> dict:
    """
    Crée un état de cadence vierge (démarre en phase PREPARE).

    Champs :
        phase            : phase courante (PHASE_*)
        frame_in_phase   : nombre de frames écoulées dans la phase courante
        capture          : liste des frames brutes (318,) enregistrées en CAPTURE
        result           : dernier résultat d'inférence (dict) ou None
        durations        : {phase: durée en frames}
        confidence_threshold : seuil d'acceptation
    """
    return {
        "phase": PHASE_PREPARE,
        "frame_in_phase": 0,
        "capture": [],
        "result": None,
        "durations": {
            PHASE_PREPARE: prepare_frames,
            PHASE_CAPTURE: capture_frames,
            PHASE_RESULT: result_frames,
            PHASE_REST: rest_frames,
        },
        "confidence_threshold": confidence_threshold,
    }


def _enter_phase(state: dict, phase: str) -> None:
    """Bascule l'état vers une nouvelle phase et réinitialise les compteurs utiles."""
    state["phase"] = phase
    state["frame_in_phase"] = 0
    if phase == PHASE_CAPTURE:
        state["capture"] = []
    if phase == PHASE_PREPARE:
        state["result"] = None   # on efface le résultat précédent avant le prochain signe


def _infer_capture(
    state: dict,
    model: nn.Module,
    label_map: dict[str, int],
    device: torch.device,
) -> dict | None:
    """
    Lance une inférence sur les frames capturées et retourne un dict résultat.

    Returns:
        {
            "label":      str | None,   # None si rejeté (sous le seuil)
            "best_label": str,          # meilleur label quoi qu'il arrive (affichage)
            "confidence": float,        # softmax du meilleur label
            "accepted":   bool,         # confidence >= seuil
            "top3":       [(label, prob), ...],
        }
        ou None si la capture est trop courte.
    """
    if len(state["capture"]) < 2:
        return None

    raw = np.stack(state["capture"])        # (N, 318)
    frames = build_feature_vector(raw)       # (64, 320) — même preprocessing qu'à l'entraînement
    x = torch.tensor(frames, dtype=torch.float32).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]

    sorted_probs, sorted_idx = probs.sort(descending=True)
    idx_to_label = {v: k for k, v in label_map.items()}
    best_label = idx_to_label.get(int(sorted_idx[0]), f"classe_{int(sorted_idx[0])}")
    confidence = float(sorted_probs[0])
    accepted = confidence >= state["confidence_threshold"]

    top3 = [
        (idx_to_label.get(int(sorted_idx[i]), f"cls_{i}"), float(sorted_probs[i]))
        for i in range(min(3, len(sorted_idx)))
    ]

    return {
        "label": best_label if accepted else None,
        "best_label": best_label,
        "confidence": confidence,
        "accepted": accepted,
        "top3": top3,
    }


def step_cadence(
    state: dict,
    frame_features: np.ndarray,
    model: nn.Module,
    label_map: dict[str, int],
    device: torch.device,
) -> tuple[dict, str | None]:
    """
    Fait avancer la machine à états d'une frame.

    À appeler une fois par frame webcam. Gère l'enregistrement pendant CAPTURE,
    déclenche l'inférence à la fin de CAPTURE, et enchaîne les phases.

    Returns:
        (status, just_confirmed) :
          status         : dict décrivant l'état courant pour l'affichage
                           (voir _build_status).
          just_confirmed : slug du signe si un signe vient d'être CONFIRMÉ à
                           cette frame (à ajouter à la phrase), sinon None.
    """
    state["frame_in_phase"] += 1
    phase = state["phase"]
    just_confirmed: str | None = None

    # Enregistrement des frames pendant la capture
    if phase == PHASE_CAPTURE:
        state["capture"].append(frame_features.astype(np.float32))

    # Fin de phase atteinte ?
    if state["frame_in_phase"] >= state["durations"][phase]:
        if phase == PHASE_CAPTURE:
            result = _infer_capture(state, model, label_map, device)
            state["result"] = result
            if result is not None and result["accepted"]:
                just_confirmed = result["label"]
                logger.info("Signe confirmé : %s (conf=%.2f)", result["label"], result["confidence"])
            elif result is not None:
                logger.info("Non reconnu (best=%s, conf=%.2f)", result["best_label"], result["confidence"])
        _enter_phase(state, _NEXT_PHASE[phase])

    return _build_status(state), just_confirmed


def current_status(state: dict) -> dict:
    """Retourne le status d'affichage courant SANS faire avancer la machine.

    Utile quand la détection est en pause (génération de phrase) : on continue
    d'afficher le dernier état sans capturer ni inférer.
    """
    return _build_status(state)


def _build_status(state: dict) -> dict:
    """Construit le dict d'état destiné à l'affichage (overlay)."""
    phase = state["phase"]
    dur = state["durations"][phase]
    elapsed = state["frame_in_phase"]
    progress = min(1.0, elapsed / dur) if dur > 0 else 1.0

    # Décompte 3-2-1 pendant PREPARE
    countdown: int | None = None
    if phase == PHASE_PREPARE:
        remaining = dur - elapsed
        countdown = max(1, math.ceil(remaining / (dur / 3.0))) if dur > 0 else 1

    return {
        "phase": phase,
        "progress": progress,
        "countdown": countdown,
        "captured": len(state["capture"]),
        "capture_target": state["durations"][PHASE_CAPTURE],
        "result": state["result"],
    }

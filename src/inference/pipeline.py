"""
Pipeline d'inférence temps réel pour la reconnaissance LSF.

Le flux de frames MediaPipe est accumulé dans un buffer circulaire.
Toutes les INFERENCE_STRIDE frames, si buffer plein + mouvement suffisant
+ pas de cooldown actif, on lance une inférence CNN.

Filtres anti-faux-positifs (en couches successives) :
  1. Gate de mouvement     : MOVEMENT_THRESHOLD — pas d'inférence au repos
  2. Seuil confiance       : CONFIDENCE_THRESHOLD — softmax dominant suffisant
  3. Seuil entropie        : UNCERTAINTY_GATE — bloque si modèle trop incertain
  4. Marge top1-top2       : CONFIDENCE_MARGIN — bloque si deux classes proches
  5. Dédup N consécutives  : DEDUP_COUNT — confirmation temporelle
  6. Reset dédup au repos  : si mouvement < seuil entre deux strides → reset
  7. Cooldown post-confirm  : buffer vidé + inférence suspendue COOLDOWN_FRAMES frames
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

# À 30fps, stride=15 → inférence toutes les ~0.5s.
INFERENCE_STRIDE: int = 15

# 2 prédictions consécutives cohérentes → confirmation (~1s de cohérence).
DEDUP_COUNT: int = 2

# Seuil softmax minimum pour qu'une prédiction entre dans la dédup.
CONFIDENCE_THRESHOLD: float = 0.6

# Mouvement minimum des mains (somme normes L2 frame-à-frame) pour déclencher
# l'inférence. Évite les prédictions en continu quand les mains sont au repos.
MOVEMENT_THRESHOLD: float = 1.5

# Marge minimum entre top1 et top2 softmax.
# Évite les confirmations ambiguës (modèle hésitant entre deux signes proches).
# Ex. top1=0.45, top2=0.42 → marge=0.03 < 0.15 → refusé.
CONFIDENCE_MARGIN: float = 0.15

# Entropie normalisée maximum pour compter dans la dédup.
# 0.75 ≈ distribution très étalée sur les classes → modèle incertain.
# Un signe non reconnu produit une entropie élevée → bloqué ici.
UNCERTAINTY_GATE: float = 0.75

# Frames de cooldown après confirmation : le buffer est vidé et l'inférence
# suspendue COOLDOWN_FRAMES frames (~0.7s à 30fps). En pratique, le buffer
# a aussi besoin de ~2s pour se remplir, ce qui donne ~2.7s minimum entre signes.
COOLDOWN_FRAMES: int = 20


def make_inference_state() -> dict:
    """
    Crée un état d'inférence vierge.

    Champs :
        buffer           : deque(maxlen=64) — frames brutes (318,)
        stride_counter   : int — frames depuis la dernière tentative d'inférence
        last_predictions : deque(maxlen=DEDUP_COUNT) — (label, confidence)
        cooldown_counter : int — frames restantes avant de reprendre l'inférence
    """
    return {
        "buffer": deque(maxlen=SEQUENCE_LENGTH),
        "stride_counter": 0,
        "last_predictions": deque(maxlen=DEDUP_COUNT),
        "cooldown_counter": 0,
    }


def push_frame(state: dict, frame_features: np.ndarray) -> dict:
    """Ajoute une frame brute (318,) au buffer et décrémente le cooldown."""
    state["buffer"].append(frame_features.astype(np.float32))
    state["stride_counter"] += 1
    if state["cooldown_counter"] > 0:
        state["cooldown_counter"] -= 1
    return state


def _window_movement(state: dict) -> float:
    """Mouvement total des mains (gauche+droite) sur la fenêtre courante."""
    if len(state["buffer"]) < 2:
        return 0.0
    frames = np.stack(list(state["buffer"]))   # (N, 318)
    hands = frames[:, 132:258]                  # (N, 126) — left+right uniquement
    diffs = np.diff(hands, axis=0)
    return float(np.linalg.norm(diffs, axis=1).sum())


def should_run_inference(state: dict) -> bool:
    """
    True si buffer plein, stride atteint, cooldown terminé ET mouvement suffisant.

    Si le mouvement est insuffisant, réinitialise la dédup : cela évite les
    confirmations différées où une prédiction ancienne se combine avec une
    nouvelle pour former une fausse concordance après une pause.
    """
    if len(state["buffer"]) < SEQUENCE_LENGTH:
        return False
    if state["stride_counter"] < INFERENCE_STRIDE:
        return False

    state["stride_counter"] = 0

    if state["cooldown_counter"] > 0:
        return False

    if _window_movement(state) < MOVEMENT_THRESHOLD:
        state["last_predictions"].clear()
        logger.debug("Mouvement insuffisant — dédup réinitialisée")
        return False

    return True


def run_inference(
    model: nn.Module,
    state: dict,
    label_map: dict[str, int],
    device: torch.device,
) -> tuple[str, float, float, float, list[tuple[str, float]]] | None:
    """
    Lance l'inférence CNN sur les 64 frames du buffer.

    Returns:
        (label, confidence, uncertainty, margin, top3) ou None si buffer incomplet.
        - confidence  : probabilité softmax du meilleur label (0–1)
        - uncertainty : entropie normalisée (0=certain, 1=aléatoire total)
        - margin      : écart top1 - top2 (0–1) — proche de 0 = ambigu
        - top3        : liste des 3 meilleures prédictions [(label, prob), ...]
    """
    if len(state["buffer"]) < SEQUENCE_LENGTH:
        return None

    raw = np.stack(list(state["buffer"]))    # (64, 318)
    frames = build_feature_vector(raw)        # (64, 320)

    x = torch.tensor(frames, dtype=torch.float32).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]

    sorted_probs, sorted_indices = probs.sort(descending=True)
    best_idx = int(sorted_indices[0])
    confidence = float(sorted_probs[0])
    margin = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else 1.0

    p = probs.cpu().numpy().astype(np.float64)
    entropy = -float(np.sum(p * np.log(p + 1e-9)))
    uncertainty = entropy / math.log(len(p))

    idx_to_label = {v: k for k, v in label_map.items()}
    label = idx_to_label.get(best_idx, f"classe_{best_idx}")

    top3 = [
        (idx_to_label.get(int(sorted_indices[i]), f"cls_{i}"), float(sorted_probs[i]))
        for i in range(min(3, len(sorted_indices)))
    ]

    return label, confidence, uncertainty, margin, top3


def deduplicate(
    state: dict,
    label: str,
    confidence: float,
    uncertainty: float,
    margin: float,
    threshold: float = CONFIDENCE_THRESHOLD,
) -> str | None:
    """
    Confirme un signe après DEDUP_COUNT prédictions consécutives de qualité.

    Une prédiction est acceptée dans la dédup si toutes ces conditions sont vraies :
    - confidence > threshold     (softmax dominant suffisant)
    - uncertainty < UNCERTAINTY_GATE (entropie basse — modèle sûr du label)
    - margin > CONFIDENCE_MARGIN (top1 nettement au-dessus de top2)

    Si une prédiction échoue, le compteur est réinitialisé : mieux vaut exiger
    N prédictions consécutives propres que tolérer du bruit entre elles.

    Après confirmation :
    - buffer vidé (les frames du signe ne sont plus réutilisées)
    - cooldown activé (inférence suspendue COOLDOWN_FRAMES frames)
    """
    quality_ok = (
        confidence > threshold
        and uncertainty < UNCERTAINTY_GATE
        and margin > CONFIDENCE_MARGIN
    )

    if not quality_ok:
        state["last_predictions"].clear()
        return None

    state["last_predictions"].append((label, confidence))

    if len(state["last_predictions"]) < DEDUP_COUNT:
        return None

    labels = [p[0] for p in state["last_predictions"]]
    if len(set(labels)) != 1:
        state["last_predictions"].clear()
        return None

    # Confirmation — réinitialise tout pour le prochain signe
    state["last_predictions"].clear()
    state["buffer"].clear()
    state["cooldown_counter"] = COOLDOWN_FRAMES
    return label


def get_dedup_progress(state: dict) -> tuple[int, str | None, int]:
    """
    Retourne (n_consécutifs, dernier_label, cooldown_restant).

    n_consécutifs : nombre de prédictions identiques depuis la plus récente.
    cooldown_restant : frames avant la prochaine inférence (0 = prêt).
    Utilisé pour l'affichage des points de progression et du flash de confirmation.
    """
    preds = list(state["last_predictions"])
    cooldown = state["cooldown_counter"]

    if not preds:
        return 0, None, cooldown

    last_label = preds[-1][0]
    count = 0
    for lbl, _ in reversed(preds):
        if lbl == last_label:
            count += 1
        else:
            break
    return count, last_label, cooldown

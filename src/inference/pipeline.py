"""
Pipeline d'inférence temps réel pour la reconnaissance LSF.

Le flux de frames MediaPipe est accumulé dans un buffer circulaire.
Toutes les INFERENCE_STRIDE frames, si le buffer est plein, on lance
une inférence et on tente de confirmer le signe par déduplication.

Design : état représenté par un dict simple (pas de classe) pour rester
cohérent avec la convention "fonctions simples" du projet V1.
"""

import logging
import math
from collections import deque

import numpy as np
import torch
import torch.nn as nn

from data.preprocessing import build_feature_vector

logger = logging.getLogger(__name__)

# Nombre de frames dans la fenêtre d'inférence (cohérent avec le dataset)
SEQUENCE_LENGTH: int = 64

# À 30fps, stride=15 = nouvelle inférence toutes les 0.5s.
# Compromis entre réactivité (stride bas) et charge CPU (stride haut).
INFERENCE_STRIDE: int = 15

# Seuil de mouvement minimum pour lancer l'inférence.
# Calculé comme la somme des normes L2 des déplacements frame-à-frame
# sur les colonnes main gauche + main droite (indices 132:258 du vecteur brut).
# Si personne ne bouge, ce score est proche de 0 (jitter MediaPipe seulement).
# Valeur empirique : à ajuster selon la sensibilité souhaitée.
MOVEMENT_THRESHOLD: float = 1.5

# Nombre de prédictions consécutives identiques avant de confirmer un signe.
# 2 × 0.5s = 1s — compromis entre filtrage des faux positifs et latence.
# Avec 3 : trop long à tenir (1.5s), peu naturel pour des signes dynamiques.
# Repasser à 3 si trop de faux positifs en pratique.
DEDUP_COUNT: int = 2

# Seuil de confiance softmax pour valider une prédiction.
# 0.6 = valeur de départ raisonnable sur 12 classes (chance = 0.083).
# À ajuster après avoir mesuré les distributions de confiance en pratique.
CONFIDENCE_THRESHOLD: float = 0.6


# ---------------------------------------------------------------------------
# Création de l'état (InferenceBuffer)
# ---------------------------------------------------------------------------

def make_inference_state() -> dict:
    """
    Crée un état d'inférence vierge.

    Structure :
        buffer           : deque(maxlen=64) — frames brutes (318 floats chacune)
        stride_counter   : int — compte les frames depuis la dernière inférence
        last_predictions : deque(maxlen=3) — tuples (label, confidence)

    Returns:
        Dict d'état prêt à être passé à push_frame.
    """
    return {
        "buffer": deque(maxlen=SEQUENCE_LENGTH),
        "stride_counter": 0,
        "last_predictions": deque(maxlen=DEDUP_COUNT),
    }


# ---------------------------------------------------------------------------
# push_frame
# ---------------------------------------------------------------------------

def push_frame(state: dict, frame_features: np.ndarray) -> dict:
    """
    Ajoute une frame préprocessée au buffer et incrémente le stride_counter.

    Args:
        state:          Dict d'état issu de make_inference_state().
        frame_features: Vecteur de 318 floats bruts pour cette frame
                        (pose + left_hand + right_hand + face — sans flags).
                        La détection dominant/passif et le z-score sont appliqués
                        sur la fenêtre complète au moment de l'inférence.

    Returns:
        Le même dict d'état mis à jour (modification en place + retour pour
        permettre un usage fonctionnel si souhaité).
    """
    state["buffer"].append(frame_features.astype(np.float32))
    state["stride_counter"] += 1
    return state


# ---------------------------------------------------------------------------
# should_run_inference
# ---------------------------------------------------------------------------

def _window_movement(state: dict) -> float:
    """
    Calcule le mouvement total des mains sur la fenêtre courante.

    On mesure la somme des normes L2 des déplacements frame-à-frame,
    uniquement sur les colonnes main gauche + droite (indices 132:258).
    La pose et le visage sont exclus : ils bougent même quand on est immobile
    (respiration, micro-mouvements de la tête).

    Args:
        state: Dict d'état avec buffer non vide.

    Returns:
        Score de mouvement (float ≥ 0). Proche de 0 si immobile.
    """
    frames = np.stack(list(state["buffer"]))   # (N, 318)
    hands = frames[:, 132:258]                  # (N, 126) — gauche + droite
    diffs = np.diff(hands, axis=0)              # (N-1, 126)
    return float(np.linalg.norm(diffs, axis=1).sum())


def should_run_inference(state: dict) -> bool:
    """
    Détermine si une inférence doit être lancée sur la frame courante.

    Conditions :
      1. Buffer plein (64 frames)
      2. stride_counter >= INFERENCE_STRIDE
      3. Mouvement des mains dans la fenêtre > MOVEMENT_THRESHOLD

    La condition 3 empêche le modèle de tourner en continu quand la personne
    est immobile — sans elle, le modèle prédit toujours la classe majoritaire
    même sans signe.

    Si les conditions 1+2 sont réunies, stride_counter est remis à 0
    (que le mouvement soit suffisant ou non, pour éviter l'accumulation).

    Args:
        state: Dict d'état courant.

    Returns:
        True si une inférence doit être lancée, False sinon.
    """
    buffer_full = len(state["buffer"]) == SEQUENCE_LENGTH
    stride_reached = state["stride_counter"] >= INFERENCE_STRIDE

    if not (buffer_full and stride_reached):
        return False

    state["stride_counter"] = 0

    movement = _window_movement(state)
    if movement < MOVEMENT_THRESHOLD:
        logger.debug("Mouvement insuffisant (%.2f < %.2f) — inférence ignorée", movement, MOVEMENT_THRESHOLD)
        return False

    return True


# ---------------------------------------------------------------------------
# run_inference
# ---------------------------------------------------------------------------

def run_inference(
    model: nn.Module,
    state: dict,
    label_map: dict[str, int],
    device: torch.device,
) -> tuple[str, float, float] | None:
    """
    Lance l'inférence sur les 64 frames du buffer.

    Applique le z-score par feature sur la fenêtre courante (même normalisation
    que build_feature_vector au training) avant de passer au modèle.

    Args:
        model:     LSFClassifier chargé en mode eval.
        state:     Dict d'état avec buffer plein.
        label_map: Dict {slug: idx} pour convertir l'indice en label.
        device:    Dispositif de calcul (cpu).

    Returns:
        (label, confidence, uncertainty) si le buffer est plein, None sinon.
        - confidence  : probabilité softmax du meilleur label (0–1)
        - uncertainty : entropie normalisée (0 = très sûr, 1 = aléatoire total)
          Un signe inconnu produit une distribution plate → uncertainty proche de 1.
    """
    if len(state["buffer"]) < SEQUENCE_LENGTH:
        return None

    # Empile les 64 frames brutes en (64, 318)
    raw = np.stack(list(state["buffer"]))   # (64, 318)

    # Applique le même preprocessing que l'entraînement :
    # trim, dominant/passif, flags de présence, z-score → (64, 320)
    frames = build_feature_vector(raw)

    x = torch.tensor(frames, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 64, 320)

    model.eval()
    with torch.no_grad():
        logits = model(x)                                   # (1, n_classes)
        probs = torch.softmax(logits, dim=1)[0]             # (n_classes,)

    best_idx = int(probs.argmax())
    confidence = float(probs[best_idx])

    # Entropie normalisée : mesure à quel point la distribution est plate.
    # entropy = -Σ p·log(p), normalisée par log(n_classes) → [0, 1]
    # Vaut 0 quand le modèle est certain, 1 quand toutes les classes sont équiprobables.
    p = probs.cpu().numpy().astype(np.float64)
    n_classes = len(p)
    entropy = -float(np.sum(p * np.log(p + 1e-9)))
    uncertainty = entropy / math.log(n_classes)

    # Mapping inverse idx → slug
    idx_to_label = {v: k for k, v in label_map.items()}
    label = idx_to_label.get(best_idx, f"classe_{best_idx}")

    return label, confidence, uncertainty


# ---------------------------------------------------------------------------
# deduplicate
# ---------------------------------------------------------------------------

def deduplicate(
    state: dict,
    label: str,
    confidence: float,
    threshold: float = CONFIDENCE_THRESHOLD,
) -> str | None:
    """
    Confirme un signe si les DEDUP_COUNT dernières prédictions sont cohérentes.

    Un signe est confirmé quand :
    - Les DEDUP_COUNT dernières prédictions (incluant la courante) sont le même label
    - Toutes ont une confiance supérieure au seuil

    Cette double condition (temporelle + confiance) réduit fortement les faux
    positifs sans introduire de délai perceptible (DEDUP_COUNT × 0.5s = 1.5s max).

    Args:
        state:      Dict d'état contenant last_predictions.
        label:      Label de la prédiction courante.
        confidence: Confiance softmax de la prédiction courante.
        threshold:  Seuil de confiance minimum (défaut CONFIDENCE_THRESHOLD).

    Returns:
        Le label si confirmé, None sinon.
    """
    state["last_predictions"].append((label, confidence))

    if len(state["last_predictions"]) < DEDUP_COUNT:
        return None

    labels = [p[0] for p in state["last_predictions"]]
    confidences = [p[1] for p in state["last_predictions"]]

    all_same_label = len(set(labels)) == 1
    all_above_threshold = all(c > threshold for c in confidences)

    if all_same_label and all_above_threshold:
        # Vide le buffer après confirmation pour éviter de confirmer
        # le même signe en boucle tant que la pose est maintenue.
        state["last_predictions"].clear()
        return label

    return None


# ---------------------------------------------------------------------------
# get_dedup_progress
# ---------------------------------------------------------------------------

def get_dedup_progress(state: dict) -> tuple[int, str | None]:
    """
    Retourne l'avancement de la déduplication pour l'affichage temps réel.

    Compte les prédictions consécutives identiques depuis la plus récente,
    ce qui permet d'afficher une progression type "●●○" vers la confirmation.

    Args:
        state: Dict d'état courant.

    Returns:
        (count, label) — nombre de prédictions consécutives du même label
        depuis la fin, et ce label. (0, None) si le buffer est vide.
    """
    preds = list(state["last_predictions"])
    if not preds:
        return 0, None

    last_label = preds[-1][0]
    count = 0
    for label, _ in reversed(preds):
        if label == last_label:
            count += 1
        else:
            break
    return count, last_label

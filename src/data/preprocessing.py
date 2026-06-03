"""
Preprocessing des samples LSF bruts vers des vecteurs features normalisés.

Pipeline par sample :
  raw (64, 318)
    → trim_rest_frames      — supprime les frames inactives début/fin
    → detect_dominant_hand  — renomme gauche/droite en dominant/passif
    → add_presence_flags    — ajoute 2 flags binaires de présence
    → z-score par feature   — normalise chaque dimension sur la séquence
  → (64, 320)
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Seuil de mouvement (norme L2 du déplacement frame-à-frame) sous lequel
# une frame est considérée "au repos". Exposé comme constante pour tuning facile.
TRIM_THRESHOLD: float = 0.01

# Nombre maximum de frames à couper en début de séquence pour l'augmentation
MAX_TRIM_START: int = 10

# Nombre minimal de frames conservées après trim avant de renoncer
MIN_FRAMES_AFTER_TRIM: int = 16

# Offsets des colonnes dans le vecteur brut (64, 318)
_POSE_START = 0
_POSE_END = 132
_LEFT_START = 132
_LEFT_END = 195
_RIGHT_START = 195
_RIGHT_END = 258
_FACE_START = 258
_FACE_END = 318

SEQUENCE_LENGTH = 64


# ---------------------------------------------------------------------------
# Fonctions utilitaires internes
# ---------------------------------------------------------------------------

def _hand_total_movement(hand_frames: np.ndarray) -> float:
    """
    Calcule le mouvement total d'une main sur toute la séquence.

    Args:
        hand_frames: shape (64, 63).

    Returns:
        Norme L2 de tous les déplacements frame-à-frame concaténés.
    """
    diffs = np.diff(hand_frames, axis=0)   # (63, 63) — déplacements frame-à-frame
    return float(np.linalg.norm(diffs))


def _resample(frames: np.ndarray, target_length: int) -> np.ndarray:
    """
    Rééchantillonne une séquence vers target_length frames par interpolation linéaire.

    Args:
        frames: shape (N, D).
        target_length: nombre de frames cible.

    Returns:
        shape (target_length, D), même dtype que l'entrée.
    """
    current_length = len(frames)
    if current_length == target_length:
        return frames

    old_idx = np.arange(current_length, dtype=np.float32)
    new_idx = np.linspace(0, current_length - 1, target_length, dtype=np.float32)

    resampled = np.empty((target_length, frames.shape[1]), dtype=frames.dtype)
    for dim in range(frames.shape[1]):
        resampled[:, dim] = np.interp(new_idx, old_idx, frames[:, dim])
    return resampled


# ---------------------------------------------------------------------------
# Fonctions publiques
# ---------------------------------------------------------------------------

def detect_dominant_hand(
    frames: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Identifie la main dominante par son mouvement total, indépendamment du côté.

    On utilise le mouvement plutôt que le côté car la webcam en mode miroir
    (comportement navigateur par défaut) inverse gauche/droite : un droitier
    signe de la main droite qui apparaît à gauche dans le flux, enregistrée
    comme left_hand. Pour certains signes, 80-100 % des samples ont la "main
    gauche" active — ce qui est incohérent biologiquement et s'explique par
    ce miroir. Le mouvement est invariant à ce retournement.

    Args:
        frames: shape (64, 318) — pose(132) + left(63) + right(63) + face(60).

    Returns:
        (dominant_hand, passive_hand), chacun shape (64, 63).
    """
    left_hand = frames[:, _LEFT_START:_LEFT_END]    # (64, 63)
    right_hand = frames[:, _RIGHT_START:_RIGHT_END]  # (64, 63)

    left_movement = _hand_total_movement(left_hand)
    right_movement = _hand_total_movement(right_hand)

    if right_movement >= left_movement:
        return right_hand.copy(), left_hand.copy()
    else:
        return left_hand.copy(), right_hand.copy()


def add_presence_flags(
    dominant: np.ndarray,
    passive: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcule les flags de présence pour la main dominante et passive.

    Une main est considérée absente si tous ses 63 floats sont exactement 0
    pour une frame donnée (convention du dataset : mains manquantes = zéros).

    Args:
        dominant: shape (64, 63).
        passive:  shape (64, 63).

    Returns:
        (dominant_flags, passive_flags), chacun shape (64,), valeurs 0.0 ou 1.0.
    """
    # np.any sur l'axe 1 : True si au moins un float non nul dans la frame
    dominant_flags = np.where(np.any(dominant != 0, axis=1), 1.0, 0.0).astype(np.float32)
    passive_flags = np.where(np.any(passive != 0, axis=1), 1.0, 0.0).astype(np.float32)
    return dominant_flags, passive_flags


def trim_rest_frames(
    frames: np.ndarray,
    threshold: float = TRIM_THRESHOLD,
) -> np.ndarray:
    """
    Supprime les frames de repos en début et fin de séquence.

    Le mouvement d'une transition i→i+1 est calculé comme la norme L2 du
    déplacement des keypoints pose+mains (258 valeurs) entre frames consécutives.
    Les frames sont rognées depuis le début et la fin jusqu'à ce que le
    mouvement dépasse le seuil.

    Si moins de MIN_FRAMES_AFTER_TRIM frames restent après le trim, la séquence
    originale est retournée sans modification (cas dégénéré ou seuil trop élevé).

    La séquence rognée est rééchantillonnée vers SEQUENCE_LENGTH frames.

    Args:
        frames:    shape (64, 318) — séquence brute.
        threshold: seuil de mouvement (norme L2 frame-à-frame).

    Returns:
        shape (64, 318), séquence rognée et rééchantillonnée.
    """
    # On utilise pose + mains uniquement pour détecter l'activité (pas la face)
    kp = frames[:, _POSE_START:_RIGHT_END]  # (64, 258)

    diffs = np.diff(kp, axis=0)                     # (63, 258)
    movement = np.linalg.norm(diffs, axis=1)         # (63,) — mouvement par transition

    # Indices des transitions actives
    active = np.where(movement > threshold)[0]

    if len(active) == 0:
        # Aucun mouvement détecté : séquence déjà plate, on retourne telle quelle
        logger.warning("trim_rest_frames : aucun mouvement > %.4f détecté, trim ignoré", threshold)
        return frames

    # La transition active[0] est entre frame active[0] et active[0]+1 ;
    # on garde à partir de la frame active[0].
    start_frame = int(active[0])
    end_frame = int(active[-1]) + 2  # +2 : on inclut la frame après la dernière transition

    end_frame = min(end_frame, SEQUENCE_LENGTH)
    n_frames = end_frame - start_frame

    if n_frames < MIN_FRAMES_AFTER_TRIM:
        logger.warning(
            "trim_rest_frames : seulement %d frames après trim (min=%d), trim ignoré",
            n_frames, MIN_FRAMES_AFTER_TRIM,
        )
        return frames

    trimmed = frames[start_frame:end_frame]  # (n_frames, 318)
    return _resample(trimmed, SEQUENCE_LENGTH)


def build_feature_vector(frames: np.ndarray) -> np.ndarray:
    """
    Construit le vecteur feature normalisé à partir d'un sample brut.

    Pipeline :
      1. Trim des frames de repos (début/fin)
      2. Détection de la main dominante par mouvement
      3. Calcul des flags de présence
      4. Reconstruction : pose(132) + dominant(63) + passive(63) + face(60) + flags(2)
      5. Z-score par feature sur les 64 frames (instance normalization)

    Args:
        frames: shape (64, 318) — sample brut issu du loader.

    Returns:
        shape (64, 320), float32, normalisé.
    """
    frames = trim_rest_frames(frames)

    pose = frames[:, _POSE_START:_POSE_END]   # (64, 132)
    face = frames[:, _FACE_START:_FACE_END]   # (64, 60)

    dominant, passive = detect_dominant_hand(frames)  # (64, 63) chacun
    dom_flags, pas_flags = add_presence_flags(dominant, passive)  # (64,) chacun

    # Concatène dans l'ordre final : pose + dominant + passive + face + flags
    feature = np.concatenate([
        pose,
        dominant,
        passive,
        face,
        dom_flags[:, np.newaxis],   # (64, 1)
        pas_flags[:, np.newaxis],   # (64, 1)
    ], axis=1)  # (64, 320)

    # Z-score par feature sur la séquence (instance normalization)
    # std=1 si std≈0 pour éviter la division par zéro sur les features constantes
    mean = feature.mean(axis=0)          # (320,)
    std = feature.std(axis=0)            # (320,)
    std = np.where(std < 1e-6, 1.0, std)
    feature = (feature - mean) / std

    return feature.astype(np.float32)


def augment_sample(
    frames: np.ndarray,
    n_augments: int = 4,
) -> list[np.ndarray]:
    """
    Génère des variantes augmentées d'un sample par trim aléatoire du début.

    Pour chaque variante, coupe entre 1 et MAX_TRIM_START frames au début,
    puis rééchantillonne vers SEQUENCE_LENGTH frames. L'original (sans coupe)
    est inclus en tête de liste.

    Args:
        frames:     shape (64, 318) — sample brut (avant build_feature_vector).
        n_augments: nombre de variantes augmentées à générer (hors original).

    Returns:
        Liste de n_augments + 1 arrays de shape (64, 318), incluant l'original.
    """
    result: list[np.ndarray] = [frames]  # original en premier

    for _ in range(n_augments):
        trim_start = np.random.randint(1, MAX_TRIM_START + 1)
        trimmed = frames[trim_start:]                        # (64-trim_start, 318)
        resampled = _resample(trimmed, SEQUENCE_LENGTH)      # (64, 318)
        result.append(resampled)

    return result

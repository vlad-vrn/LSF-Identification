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

# Indices des landmarks d'épaules dans la pose MediaPipe (33 landmarks).
# Utilisés comme repère corporel pour la normalisation géométrique.
_LEFT_SHOULDER_IDX = 11
_RIGHT_SHOULDER_IDX = 12

# Visibilité minimale d'un landmark pose pour qu'il serve d'ancre fiable.
_SHOULDER_MIN_VIS = 0.2

# Largeur d'épaules plancher (en unités normalisées [0,1]) pour éviter une
# division par ~0 quand le signeur est de profil ou mal détecté.
_MIN_SHOULDER_WIDTH = 0.05


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


def resample_sequence(frames: np.ndarray, target_length: int) -> np.ndarray:
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
    return resample_sequence(trimmed, SEQUENCE_LENGTH)


def _shoulder_anchor_scale(pose: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Calcule l'ancre de recentrage (milieu des épaules par frame) et l'échelle
    (largeur d'épaules, médiane sur la séquence) à partir de la pose.

    Le repère épaules est utilisé car ces deux landmarks sont stables et
    bien détectés ; ils définissent un référentiel corporel invariant à la
    position du signeur dans le cadre et à sa distance à la caméra.

    Args:
        pose: shape (64, 132) — 33 landmarks × (x, y, z, vis).

    Returns:
        (anchor_xy, scale) :
          anchor_xy : shape (64, 2) — milieu des épaules (x, y) par frame.
          scale     : scalaire — largeur d'épaules médiane (planchée).
    """
    pose_lm = pose.reshape(pose.shape[0], 33, 4)        # (64, 33, 4)
    left = pose_lm[:, _LEFT_SHOULDER_IDX]                # (64, 4)
    right = pose_lm[:, _RIGHT_SHOULDER_IDX]              # (64, 4)

    mid_xy = (left[:, :2] + right[:, :2]) / 2.0          # (64, 2)

    # Largeur d'épaules par frame, restreinte aux frames où les deux épaules
    # sont suffisamment visibles (sinon la valeur est ignorée pour la médiane).
    width = np.linalg.norm(left[:, :2] - right[:, :2], axis=1)   # (64,)
    visible = (left[:, 3] > _SHOULDER_MIN_VIS) & (right[:, 3] > _SHOULDER_MIN_VIS)
    valid_widths = width[visible]

    if valid_widths.size > 0:
        scale = float(np.median(valid_widths))
    else:
        # Pose non fiable : échelle neutre, l'invariance d'échelle est perdue
        # mais on évite une division aberrante.
        scale = 1.0
    scale = max(scale, _MIN_SHOULDER_WIDTH)

    return mid_xy.astype(np.float32), scale


def normalize_geometry(
    pose: np.ndarray,
    dominant: np.ndarray,
    passive: np.ndarray,
    face: np.ndarray,
    dom_flags: np.ndarray,
    pas_flags: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalisation corps-relative PAR FRAME (remplace l'ancien z-score par feature).

    Chaque frame est recentrée sur le milieu de ses épaules et mise à l'échelle
    par la largeur d'épaules de la séquence. Résultat : coordonnées exprimées en
    « largeurs d'épaules » autour du torse, invariantes à la position et à la
    taille apparente du signeur — ce qui aide la généralisation à un nouveau
    signeur (cf. split par contributeur).

    - x, y : recentrés (ancre) puis divisés par l'échelle.
    - z     : divisé par l'échelle (profondeur relative), pas de recentrage.
    - vis (pose) et 4e colonne face : inchangés.
    - Les mains absentes (flag=0) restent exactement à 0 pour rester cohérentes
      avec les flags de présence.

    Args:
        pose:      (64, 132)   dominant/passive: (64, 63)   face: (64, 60)
        dom_flags, pas_flags: (64,) — 1.0 si la main est présente sur la frame.

    Returns:
        (pose, dominant, passive, face) normalisés, mêmes shapes.
    """
    anchor, scale = _shoulder_anchor_scale(pose)   # (64,2), scalaire
    ax = anchor[:, 0:1]   # (64,1) — broadcast sur les landmarks
    ay = anchor[:, 1:2]

    def _norm_xyz(block: np.ndarray, n_lm: int, stride: int) -> np.ndarray:
        """Recentre/échelle les x,y,z d'un bloc (64, n_lm*stride), garde les autres colonnes."""
        out = block.reshape(block.shape[0], n_lm, stride).copy()
        out[:, :, 0] = (out[:, :, 0] - ax) / scale   # x
        out[:, :, 1] = (out[:, :, 1] - ay) / scale   # y
        out[:, :, 2] = out[:, :, 2] / scale          # z (relatif)
        return out.reshape(block.shape[0], n_lm * stride)

    pose_n = _norm_xyz(pose, 33, 4)   # vis (col 3) laissée intacte par _norm_xyz
    face_n = _norm_xyz(face, 15, 4)   # 4e colonne (0.0) laissée intacte
    dom_n = _norm_xyz(dominant, 21, 3)
    pas_n = _norm_xyz(passive, 21, 3)

    # Réimpose les zéros sur les mains absentes (le recentrage les avait décalées)
    dom_n[dom_flags == 0.0] = 0.0
    pas_n[pas_flags == 0.0] = 0.0

    return pose_n, dom_n, pas_n, face_n


def build_feature_vector(frames: np.ndarray) -> np.ndarray:
    """
    Construit le vecteur feature normalisé à partir d'un sample brut.

    Pipeline :
      1. Trim des frames de repos (début/fin)
      2. Détection de la main dominante par mouvement
      3. Calcul des flags de présence
      4. Normalisation géométrique corps-relative par frame (épaules)
      5. Reconstruction : pose(132) + dominant(63) + passive(63) + face(60) + flags(2)

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

    # Normalisation géométrique corps-relative (remplace le z-score par feature).
    # Les flags binaires 0/1 ne sont PAS normalisés : leur sémantique est conservée.
    pose, dominant, passive, face = normalize_geometry(
        pose, dominant, passive, face, dom_flags, pas_flags
    )

    # Concatène dans l'ordre final : pose + dominant + passive + face + flags
    feature = np.concatenate([
        pose,
        dominant,
        passive,
        face,
        dom_flags[:, np.newaxis],   # (64, 1)
        pas_flags[:, np.newaxis],   # (64, 1)
    ], axis=1)  # (64, 320)

    return feature.astype(np.float32)


def augment_noise(frames: np.ndarray, std: float = 0.003) -> np.ndarray:
    """
    Ajoute du bruit gaussien aux keypoints pour simuler le jitter MediaPipe.

    Les frames où une main est entièrement absente (convention : tous les floats
    exactement à 0) ne sont pas bruitées pour ne pas créer de fausse détection
    de présence lors du calcul des flags dans build_feature_vector.

    Args:
        frames: shape (N, 318) — sample brut.
        std:    écart-type du bruit (défaut 0.003 ≈ 0.3% de la plage [0, 1]).

    Returns:
        shape (N, 318), float32.
    """
    result = frames.copy()
    noise = np.random.normal(0, std, frames.shape).astype(np.float32)
    result += noise

    # Remet à zéro les frames des mains absentes pour conserver la convention
    left_absent = np.all(frames[:, _LEFT_START:_LEFT_END] == 0, axis=1)   # (N,) bool
    right_absent = np.all(frames[:, _RIGHT_START:_RIGHT_END] == 0, axis=1)
    result[left_absent,  _LEFT_START:_LEFT_END]   = 0.0
    result[right_absent, _RIGHT_START:_RIGHT_END] = 0.0

    return result


def augment_mirror(frames: np.ndarray) -> np.ndarray:
    """
    Augmentation miroir : échange les colonnes main gauche et main droite.

    Simule l'effet miroir de la webcam (comportement navigateur par défaut)
    et la variabilité gaucher/droitier. La détection de main dominante dans
    build_feature_vector absorbera naturellement cet échange.

    Args:
        frames: shape (N, 318) — sample brut.

    Returns:
        shape (N, 318), float32, mains échangées.
    """
    result = frames.copy()
    left  = frames[:, _LEFT_START:_LEFT_END].copy()
    right = frames[:, _RIGHT_START:_RIGHT_END].copy()
    result[:, _LEFT_START:_LEFT_END]   = right
    result[:, _RIGHT_START:_RIGHT_END] = left
    return result


def augment_sample(
    frames: np.ndarray,
    n_augments: int = 4,
) -> list[np.ndarray]:
    """
    Génère des variantes augmentées par combinaison de quatre transformations :

    1. Trim aléatoire du début (0–MAX_TRIM_START frames) — simule un démarrage
       du signe à différents moments dans la fenêtre.
    2. Jitter de vitesse (50 % de chance) : prend 70–100 % des frames puis
       rééchantillonne à 64 — simule un signe exécuté plus rapidement.
    3. Bruit gaussien faible (toujours) — simule le jitter de détection MediaPipe.
    4. Augmentation miroir (30 % de chance) — échange main gauche et droite
       pour la robustesse à l'effet miroir webcam et aux gauchers.

    L'original est toujours inclus en premier, sans aucune transformation.

    Args:
        frames:     shape (64, 318) — sample brut (avant build_feature_vector).
        n_augments: nombre de variantes à générer (hors original).

    Returns:
        Liste de n_augments + 1 arrays de shape (64, 318).
    """
    result: list[np.ndarray] = [frames]

    for _ in range(n_augments):
        aug = frames.copy()

        # 1. Trim aléatoire du début
        trim_start = np.random.randint(0, MAX_TRIM_START + 1)
        if trim_start > 0:
            aug = aug[trim_start:]
            aug = resample_sequence(aug, SEQUENCE_LENGTH)

        # 2. Jitter de vitesse : prend une fraction des frames (signe plus rapide)
        if np.random.random() < 0.5:
            factor = np.random.uniform(0.7, 1.0)
            n_keep = max(MIN_FRAMES_AFTER_TRIM, round(len(aug) * factor))
            aug = resample_sequence(aug[:n_keep], SEQUENCE_LENGTH)

        # 3. Bruit gaussien (simule le jitter MediaPipe)
        aug = augment_noise(aug, std=0.003)

        # 4. Miroir (simule webcam non miroir et gauchers)
        if np.random.random() < 0.3:
            aug = augment_mirror(aug)

        result.append(aug)

    return result

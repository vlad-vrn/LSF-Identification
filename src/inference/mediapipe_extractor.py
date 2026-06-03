"""
Extraction de keypoints depuis un flux webcam via MediaPipe Holistic.

Produit un vecteur (318,) float32 par frame, dans le même format que
le dataset : pose(132) + left_hand(63) + right_hand(63) + face(60).

Note sur les coordonnées : le dataset a été collecté via une interface web
(MediaPipe JS). La librairie Python utilise le même espace normalisé [0, 1]
pour x/y. Les coordonnées z et les valeurs de visibilité (pose) peuvent
légèrement différer selon la version de MediaPipe, mais le modèle est
robuste à ces variations après z-score par séquence.

Note sur la face : le dataset stocke 60 valeurs (15 landmarks × 4 floats).
MediaPipe Holistic retourne 468 landmarks faciaux avec (x, y, z) seulement.
On sélectionne 15 landmarks-clés pour l'expression faciale (sourcils, bouche,
yeux) et on ajoute 0.0 comme 4ème valeur, cohérent avec le format du dataset.
"""

import logging

import cv2
import mediapipe as mp
import numpy as np

logger = logging.getLogger(__name__)

# Dimensions cohérentes avec loader.py
POSE_DIM = 132      # 33 landmarks × (x, y, z, vis)
LEFT_HAND_DIM = 63  # 21 landmarks × (x, y, z)
RIGHT_HAND_DIM = 63
FACE_DIM = 60       # 15 landmarks × (x, y, z, 0.0)
FEATURE_DIM = 318

# 15 indices de landmarks faciaux MediaPipe couvrant les zones grammaticalement
# pertinentes en LSF : sourcils (46, 105, 334, 276), yeux (33, 263),
# nez (1, 4), bouche (61, 291, 0, 17), joues (117, 346, 152).
FACE_LANDMARK_INDICES: list[int] = [46, 105, 334, 276, 33, 263, 1, 4, 61, 291, 0, 17, 117, 346, 152]


def make_holistic(
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> mp.solutions.holistic.Holistic:
    """
    Crée une instance MediaPipe Holistic configurée pour la démonstration temps réel.

    Args:
        min_detection_confidence: Seuil de détection initiale (défaut 0.5).
        min_tracking_confidence:  Seuil de suivi inter-frames (défaut 0.5).

    Returns:
        Instance Holistic prête à être utilisée comme context manager.
    """
    return mp.solutions.holistic.Holistic(
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
        # model_complexity=1 : bon compromis précision/latence sur CPU
        model_complexity=1,
    )


def extract_keypoints(results: mp.solutions.holistic.Holistic) -> np.ndarray:
    """
    Convertit les résultats MediaPipe Holistic en vecteur feature (318,).

    Les landmarks absents (main non détectée, pose non détectée) sont
    remplacés par des zéros — même convention que le dataset.

    Args:
        results: Objet résultat de holistic.process(frame_rgb).

    Returns:
        np.ndarray float32 de shape (318,) :
        pose(132) + left_hand(63) + right_hand(63) + face(60).
    """
    # --- Pose : 33 landmarks × (x, y, z, visibility) = 132 ---
    if results.pose_landmarks:
        pose = np.array(
            [[lm.x, lm.y, lm.z, lm.visibility]
             for lm in results.pose_landmarks.landmark],
            dtype=np.float32,
        ).flatten()
    else:
        pose = np.zeros(POSE_DIM, dtype=np.float32)

    # --- Main gauche : 21 landmarks × (x, y, z) = 63 ---
    if results.left_hand_landmarks:
        left_hand = np.array(
            [[lm.x, lm.y, lm.z]
             for lm in results.left_hand_landmarks.landmark],
            dtype=np.float32,
        ).flatten()
    else:
        left_hand = np.zeros(LEFT_HAND_DIM, dtype=np.float32)

    # --- Main droite : 21 landmarks × (x, y, z) = 63 ---
    if results.right_hand_landmarks:
        right_hand = np.array(
            [[lm.x, lm.y, lm.z]
             for lm in results.right_hand_landmarks.landmark],
            dtype=np.float32,
        ).flatten()
    else:
        right_hand = np.zeros(RIGHT_HAND_DIM, dtype=np.float32)

    # --- Face : 15 landmarks sélectionnés × (x, y, z, 0.0) = 60 ---
    if results.face_landmarks:
        all_lms = results.face_landmarks.landmark
        face = np.array(
            [[all_lms[i].x, all_lms[i].y, all_lms[i].z, 0.0]
             for i in FACE_LANDMARK_INDICES],
            dtype=np.float32,
        ).flatten()
    else:
        face = np.zeros(FACE_DIM, dtype=np.float32)

    return np.concatenate([pose, left_hand, right_hand, face])


def process_frame(
    frame_bgr: np.ndarray,
    holistic: mp.solutions.holistic.Holistic,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Traite une frame BGR (sortie OpenCV) et retourne keypoints + frame annotée.

    Args:
        frame_bgr: Frame BGR shape (H, W, 3) issue de cv2.VideoCapture.
        holistic:  Instance MediaPipe Holistic déjà créée.

    Returns:
        (keypoints, annotated_frame) :
            keypoints       — np.ndarray float32 shape (318,)
            annotated_frame — frame BGR avec squelette dessiné
    """
    # MediaPipe attend du RGB
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_rgb.flags.writeable = False

    results = holistic.process(frame_rgb)

    frame_rgb.flags.writeable = True
    annotated = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    _draw_landmarks(annotated, results)

    return extract_keypoints(results), annotated


def _draw_landmarks(frame: np.ndarray, results) -> None:
    """Dessine les landmarks MediaPipe sur la frame (modification en place)."""
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles
    mp_holistic = mp.solutions.holistic

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style(),
        )
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
        )
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
        )

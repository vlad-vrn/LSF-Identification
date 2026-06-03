"""
Chargement du dataset LSF depuis dataset/approved/.

Chaque sample = un .parquet (64 frames) + un .meta.json (contributor_id).
Seul le dossier approved/ est chargé — raw/ et tout autre dossier sont ignorés.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Dimensions des colonnes brutes du parquet (concaténées dans l'ordre)
POSE_DIM = 132       # 33 landmarks × (x, y, z, vis)
LEFT_HAND_DIM = 63   # 21 landmarks × (x, y, z)
RIGHT_HAND_DIM = 63  # 21 landmarks × (x, y, z)
FACE_DIM = 60        # 15 landmarks × (x, y, z, ?)

FEATURE_DIM = POSE_DIM + LEFT_HAND_DIM + RIGHT_HAND_DIM + FACE_DIM  # 318
SEQUENCE_LENGTH = 64


def get_label_map(dataset_dir: str | Path) -> dict[str, int]:
    """
    Construit le mapping {slug: idx} à partir des sous-dossiers de approved/.

    Les labels sont triés alphabétiquement pour garantir la reproductibilité :
    le même mapping est produit quel que soit l'ordre d'exploration du disque.

    Args:
        dataset_dir: Chemin vers le dossier dataset/ (qui contient approved/).

    Returns:
        Dict {slug: idx} avec idx de 0 à N-1.

    Raises:
        FileNotFoundError: Si approved/ n'existe pas.
    """
    approved_dir = Path(dataset_dir) / "approved"
    if not approved_dir.exists():
        raise FileNotFoundError(f"Dossier approved/ introuvable : {approved_dir}")

    slugs = sorted(p.name for p in approved_dir.iterdir() if p.is_dir())
    return {slug: idx for idx, slug in enumerate(slugs)}


def load_dataset(dataset_dir: str | Path) -> list[dict]:
    """
    Charge tous les samples depuis dataset_dir/approved/<slug>/*.parquet.

    Pour chaque UUID trouvé, lit le .parquet et le .meta.json correspondant.
    Les frames sont concaténées en un vecteur de 318 floats par frame :
    pose(132) + left_hand(63) + right_hand(63) + face(60).

    Args:
        dataset_dir: Chemin vers le dossier dataset/.

    Returns:
        Liste de dicts, chacun de la forme :
        {
            "label": str,           # slug du signe, ex: "bonjour"
            "label_idx": int,       # index numérique selon get_label_map()
            "frames": np.ndarray,   # shape (64, 318), float32
            "contributor_id": str   # UUID du contributeur (pour le split)
        }

    Raises:
        FileNotFoundError: Si approved/ n'existe pas.
        ValueError: Si un parquet ne contient pas exactement 64 frames.
    """
    label_map = get_label_map(dataset_dir)
    approved_dir = Path(dataset_dir) / "approved"

    samples: list[dict] = []

    for sign_dir in sorted(approved_dir.iterdir()):
        if not sign_dir.is_dir():
            continue

        label = sign_dir.name
        label_idx = label_map[label]
        sign_samples: list[dict] = []

        for parquet_path in sorted(sign_dir.glob("*.parquet")):
            sample = _load_single(parquet_path, label, label_idx)
            sign_samples.append(sample)

        logger.info("  %s : %d samples", label, len(sign_samples))
        samples.extend(sign_samples)

    logger.info("Total : %d samples, %d signes", len(samples), len(label_map))
    return samples


def _load_single(parquet_path: Path, label: str, label_idx: int) -> dict:
    """
    Charge un fichier .parquet + son .meta.json associé en un sample dict.

    Args:
        parquet_path: Chemin vers le fichier .parquet.
        label: Slug du signe (nom du dossier parent).
        label_idx: Index numérique du label.

    Returns:
        Dict avec clés "label", "label_idx", "frames", "contributor_id".

    Raises:
        ValueError: Si le parquet ne contient pas exactement SEQUENCE_LENGTH frames.
        FileNotFoundError: Si le .meta.json associé est manquant.
    """
    meta_path = parquet_path.with_suffix(".meta.json")

    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json manquant pour {parquet_path.name}")

    with meta_path.open(encoding="utf-8") as f:
        meta = json.load(f)
    contributor_id: str = meta["contributor_id"]

    df = pd.read_parquet(parquet_path)

    if len(df) != SEQUENCE_LENGTH:
        raise ValueError(
            f"{parquet_path.name} : attendu {SEQUENCE_LENGTH} frames, "
            f"trouvé {len(df)}"
        )

    # Empile chaque colonne en matrice puis concatène dans l'ordre
    pose = np.stack(df["pose"].values).astype(np.float32)          # (64, 132)
    left_hand = np.stack(df["left_hand"].values).astype(np.float32)  # (64, 63)
    right_hand = np.stack(df["right_hand"].values).astype(np.float32)  # (64, 63)
    face = np.stack(df["face"].values).astype(np.float32)           # (64, 60)

    frames = np.concatenate([pose, left_hand, right_hand, face], axis=1)  # (64, 318)

    return {
        "label": label,
        "label_idx": label_idx,
        "frames": frames,
        "contributor_id": contributor_id,
    }

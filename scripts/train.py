"""
Point d'entrée pour l'entraînement du modèle LSF.

Usage :
    python scripts/train.py
    python scripts/train.py --epochs 100 --lr 0.001 --save models/lsf_v2.pt

Pour restreindre l'entraînement à un sous-ensemble de signes, modifier
ALLOWED_SIGNS ci-dessous (None = tous les signes).
"""

import argparse
import logging
import sys
from pathlib import Path

# Ajoute src/ au path pour les imports relatifs
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.loader import get_label_map, load_dataset
from model.train import N_EPOCHS, LR, train

# Restreint l'entraînement aux 5 signes les mieux représentés.
# Mettre à None pour entraîner sur tous les signes disponibles.
ALLOWED_SIGNS: list[str] | None = ["soif", "boire", "aujourd_hui", "aimer", "aller"]


def _filter_samples(
    samples: list[dict],
    allowed: list[str],
) -> tuple[list[dict], dict[str, int]]:
    """
    Filtre les samples et reconstruit un label_map contigu (0..N-1).

    Args:
        samples: Liste complète issue de load_dataset().
        allowed: Slugs des signes à conserver, triés alphabétiquement.

    Returns:
        (samples_filtrés, label_map) avec des indices remappés depuis 0.
    """
    allowed_set = set(allowed)
    filtered = [s for s in samples if s["label"] in allowed_set]

    # Tri alphabétique pour reproductibilité — cohérent avec get_label_map()
    new_label_map = {slug: idx for idx, slug in enumerate(sorted(allowed_set))}

    # Réindexe label_idx dans chaque sample
    for s in filtered:
        s["label_idx"] = new_label_map[s["label"]]

    return filtered, new_label_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entraîne le classifieur LSF 1D-CNN")
    parser.add_argument(
        "--dataset",
        default="dataset",
        help="Chemin vers le dossier dataset/ (défaut : dataset/)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=N_EPOCHS,
        help=f"Nombre d'epochs (défaut : {N_EPOCHS})",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=LR,
        help=f"Learning rate Adam (défaut : {LR})",
    )
    parser.add_argument(
        "--save",
        default="models/lsf_v1.pt",
        help="Chemin de sauvegarde du modèle (défaut : models/lsf_v1.pt)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        logging.error("Dossier dataset introuvable : %s", dataset_path)
        sys.exit(1)

    logging.info("Chargement du dataset depuis %s", dataset_path)
    samples = load_dataset(dataset_path)

    if ALLOWED_SIGNS is not None:
        samples, label_map = _filter_samples(samples, ALLOWED_SIGNS)
        logging.info("Filtre actif — %d signes retenus : %s", len(label_map), sorted(label_map))
    else:
        label_map = get_label_map(dataset_path)

    logging.info("%d samples, %d signes : %s", len(samples), len(label_map), list(label_map.keys()))

    train(
        samples=samples,
        label_map=label_map,
        n_epochs=args.epochs,
        lr=args.lr,
        save_path=args.save,
    )


if __name__ == "__main__":
    main()

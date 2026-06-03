"""
Point d'entrée pour la démonstration temps réel LSF.

Usage :
    python scripts/run.py
    python scripts/run.py --model models/lsf_v1.pt --camera 0

Raccourcis :
    Q : quitter
    C : effacer la phrase accumulée
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from display.overlay import draw_help, draw_overlay
from inference.mediapipe_extractor import make_holistic, process_frame
from inference.pipeline import (
    deduplicate,
    get_dedup_progress,
    make_inference_state,
    push_frame,
    run_inference,
    should_run_inference,
    DEDUP_COUNT,
    SEQUENCE_LENGTH,
)
from model.classifier import make_classifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Démonstration LSF temps réel")
    parser.add_argument(
        "--model",
        default="models/lsf_v1.pt",
        help="Chemin vers le checkpoint du modèle (défaut : models/lsf_v1.pt)",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Index de la webcam (défaut : 0)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.6,
        help="Seuil de confiance pour la déduplication (défaut : 0.6)",
    )
    return parser.parse_args()


def load_model(
    model_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, int]]:
    """
    Charge le modèle et le label_map depuis un checkpoint.

    Args:
        model_path: Chemin vers le fichier .pt sauvegardé par train().
        device:     Dispositif de calcul.

    Returns:
        (model, label_map) prêts pour l'inférence.
    """
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    label_map: dict[str, int] = checkpoint["label_map"]
    n_classes: int = checkpoint["n_classes"]

    model = make_classifier(n_classes=n_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, label_map


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    model_path = Path(args.model)
    if not model_path.exists():
        logging.error("Modèle introuvable : %s", model_path)
        logging.error("Lancez d'abord : python scripts/train.py")
        sys.exit(1)

    device = torch.device("cpu")
    logging.info("Chargement du modèle depuis %s", model_path)
    model, label_map = load_model(model_path, device)
    logging.info("Modèle chargé — %d classes : %s", len(label_map), list(label_map.keys()))

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        logging.error("Impossible d'ouvrir la caméra %d", args.camera)
        sys.exit(1)

    state = make_inference_state()
    phrase: list[str] = []
    current_label: str | None = None
    current_confidence: float = 0.0
    current_uncertainty: float = 1.0   # commence à 1.0 = totalement incertain
    gate_active: bool = False

    logging.info("Démonstration démarrée. Q = quitter, C = effacer la phrase.")

    with make_holistic() as holistic:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                logging.warning("Frame manquante — fin du flux caméra.")
                break

            keypoints, annotated = process_frame(frame_bgr, holistic)
            state = push_frame(state, keypoints)

            gate_active = should_run_inference(state)
            if gate_active:
                result = run_inference(model, state, label_map, device)
                if result is not None:
                    label, confidence, uncertainty = result
                    current_label = label
                    current_confidence = confidence
                    current_uncertainty = uncertainty

                    confirmed = deduplicate(state, label, confidence, threshold=args.confidence)
                    if confirmed is not None:
                        phrase.append(confirmed)
                        logging.info(
                            "Signe confirme : %s (conf=%.2f, incertitude=%.2f)",
                            confirmed, confidence, uncertainty,
                        )

            dedup_count, _ = get_dedup_progress(state)

            display = draw_overlay(
                annotated,
                current_label=current_label,
                confidence=current_confidence,
                uncertainty=current_uncertainty,
                phrase=phrase,
                buffer_fill=len(state["buffer"]),
                dedup_count=dedup_count,
                dedup_max=DEDUP_COUNT,
                buffer_max=SEQUENCE_LENGTH,
                active=gate_active,
            )

            cv2.imshow("LSF Recognition", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c"):
                phrase.clear()
                logging.info("Phrase effacée.")

    cap.release()
    cv2.destroyAllWindows()
    logging.info("Session terminée. Phrase : %s", " ".join(phrase) if phrase else "(vide)")


if __name__ == "__main__":
    main()

"""
Point d'entrée pour la démonstration temps réel LSF — mode CAPTURE CADENCÉE.

L'application impose un rythme (métronome) : à chaque battement, l'utilisateur
est invité à réaliser UN signe pendant une fenêtre de capture, puis le modèle
prédit une fois. Cela synchronise l'utilisateur sur la fenêtre attendue par le
modèle et supprime l'ancienne détection continue.

Usage :
    python scripts/run.py
    python scripts/run.py --model models/lsf_v1.pt --camera 0
    python scripts/run.py --capture-frames 64 --prepare-frames 60
    python scripts/run.py --api-key sk-ant-...

Raccourcis :
    Q : quitter
    C : effacer les signes et la phrase construite
    S : construire une phrase française avec NLP (Anthropic API)
"""

import argparse
import logging
import sys
import threading
from pathlib import Path

import cv2
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from display.overlay import draw_overlay
from inference.mediapipe_extractor import make_holistic, process_frame
from inference.pipeline import (
    CAPTURE_FRAMES,
    CONFIDENCE_THRESHOLD,
    PREPARE_FRAMES,
    REST_FRAMES,
    RESULT_FRAMES,
    make_cadence_state,
    step_cadence,
)
from model.classifier import make_classifier
from nlp.sentence_builder import build_sentence

# Largeur d'affichage en pixels — la frame est redimensionnée à cette largeur.
DISPLAY_WIDTH: int = 960


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Démonstration LSF — capture cadencée")
    parser.add_argument("--model", default="models/lsf_v1.pt",
                        help="Chemin vers le checkpoint (défaut : models/lsf_v1.pt)")
    parser.add_argument("--camera", type=int, default=0, help="Index webcam (défaut : 0)")
    parser.add_argument("--confidence", type=float, default=CONFIDENCE_THRESHOLD,
                        help=f"Seuil de confiance (défaut : {CONFIDENCE_THRESHOLD})")
    parser.add_argument("--prepare-frames", type=int, default=PREPARE_FRAMES,
                        help=f"Durée du décompte en frames (défaut : {PREPARE_FRAMES})")
    parser.add_argument("--capture-frames", type=int, default=CAPTURE_FRAMES,
                        help=f"Durée de capture en frames (défaut : {CAPTURE_FRAMES})")
    parser.add_argument("--result-frames", type=int, default=RESULT_FRAMES,
                        help=f"Durée d'affichage du résultat (défaut : {RESULT_FRAMES})")
    parser.add_argument("--rest-frames", type=int, default=REST_FRAMES,
                        help=f"Durée de la pause (défaut : {REST_FRAMES})")
    parser.add_argument("--api-key", default=None, dest="api_key",
                        help="Clé API Anthropic (sinon : variable ANTHROPIC_API_KEY)")
    return parser.parse_args()


def load_model(
    model_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, int]]:
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = make_classifier(n_classes=checkpoint["n_classes"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, checkpoint["label_map"]


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
        logging.error("Lancez d'abord : python scripts/train.py --dataset src/data --min-contrib-keep 3")
        sys.exit(1)

    device = torch.device("cpu")
    logging.info("Chargement du modèle...")
    model, label_map = load_model(model_path, device)
    logging.info("Modèle chargé — %d classes : %s", len(label_map), list(label_map.keys()))

    # Sur Windows, le backend par défaut (MSMF) est lent à s'initialiser et
    # peut bloquer plusieurs secondes ; DirectShow s'ouvre quasi instantanément.
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    logging.info("Ouverture de la caméra %d...", args.camera)
    cap = cv2.VideoCapture(args.camera, backend)
    if not cap.isOpened():
        logging.error("Impossible d'ouvrir la caméra %d", args.camera)
        logging.error("Essayez un autre index : --camera 1 (ou 2).")
        sys.exit(1)

    # ── État cadence ─────────────────────────────────────────────────────
    state = make_cadence_state(
        prepare_frames=args.prepare_frames,
        capture_frames=args.capture_frames,
        result_frames=args.result_frames,
        rest_frames=args.rest_frames,
        confidence_threshold=args.confidence,
    )
    phrase: list[str] = []

    # ── État NLP (partagé avec le thread de construction) ────────────────
    nlp: dict = {"building": False, "sentence": None}

    logging.info("Démarré (mode cadencé) — Q=quitter  C=effacer  S=construire phrase")

    with make_holistic() as holistic:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                logging.warning("Fin du flux caméra.")
                break

            keypoints, annotated = process_frame(frame_bgr, holistic)
            status, just_confirmed = step_cadence(state, keypoints, model, label_map, device)

            if just_confirmed is not None:
                phrase.append(just_confirmed)
                nlp["sentence"] = None   # invalide la phrase NLP précédente

            # ── Redimensionnement pour l'affichage ───────────────────────
            h_orig, w_orig = annotated.shape[:2]
            scale = DISPLAY_WIDTH / w_orig
            frame_display = cv2.resize(annotated, (DISPLAY_WIDTH, int(h_orig * scale)),
                                       interpolation=cv2.INTER_LINEAR)

            display = draw_overlay(
                frame_display,
                status=status,
                phrase=phrase,
                constructed_sentence=nlp["sentence"],
                sentence_building=nlp["building"],
            )

            cv2.imshow("LSF Recognition", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            elif key == ord("c"):
                phrase.clear()
                nlp["sentence"] = None
                logging.info("Phrase effacée.")

            elif key == ord("s"):
                if nlp["building"]:
                    logging.info("Construction déjà en cours.")
                elif not phrase:
                    logging.info("Aucun signe à construire.")
                else:
                    snapshot = phrase.copy()
                    nlp["building"] = True
                    nlp["sentence"] = None

                    def _build(signs: list[str], api_key: str | None) -> None:
                        nlp["sentence"] = build_sentence(signs, api_key)
                        nlp["building"] = False

                    threading.Thread(target=_build, args=(snapshot, args.api_key), daemon=True).start()
                    logging.info("Construction NLP démarrée : %s", snapshot)

    cap.release()
    cv2.destroyAllWindows()
    logging.info("Session terminée — signes : %s", " ".join(phrase) if phrase else "(vide)")


if __name__ == "__main__":
    main()

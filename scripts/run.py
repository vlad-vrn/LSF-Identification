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
import os
import sys
import threading
from pathlib import Path

import cv2
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

_PROJECT_ROOT = Path(__file__).parent.parent


def load_dotenv(path: Path) -> None:
    """
    Charge un fichier .env (KEY=VALUE par ligne) dans os.environ.

    Loader minimal sans dépendance : ignore lignes vides et commentaires (#),
    retire les guillemets autour des valeurs. Les variables déjà présentes dans
    l'environnement ont la priorité (un export shell explicite l'emporte sur .env).
    """
    if not path.exists():
        return
    # utf-8-sig : retire un éventuel BOM (PowerShell écrit du UTF-8 avec BOM),
    # sinon la première variable du fichier serait corrompue (﻿KEY).
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


from display.overlay import draw_overlay
from inference.mediapipe_extractor import make_holistic, process_frame
from inference.pipeline import (
    CAPTURE_FRAMES,
    CONFIDENCE_THRESHOLD,
    PREPARE_FRAMES,
    REST_FRAMES,
    RESULT_FRAMES,
    current_status,
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
                        help="Clé API NLP (sinon : LSF_NLP_API_KEY ou ANTHROPIC_API_KEY)")
    parser.add_argument("--nlp-provider", default=None, choices=["anthropic", "openai"],
                        dest="nlp_provider",
                        help="Backend NLP : 'openai' pour nouslabs/compatible OpenAI, "
                             "'anthropic' pour Claude (défaut : auto selon --nlp-base-url)")
    parser.add_argument("--nlp-base-url", default=None, dest="nlp_base_url",
                        help="URL de base OpenAI-compatible (ex: nouslabs), termine par /v1")
    parser.add_argument("--nlp-model", default=None, dest="nlp_model",
                        help="ID du modèle NLP (selon ton provider)")
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
    # Charge .env à la racine du projet avant tout (clés API, config NLP)
    load_dotenv(_PROJECT_ROOT / ".env")

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

    # Détection en pause pendant la génération de phrase (et jusqu'à reprise).
    paused: bool = False

    # ── État NLP (partagé avec le thread de construction) ────────────────
    nlp: dict = {"building": False, "sentence": None}

    logging.info("Démarré (mode cadencé) — S=construire (met en pause)  ESPACE=reprendre  C=effacer  Q=quitter")

    with make_holistic() as holistic:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                logging.warning("Fin du flux caméra.")
                break

            keypoints, annotated = process_frame(frame_bgr, holistic)

            # En pause : on n'avance plus la cadence (ni capture, ni inférence),
            # on fige le dernier état affiché.
            if paused:
                status = current_status(state)
            else:
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
                paused=paused,
            )

            cv2.imshow("LSF Recognition", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            elif key == ord("c"):
                phrase.clear()
                nlp["sentence"] = None
                logging.info("Phrase effacée.")

            elif key == ord(" "):
                # Reprise de la détection après une génération
                if paused:
                    paused = False
                    state = make_cadence_state(
                        prepare_frames=args.prepare_frames,
                        capture_frames=args.capture_frames,
                        result_frames=args.result_frames,
                        rest_frames=args.rest_frames,
                        confidence_threshold=args.confidence,
                    )
                    logging.info("Détection reprise.")

            elif key == ord("s"):
                if nlp["building"]:
                    logging.info("Construction déjà en cours.")
                elif not phrase:
                    logging.info("Aucun signe à construire.")
                else:
                    # Met la détection en pause pendant la génération
                    paused = True
                    snapshot = phrase.copy()
                    nlp["building"] = True
                    nlp["sentence"] = None

                    def _build(signs: list[str]) -> None:
                        nlp["sentence"] = build_sentence(
                            signs,
                            api_key=args.api_key,
                            provider=args.nlp_provider,
                            base_url=args.nlp_base_url,
                            model=args.nlp_model,
                        )
                        nlp["building"] = False

                    threading.Thread(target=_build, args=(snapshot,), daemon=True).start()
                    logging.info("Construction NLP démarrée (détection en pause) : %s", snapshot)

    cap.release()
    cv2.destroyAllWindows()
    logging.info("Session terminée — signes : %s", " ".join(phrase) if phrase else "(vide)")


if __name__ == "__main__":
    main()

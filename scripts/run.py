"""
Point d'entrée pour la démonstration temps réel LSF.

Usage :
    python scripts/run.py
    python scripts/run.py --model models/lsf_v1.pt --camera 0
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
    CONFIDENCE_MARGIN,
    CONFIDENCE_THRESHOLD,
    DEDUP_COUNT,
    SEQUENCE_LENGTH,
    UNCERTAINTY_GATE,
    deduplicate,
    get_dedup_progress,
    make_inference_state,
    push_frame,
    run_inference,
    should_run_inference,
)
from model.classifier import make_classifier
from nlp.sentence_builder import build_sentence

# Largeur de l'affichage en pixels — le frame est redimensionné à cette largeur,
# la hauteur suit le ratio de la caméra.
DISPLAY_WIDTH: int = 960


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Démonstration LSF temps réel")
    parser.add_argument("--model", default="models/lsf_v1.pt",
                        help="Chemin vers le checkpoint (défaut : models/lsf_v1.pt)")
    parser.add_argument("--camera", type=int, default=0,
                        help="Index webcam (défaut : 0)")
    parser.add_argument("--confidence", type=float, default=CONFIDENCE_THRESHOLD,
                        help=f"Seuil de confiance pour la dédup (défaut : {CONFIDENCE_THRESHOLD})")
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


def _compute_ui_state(
    gate_active: bool,
    dedup_count: int,
    cooldown_frames: int,
    label: str | None,
    confidence: float,
    uncertainty: float,
    margin: float,
    conf_thresh: float,
) -> str:
    """Dérive la chaîne d'état affichée dans la bannière."""
    if cooldown_frames > 0:
        return "CONFIRME !"
    if not gate_active:
        return "EN ATTENTE"
    quality_ok = (
        label is not None
        and confidence >= conf_thresh
        and uncertainty < UNCERTAINTY_GATE
        and margin >= CONFIDENCE_MARGIN
    )
    if quality_ok:
        return f"CONFIRMATION  {dedup_count}/{DEDUP_COUNT}" if dedup_count > 0 else "DETECTION"
    return "MOUVEMENT DETECTE"


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
    logging.info("Chargement du modèle...")
    model, label_map = load_model(model_path, device)
    logging.info("Modèle chargé — %d classes : %s", len(label_map), list(label_map.keys()))

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        logging.error("Impossible d'ouvrir la caméra %d", args.camera)
        sys.exit(1)

    # ── État reconnaissance ──────────────────────────────────────────────
    state = make_inference_state()
    phrase: list[str] = []
    current_label: str | None = None
    current_confidence: float = 0.0
    current_uncertainty: float = 1.0
    current_margin: float = 0.0
    current_top3: list[tuple[str, float]] = []
    gate_active: bool = False

    # ── État NLP (partagé avec le thread de construction) ────────────────
    nlp: dict = {"building": False, "sentence": None}

    logging.info("Démarré — Q=quitter  C=effacer  S=construire phrase")

    with make_holistic() as holistic:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                logging.warning("Fin du flux caméra.")
                break

            keypoints, annotated = process_frame(frame_bgr, holistic)
            state = push_frame(state, keypoints)

            gate_active = should_run_inference(state)
            if gate_active:
                result = run_inference(model, state, label_map, device)
                if result is not None:
                    label, confidence, uncertainty, margin, top3 = result
                    current_label      = label
                    current_confidence = confidence
                    current_uncertainty = uncertainty
                    current_margin     = margin
                    current_top3       = top3

                    confirmed = deduplicate(
                        state, label, confidence, uncertainty, margin,
                        threshold=args.confidence,
                    )
                    if confirmed is not None:
                        phrase.append(confirmed)
                        nlp["sentence"] = None   # invalide la phrase NLP
                        logging.info(
                            "Signe confirmé : %s  (conf=%.2f  cert=%.2f  marge=%.2f)",
                            confirmed, confidence, 1.0 - uncertainty, margin,
                        )

            dedup_count, _, cooldown = get_dedup_progress(state)

            # ── Calcul de l'état UI et des blockers ──────────────────────
            ui_state = _compute_ui_state(
                gate_active, dedup_count, cooldown,
                current_label, current_confidence, current_uncertainty,
                current_margin, args.confidence,
            )
            blockers: dict[str, bool] | None = None
            if gate_active and current_label is not None:
                blockers = {
                    "mvt":   True,
                    "conf":  current_confidence >= args.confidence,
                    "cert":  current_uncertainty < UNCERTAINTY_GATE,
                    "marge": current_margin >= CONFIDENCE_MARGIN,
                }

            # ── Redimensionnement pour l'affichage ───────────────────────
            h_orig, w_orig = annotated.shape[:2]
            scale = DISPLAY_WIDTH / w_orig
            display_h = int(h_orig * scale)
            frame_display = cv2.resize(annotated, (DISPLAY_WIDTH, display_h),
                                       interpolation=cv2.INTER_LINEAR)

            # ── Rendu overlay ─────────────────────────────────────────────
            display = draw_overlay(
                frame_display,
                current_label=current_label,
                confidence=current_confidence,
                uncertainty=current_uncertainty,
                margin=current_margin,
                top3=current_top3,
                phrase=phrase,
                buffer_fill=len(state["buffer"]),
                dedup_count=dedup_count,
                cooldown_frames=cooldown,
                ui_state=ui_state,
                blockers=blockers,
                constructed_sentence=nlp["sentence"],
                sentence_building=nlp["building"],
                dedup_max=DEDUP_COUNT,
                buffer_max=SEQUENCE_LENGTH,
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

                    threading.Thread(
                        target=_build,
                        args=(snapshot, args.api_key),
                        daemon=True,
                    ).start()
                    logging.info("Construction NLP démarrée : %s", snapshot)

    cap.release()
    cv2.destroyAllWindows()
    logging.info("Session terminée — signes : %s", " ".join(phrase) if phrase else "(vide)")


if __name__ == "__main__":
    main()

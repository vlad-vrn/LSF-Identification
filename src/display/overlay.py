"""
Affichage OpenCV pour la démonstration LSF en mode CAPTURE CADENCÉE.

L'interface guide l'utilisateur au rythme du métronome :
  - PREPARE : gros décompte 3-2-1 au centre
  - CAPTURE : bandeau « SIGNEZ ! » + point d'enregistrement + barre de progression
  - RESULT  : signe reconnu (ou « NON RECONNU ») + concurrents
  - REST    : courte pause

Bas de l'écran : signes accumulés, phrase reconstruite (NLP) et raccourcis.
"""

import cv2
import numpy as np

from inference.pipeline import (
    PHASE_CAPTURE,
    PHASE_PREPARE,
    PHASE_REST,
    PHASE_RESULT,
)

# ── Palette (BGR) ──────────────────────────────────────────────────────────
_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)
_GRAY = (120, 120, 120)
_DGRAY = (50, 50, 50)
_GREEN = (60, 210, 60)
_ORANGE = (0, 165, 255)
_RED = (50, 50, 220)
_YELLOW = (0, 220, 220)
_CYAN = (220, 200, 0)

# Couleur et libellé par phase
_PHASE_INFO = {
    PHASE_PREPARE: ("PREPAREZ-VOUS", _ORANGE),
    PHASE_CAPTURE: ("SIGNEZ !", _GREEN),
    PHASE_RESULT:  ("RESULTAT", _CYAN),
    PHASE_REST:    ("...", _GRAY),
}

_PANEL_H = 170
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_overlay(
    frame: np.ndarray,
    status: dict,
    phrase: list[str],
    constructed_sentence: str | None = None,
    sentence_building: bool = False,
) -> np.ndarray:
    """
    Dessine l'interface complète sur la frame.

    Args:
        frame:    image BGR (déjà redimensionnée) avec squelette MediaPipe.
        status:   dict renvoyé par pipeline.step_cadence (phase, progress,
                  countdown, captured, capture_target, result).
        phrase:   liste des signes confirmés.
        constructed_sentence: phrase NLP reconstruite, ou None.
        sentence_building:    True si la construction NLP est en cours.
    """
    out = frame.copy()
    h, w = out.shape[:2]

    _draw_phase_progress_bar(out, status, w)
    _draw_center_cue(out, status, w, h)

    y0 = h - _PANEL_H
    _draw_panel_bg(out, y0, w)
    _draw_phase_banner(out, status, y0, w)
    _draw_result_or_capture(out, status, y0, w)
    cv2.line(out, (12, y0 + 96), (w - 12, y0 + 96), (55, 55, 55), 1)
    _draw_signs_phrase(out, phrase, y0, w)
    _draw_nlp_row(out, constructed_sentence, sentence_building, y0, w)
    _draw_hints(out, y0, w)
    return out


# ── Indicateurs visuels ────────────────────────────────────────────────────

def _draw_phase_progress_bar(frame: np.ndarray, status: dict, w: int) -> None:
    """Barre fine en haut indiquant l'avancement dans la phase courante."""
    _, color = _PHASE_INFO.get(status["phase"], ("", _GRAY))
    bar_w = int(w * status["progress"])
    cv2.rectangle(frame, (0, 0), (bar_w, 5), color, -1)
    cv2.rectangle(frame, (bar_w, 0), (w, 5), _DGRAY, -1)


def _draw_center_cue(frame: np.ndarray, status: dict, w: int, h: int) -> None:
    """Repère central : décompte, bandeau SIGNEZ, ou label résultat."""
    phase = status["phase"]
    cx, cy = w // 2, h // 2 - 40

    if phase == PHASE_PREPARE and status["countdown"] is not None:
        n = str(status["countdown"])
        scale, thick = 5.0, 12
        (tw, th), _ = cv2.getTextSize(n, _FONT, scale, thick)
        cv2.putText(frame, n, (cx - tw // 2, cy + th // 2), _FONT, scale, _ORANGE, thick, cv2.LINE_AA)

    elif phase == PHASE_CAPTURE:
        # Point d'enregistrement clignotant + texte
        blink = (status["captured"] // 5) % 2 == 0
        if blink:
            cv2.circle(frame, (cx - 90, cy - 8), 14, _RED, -1)
        txt = "SIGNEZ"
        scale, thick = 1.6, 4
        (tw, th), _ = cv2.getTextSize(txt, _FONT, scale, thick)
        cv2.putText(frame, txt, (cx - tw // 2 + 20, cy + th // 2), _FONT, scale, _GREEN, thick, cv2.LINE_AA)

    elif phase == PHASE_RESULT and status["result"] is not None:
        res = status["result"]
        if res["accepted"]:
            txt, color = res["label"].upper(), _GREEN
        else:
            txt, color = "NON RECONNU", _RED
        scale, thick = 2.2, 5
        (tw, th), _ = cv2.getTextSize(txt, _FONT, scale, thick)
        cv2.putText(frame, txt, (cx - tw // 2, cy + th // 2), _FONT, scale, color, thick, cv2.LINE_AA)


def _draw_panel_bg(frame: np.ndarray, y0: int, w: int) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y0), (w, frame.shape[0]), _BLACK, -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)


def _draw_phase_banner(frame: np.ndarray, status: dict, y0: int, w: int) -> None:
    """Bandeau de phase en haut du panneau."""
    label, color = _PHASE_INFO.get(status["phase"], ("", _GRAY))
    cv2.putText(frame, label, (14, y0 + 30), _FONT, 0.8, color, 2, cv2.LINE_AA)


def _draw_result_or_capture(frame: np.ndarray, status: dict, y0: int, w: int) -> None:
    """Sous le bandeau : barre de capture (CAPTURE) ou détail du résultat (RESULT)."""
    phase = status["phase"]
    base = y0 + 44

    if phase == PHASE_CAPTURE:
        captured, target = status["captured"], status["capture_target"]
        cv2.putText(frame, f"capture  {captured}/{target} frames", (14, base + 14),
                    _FONT, 0.5, _WHITE, 1, cv2.LINE_AA)
        bx, by, bw, bh = 14, base + 24, w - 28, 12
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), _DGRAY, -1)
        fill = int(bw * captured / max(1, target))
        cv2.rectangle(frame, (bx, by), (bx + fill, by + bh), _GREEN, -1)

    elif phase == PHASE_RESULT and status["result"] is not None:
        res = status["result"]
        # Confiance du meilleur label
        col = _GREEN if res["accepted"] else _RED
        cv2.putText(frame, f"{res['best_label'].upper()}   confiance {res['confidence'] * 100:.0f}%",
                    (14, base + 14), _FONT, 0.6, col, 1, cv2.LINE_AA)
        # Concurrents (top2/top3)
        alts = res["top3"][1:3]
        x = 14
        for lbl, prob in alts:
            txt = f"{lbl}:{prob * 100:.0f}%"
            cv2.putText(frame, txt, (x, base + 38), _FONT, 0.46, _GRAY, 1, cv2.LINE_AA)
            x += cv2.getTextSize(txt, _FONT, 0.46, 1)[0][0] + 22
    else:
        # PREPARE / REST : invite
        msg = "Mettez-vous en position..." if phase == PHASE_PREPARE else ""
        if msg:
            cv2.putText(frame, msg, (14, base + 20), _FONT, 0.5, _GRAY, 1, cv2.LINE_AA)


def _draw_signs_phrase(frame: np.ndarray, phrase: list[str], y0: int, w: int) -> None:
    base = y0 + 96
    if not phrase:
        cv2.putText(frame, "[ aucun signe confirme ]", (14, base + 22),
                    _FONT, 0.52, _GRAY, 1, cv2.LINE_AA)
    else:
        text = "  ".join(s.upper() for s in phrase[-8:])
        cv2.putText(frame, text, (14, base + 22), _FONT, 0.58, _WHITE, 1, cv2.LINE_AA)


def _draw_nlp_row(frame: np.ndarray, sentence: str | None, building: bool, y0: int, w: int) -> None:
    base = y0 + 122
    if building:
        cv2.putText(frame, "Construction en cours...", (14, base + 18),
                    _FONT, 0.52, _ORANGE, 1, cv2.LINE_AA)
    elif sentence:
        display = sentence if len(sentence) <= 80 else sentence[:77] + "..."
        cv2.putText(frame, f"> {display}", (14, base + 18), _FONT, 0.54, _YELLOW, 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "S : construire la phrase", (14, base + 18),
                    _FONT, 0.44, _GRAY, 1, cv2.LINE_AA)


def _draw_hints(frame: np.ndarray, y0: int, w: int) -> None:
    cv2.putText(frame, "Q : quitter      C : effacer signes      S : construire phrase NLP",
                (14, y0 + _PANEL_H - 10), _FONT, 0.37, (75, 75, 75), 1, cv2.LINE_AA)

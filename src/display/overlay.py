"""
Affichage OpenCV pour la démonstration temps réel.

Dessine sur la frame webcam :
- Le signe prédit + barre de certitude (basée sur l'entropie, pas la confiance brute)
- Les points de progression vers la confirmation (●●○)
- La phrase accumulée
- Indicateurs d'état du buffer et du gate de mouvement
"""

import cv2
import numpy as np

# Palette de couleurs BGR
_COLOR_WHITE = (255, 255, 255)
_COLOR_BLACK = (0, 0, 0)
_COLOR_GREEN = (50, 200, 50)
_COLOR_ORANGE = (0, 165, 255)
_COLOR_RED = (50, 50, 220)
_COLOR_GRAY = (120, 120, 120)
_COLOR_ACCENT = (0, 200, 255)    # jaune-orange pour le signe courant

# Géométrie de l'UI
_PANEL_HEIGHT = 130
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_LARGE = 1.1
_FONT_MEDIUM = 0.7
_FONT_SMALL = 0.5
_THICKNESS = 2


def draw_overlay(
    frame: np.ndarray,
    current_label: str | None,
    confidence: float,
    uncertainty: float,
    phrase: list[str],
    buffer_fill: int,
    dedup_count: int,
    dedup_max: int = 3,
    buffer_max: int = 64,
    active: bool = False,
) -> np.ndarray:
    """
    Dessine l'interface complète sur la frame webcam.

    Args:
        frame:         Frame BGR avec landmarks MediaPipe.
        current_label: Dernier label prédit (None si pas encore d'inférence).
        confidence:    Probabilité softmax du meilleur label (0–1).
        uncertainty:   Entropie normalisée (0=certain, 1=aléatoire total).
                       Élevée quand le geste ne correspond à aucun signe connu.
        phrase:        Liste des signes confirmés.
        buffer_fill:   Frames actuellement dans le buffer.
        dedup_count:   Nombre de prédictions consécutives identiques (0–3).
        dedup_max:     Nombre requis pour confirmer (défaut 3).
        buffer_max:    Taille max du buffer (défaut 64).
        active:        True si le gate de mouvement a déclenché l'inférence.

    Returns:
        Frame BGR annotée (nouvelle copie).
    """
    out = frame.copy()
    h, w = out.shape[:2]

    _draw_panel(out, h, w)
    _draw_gate_indicator(out, active, w)
    _draw_buffer_bar(out, buffer_fill, buffer_max, w)

    if current_label is not None:
        _draw_current_sign(out, current_label, confidence, uncertainty, h, w)
        _draw_dedup_dots(out, dedup_count, dedup_max, h, w)

    _draw_phrase(out, phrase, h, w)

    return out


def _draw_panel(frame: np.ndarray, h: int, w: int) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - _PANEL_HEIGHT), (w, h), _COLOR_BLACK, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)


def _draw_gate_indicator(frame: np.ndarray, active: bool, w: int) -> None:
    """Point coloré en haut à droite : vert = mouvement détecté, gris = repos."""
    color = _COLOR_GREEN if active else _COLOR_GRAY
    cv2.circle(frame, (w - 20, 20), 8, color, -1)
    text = "ACTIF" if active else "repos"
    cv2.putText(frame, text, (w - 72, 24), _FONT, _FONT_SMALL, color, 1, cv2.LINE_AA)


def _draw_buffer_bar(frame: np.ndarray, fill: int, max_val: int, w: int) -> None:
    """Barre fine en haut montrant le remplissage du buffer."""
    bar_w = int(w * fill / max_val)
    cv2.rectangle(frame, (0, 0), (bar_w, 5), _COLOR_ACCENT, -1)
    cv2.rectangle(frame, (bar_w, 0), (w, 5), _COLOR_GRAY, -1)


def _draw_current_sign(
    frame: np.ndarray,
    label: str,
    confidence: float,
    uncertainty: float,
    h: int,
    w: int,
) -> None:
    """
    Affiche le label prédit et une barre de certitude basée sur l'entropie.

    La barre représente 1 - uncertainty (la certitude), pas la confiance softmax brute :
    - verte  : uncertainty < 0.4 → le modèle est relativement sûr
    - orange : 0.4 ≤ uncertainty < 0.7 → incertain
    - rouge  : uncertainty ≥ 0.7 → geste probablement hors dataset

    La confiance softmax brute est affichée en petit à côté pour référence.
    """
    certainty = 1.0 - uncertainty   # 1 = sûr, 0 = aléatoire total

    # Couleur du label selon l'incertitude
    if uncertainty < 0.4:
        label_color = _COLOR_GREEN
    elif uncertainty < 0.7:
        label_color = _COLOR_ORANGE
    else:
        label_color = _COLOR_RED

    y_label = h - _PANEL_HEIGHT + 42
    cv2.putText(frame, label.upper(), (20, y_label),
                _FONT, _FONT_LARGE, label_color, _THICKNESS, cv2.LINE_AA)

    # Barre de certitude (entropie inversée)
    bar_x, bar_y = 20, h - _PANEL_HEIGHT + 55
    bar_total_w = 220
    bar_h = 12
    filled_w = int(bar_total_w * certainty)

    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_total_w, bar_y + bar_h), _COLOR_GRAY, -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled_w, bar_y + bar_h), label_color, -1)

    # Texte : certitude % + confiance brute en petit
    cv2.putText(frame, f"certitude {certainty * 100:.0f}%",
                (bar_x + bar_total_w + 8, bar_y + 10),
                _FONT, _FONT_SMALL, _COLOR_WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, f"(softmax {confidence * 100:.0f}%)",
                (bar_x + bar_total_w + 8, bar_y + 24),
                _FONT, 0.4, _COLOR_GRAY, 1, cv2.LINE_AA)


def _draw_dedup_dots(
    frame: np.ndarray,
    count: int,
    max_count: int,
    h: int,
    w: int,
) -> None:
    """
    Affiche max_count points montrant la progression vers la confirmation.

    ○ ○ ○  → 0 prédictions accumulées
    ● ○ ○  → 1/3
    ● ● ○  → 2/3
    ● ● ●  → confirmé (flash bref, puis vidé)
    """
    dot_radius = 7
    spacing = 22
    start_x = 20
    y = h - _PANEL_HEIGHT + 88

    cv2.putText(frame, "confirmation :", (start_x, y - 10),
                _FONT, 0.42, _COLOR_GRAY, 1, cv2.LINE_AA)

    for i in range(max_count):
        cx = start_x + i * spacing + 10
        if i < count:
            cv2.circle(frame, (cx, y + 4), dot_radius, _COLOR_GREEN, -1)
        else:
            cv2.circle(frame, (cx, y + 4), dot_radius, _COLOR_GRAY, 1)


def _draw_phrase(frame: np.ndarray, phrase: list[str], h: int, w: int) -> None:
    """Phrase accumulée en bas du bandeau."""
    if not phrase:
        text = "[ aucun signe confirme ]"
        color = _COLOR_GRAY
    else:
        visible = phrase[-6:]
        text = "  ".join(s.upper() for s in visible)
        color = _COLOR_WHITE

    cv2.putText(frame, text, (20, h - 14),
                _FONT, _FONT_MEDIUM, color, 1, cv2.LINE_AA)


def draw_help(frame: np.ndarray) -> np.ndarray:
    """Superpose les raccourcis clavier (Q = quitter, C = effacer)."""
    out = frame.copy()
    for i, line in enumerate(["Q : quitter", "C : effacer la phrase"]):
        cv2.putText(out, line, (10, 30 + i * 25),
                    _FONT, _FONT_SMALL, _COLOR_WHITE, 1, cv2.LINE_AA)
    return out

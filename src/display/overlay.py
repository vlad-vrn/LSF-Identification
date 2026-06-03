"""
Affichage OpenCV pour la démonstration temps réel LSF.

Panneau en bas divisé en zones :
  ┌────────────────── État ──────────────────────────────────┐
  │  SOIF                      │  Concurrents :              │
  │  Certitude ████████░░ 78%  │  2. BOIRE   ████░░░  18%   │
  │  smax 62%   marge 34%      │  3. AIMER   ██░░░░░   8%   │
  ├── Confirmation ●●○  [mvt✓][conf✓][cert!][marge✓] ───────┤
  │  Signes : JE  VOULOIR  BOIRE                            │
  │  ▶ Je voudrais boire.                                   │
  │  Q:quitter   C:effacer   S:construire                   │
  └─────────────────────────────────────────────────────────┘
"""

import cv2
import numpy as np

# ── Palette ───────────────────────────────────────────────────────────────
_WHITE  = (255, 255, 255)
_BLACK  = (0,   0,   0)
_GRAY   = (120, 120, 120)
_DGRAY  = (50,  50,  50)
_GREEN  = (60,  210, 60)
_ORANGE = (0,   165, 255)
_RED    = (50,  50,  220)
_CYAN   = (220, 200, 0)      # BGR cyan
_YELLOW = (0,   220, 220)    # BGR yellow
_TEAL   = (160, 180, 0)      # BGR teal

# Couleurs par état
_STATE_COLORS = {
    "EN ATTENTE":         ((30,  30,  30),  _GRAY),
    "MOUVEMENT DETECTE":  ((0,   60,  90),  _ORANGE),
    "DETECTION":          ((0,   60, 100),  _CYAN),
    "CONFIRME !":         ((0,  100,  0),   _GREEN),
}

# ── Géométrie du panneau ──────────────────────────────────────────────────
_PANEL_H = 250          # hauteur totale du panneau

# Y-offsets depuis le haut du panneau (h - _PANEL_H)
_Y_STATE  = 0    # ↓ bannière état                   [0 → 38)
_Y_MAIN   = 38   # ↓ prédiction + concurrents        [38 → 150)
_Y_DEDUP  = 150  # ↓ confirmation + blocker pills    [150 → 185)
_Y_SEP    = 185  # ─ séparateur horizontal
_Y_SIGNS  = 188  # ↓ signes accumulés                [188 → 213)
_Y_NLP    = 213  # ↓ phrase construite NLP           [213 → 235)
_Y_HINTS  = 237  # ↓ raccourcis                      [237 → 250)

_FONT = cv2.FONT_HERSHEY_SIMPLEX


# ── Point d'entrée principal ──────────────────────────────────────────────

def draw_overlay(
    frame: np.ndarray,
    current_label: str | None,
    confidence: float,
    uncertainty: float,
    margin: float,
    top3: list[tuple[str, float]],
    phrase: list[str],
    buffer_fill: int,
    dedup_count: int,
    cooldown_frames: int,
    ui_state: str,
    blockers: dict[str, bool] | None,
    constructed_sentence: str | None = None,
    sentence_building: bool = False,
    dedup_max: int = 2,
    buffer_max: int = 64,
) -> np.ndarray:
    """
    Dessine l'interface complète sur la frame.

    Args:
        ui_state:  Chaîne d'état calculée dans run.py
                   ("EN ATTENTE" / "MOUVEMENT DETECTE" / "DETECTION" /
                    "CONFIRMATION N/M" / "CONFIRME !").
        blockers:  Dict {clé: bool} indiquant si chaque condition est remplie.
                   None si pas encore d'inférence.
        top3:      [(label, prob)] des 3 meilleures prédictions.
    """
    out = frame.copy()
    h, w = out.shape[:2]
    y0 = h - _PANEL_H          # y du bord haut du panneau
    div = int(w * 0.52)        # séparateur vertical L/R

    _draw_panel_bg(out, y0, w)
    _draw_buffer_bar(out, buffer_fill, buffer_max, w)
    _draw_state_banner(out, ui_state, y0, w)

    if current_label is not None:
        _draw_prediction_left(out, current_label, confidence, uncertainty, margin, y0, div)
    if len(top3) > 1:
        _draw_alternatives_right(out, top3[1:], y0, div, w)
    if div > 0:
        cv2.line(out, (div, y0 + _Y_MAIN + 5), (div, y0 + _Y_DEDUP - 5), (60, 60, 60), 1)

    _draw_dedup_row(out, dedup_count, dedup_max, cooldown_frames, blockers, y0, w)
    cv2.line(out, (12, y0 + _Y_SEP), (w - 12, y0 + _Y_SEP), (55, 55, 55), 1)
    _draw_signs_phrase(out, phrase, y0, w)
    _draw_nlp_row(out, constructed_sentence, sentence_building, y0, w)
    _draw_hints(out, y0, w)

    return out


# ── Sections ─────────────────────────────────────────────────────────────

def _draw_panel_bg(frame: np.ndarray, y0: int, w: int) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y0), (w, frame.shape[0]), _BLACK, -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)


def _draw_buffer_bar(frame: np.ndarray, fill: int, max_val: int, w: int) -> None:
    """Barre fine en haut de frame indiquant le remplissage du buffer."""
    bar_w = int(w * fill / max_val)
    cv2.rectangle(frame, (0, 0), (bar_w, 4), _CYAN, -1)
    cv2.rectangle(frame, (bar_w, 0), (w, 4), _DGRAY, -1)


def _draw_state_banner(frame: np.ndarray, ui_state: str, y0: int, w: int) -> None:
    """Bannière colorée pleine largeur indiquant l'état courant du pipeline."""
    # Nettoie "CONFIRMATION N/M" pour la palette (clé partielle)
    key = ui_state if ui_state in _STATE_COLORS else (
        "CONFIRME !" if "CONFIRME" in ui_state else
        "DETECTION" if "CONFIRMATION" in ui_state else
        "EN ATTENTE"
    )
    bg_color, txt_color = _STATE_COLORS.get(key, ((30, 30, 30), _GRAY))

    y1 = y0 + _Y_STATE
    y2 = y0 + _Y_MAIN - 1
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y1), (w, y2), bg_color, -1)
    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)

    # Texte centré
    scale, thick = 0.68, 2
    tw, th = cv2.getTextSize(ui_state, _FONT, scale, thick)[0]
    tx = (w - tw) // 2
    ty = y1 + (y2 - y1 + th) // 2
    cv2.putText(frame, ui_state, (tx, ty), _FONT, scale, txt_color, thick, cv2.LINE_AA)


def _draw_prediction_left(
    frame: np.ndarray,
    label: str,
    confidence: float,
    uncertainty: float,
    margin: float,
    y0: int,
    div: int,
) -> None:
    """Panneau gauche : label courant + barre de certitude + smax/marge."""
    certainty = 1.0 - uncertainty
    if uncertainty < 0.4:
        color = _GREEN
    elif uncertainty < 0.7:
        color = _ORANGE
    else:
        color = _RED

    base = y0 + _Y_MAIN

    # Label grand
    cv2.putText(frame, label.upper(), (14, base + 40),
                _FONT, 1.15, color, 2, cv2.LINE_AA)

    # Barre certitude
    bx, by = 14, base + 55
    bw, bh = div - 28, 13
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), _DGRAY, -1)
    cv2.rectangle(frame, (bx, by), (bx + int(bw * certainty), by + bh), color, -1)
    # Tick de seuil de qualité (certitude min ≈ 1-0.75 = 25%)
    tick_x = bx + int(bw * 0.25)
    cv2.line(frame, (tick_x, by - 2), (tick_x, by + bh + 2), (80, 80, 80), 1)

    cv2.putText(frame, f"certitude  {certainty * 100:.0f}%",
                (bx, by + bh + 18), _FONT, 0.5, _WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, f"smax {confidence * 100:.0f}%     marge {margin * 100:.0f}%",
                (bx, by + bh + 34), _FONT, 0.42, _GRAY, 1, cv2.LINE_AA)


def _draw_alternatives_right(
    frame: np.ndarray,
    alternatives: list[tuple[str, float]],
    y0: int,
    div: int,
    w: int,
) -> None:
    """Panneau droit : top-2 et top-3 concurrents avec mini-barres."""
    base = y0 + _Y_MAIN
    x = div + 14
    bar_max = w - x - 52   # largeur max des barres

    cv2.putText(frame, "concurrents :", (x, base + 18),
                _FONT, 0.42, _GRAY, 1, cv2.LINE_AA)

    for i, (lbl, prob) in enumerate(alternatives[:2]):
        y = base + 42 + i * 42
        lbl_short = lbl.upper()[:11]

        cv2.putText(frame, f"{i + 2}.  {lbl_short}", (x, y),
                    _FONT, 0.5, _GRAY, 1, cv2.LINE_AA)

        bx, by = x, y + 6
        bh = 9
        cv2.rectangle(frame, (bx, by), (bx + bar_max, by + bh), (40, 40, 40), -1)
        cv2.rectangle(frame, (bx, by), (bx + int(bar_max * prob), by + bh), (80, 80, 130), -1)
        cv2.putText(frame, f"{prob * 100:.0f}%", (bx + bar_max + 4, by + bh),
                    _FONT, 0.4, _GRAY, 1, cv2.LINE_AA)


def _draw_dedup_row(
    frame: np.ndarray,
    count: int,
    max_count: int,
    cooldown_frames: int,
    blockers: dict[str, bool] | None,
    y0: int,
    w: int,
) -> None:
    """Ligne de confirmation : dots ●●○ ou flash CONFIRME + pills de condition."""
    base = y0 + _Y_DEDUP

    # ── Côté gauche : dots ou flash ──
    if cooldown_frames > 0:
        cv2.putText(frame, ">> CONFIRME !",
                    (14, base + 24), _FONT, 0.72, _GREEN, 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "confirmation :", (14, base + 16),
                    _FONT, 0.42, _GRAY, 1, cv2.LINE_AA)
        for i in range(max_count):
            cx = 14 + i * 24 + 12
            if i < count:
                cv2.circle(frame, (cx, base + 28), 8, _GREEN, -1)
            else:
                cv2.circle(frame, (cx, base + 28), 8, _DGRAY, 1)

    # ── Côté droit : blocker pills ──
    if blockers is not None:
        labels = {"mvt": "MVT", "conf": "CONF", "cert": "CERT", "marge": "MARGE"}
        px = w - 14
        for key in reversed(list(labels.keys())):
            ok = blockers.get(key, False)
            txt = labels[key]
            scale = 0.38
            (tw, th), _ = cv2.getTextSize(txt, _FONT, scale, 1)
            pad = 4
            rx2 = px
            rx1 = rx2 - tw - pad * 2
            ry1 = base + 16
            ry2 = ry1 + th + pad * 2
            bg = (0, 90, 0) if ok else (0, 0, 120)
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), bg, -1)
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), _GREEN if ok else _RED, 1)
            cv2.putText(frame, txt, (rx1 + pad, ry2 - pad),
                        _FONT, scale, _GREEN if ok else _RED, 1, cv2.LINE_AA)
            px = rx1 - 6


def _draw_signs_phrase(frame: np.ndarray, phrase: list[str], y0: int, w: int) -> None:
    base = y0 + _Y_SIGNS
    if not phrase:
        cv2.putText(frame, "[ aucun signe confirme ]", (14, base + 18),
                    _FONT, 0.52, _GRAY, 1, cv2.LINE_AA)
    else:
        text = "  ".join(s.upper() for s in phrase[-8:])
        cv2.putText(frame, text, (14, base + 18), _FONT, 0.58, _WHITE, 1, cv2.LINE_AA)


def _draw_nlp_row(
    frame: np.ndarray,
    sentence: str | None,
    building: bool,
    y0: int,
    w: int,
) -> None:
    base = y0 + _Y_NLP
    if building:
        cv2.putText(frame, "Construction en cours...", (14, base + 17),
                    _FONT, 0.52, _ORANGE, 1, cv2.LINE_AA)
    elif sentence:
        display = sentence if len(sentence) <= 80 else sentence[:77] + "..."
        cv2.putText(frame, f"> {display}", (14, base + 17),
                    _FONT, 0.54, _YELLOW, 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "S : construire la phrase", (14, base + 17),
                    _FONT, 0.44, _GRAY, 1, cv2.LINE_AA)


def _draw_hints(frame: np.ndarray, y0: int, w: int) -> None:
    cv2.putText(frame, "Q : quitter      C : effacer signes      S : construire phrase NLP",
                (14, y0 + _Y_HINTS + 12), _FONT, 0.37, (75, 75, 75), 1, cv2.LINE_AA)

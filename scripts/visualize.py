"""
Visualiseur interactif de signes LSF.

Affiche une liste de signes et anime le squelette MediaPipe des samples
enregistrés pour aider à comprendre quel geste effectuer.

Usage :
    python scripts/visualize.py
    python scripts/visualize.py --dataset dataset/

Contrôles :
    ↑ / ↓      : naviguer dans la liste
    ENTRÉE     : sélectionner le signe surligné
    ← / →      : sample précédent / suivant
    ESPACE     : pause / lecture
    Q          : quitter
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data.loader import load_dataset

# ── Dimensions de la fenêtre ──────────────────────────────────────────────
WIN_W, WIN_H = 1100, 660
LIST_W = 235          # panneau gauche : liste des signes
SKEL_W = WIN_W - LIST_W

PLAYBACK_FPS  = 12    # images par seconde pour l'animation
FRAME_DELAY   = max(1, int(1000 / PLAYBACK_FPS))

FONT = cv2.FONT_HERSHEY_SIMPLEX

# ── Palette (BGR) ──────────────────────────────────────────────────────────
C_BG_LIST  = (20, 20, 20)
C_BG_SKEL  = (10, 10, 22)
C_WHITE    = (255, 255, 255)
C_GRAY     = (105, 105, 105)
C_DGRAY    = (42,  42,  42)
C_SELECT   = (0,  185, 255)   # orange vif — signe sélectionné
C_CURSOR   = (50,  50,  50)   # fond ligne surligné
C_ACCENT   = (0,  200, 255)   # barre de progression
C_GREEN    = (75, 210, 75)

# Couleurs squelette
C_POSE_LINE  = (60,  60,  68)
C_POSE_DOT   = (110, 110, 120)
C_HAND_L_LINE = (210, 120, 50)   # main "left" du parquet — orangé
C_HAND_L_DOT  = (240, 160, 80)
C_HAND_R_LINE = (50,  150, 235)  # main "right" du parquet — bleu
C_HAND_R_DOT  = (90,  195, 255)

# ── Connexions du squelette (haut du corps uniquement) ─────────────────────
# Indices MediaPipe Pose (33 landmarks, pertinents pour les signes)
POSE_CONN = [
    (11, 12),                                    # épaules
    (11, 13), (13, 15),                          # bras gauche
    (12, 14), (14, 16),                          # bras droit
    (15, 17), (15, 19), (15, 21), (17, 19),     # main gauche (attaches)
    (16, 18), (16, 20), (16, 22), (18, 20),     # main droite (attaches)
    (11, 23), (12, 24), (23, 24),               # torse supérieur
    (0, 1), (1, 2), (2, 3), (3, 7),             # visage droit
    (0, 4), (4, 5), (5, 6), (6, 8),             # visage gauche
]

HAND_CONN = [
    (0, 1),  (1, 2),  (2, 3),  (3, 4),          # pouce
    (0, 5),  (5, 6),  (6, 7),  (7, 8),          # index
    (0, 9),  (9, 10), (10, 11),(11, 12),         # majeur
    (0, 13),(13, 14),(14, 15),(15, 16),           # annulaire
    (0, 17),(17, 18),(18, 19),(19, 20),           # auriculaire
    (5, 9),  (9, 13),(13, 17),                   # paume
]

# Codes touches (cv2.waitKeyEx, Windows + Linux)
KEY_UP    = [2490368, 65362, 82]
KEY_DOWN  = [2621440, 65364, 84]
KEY_LEFT  = [2424832, 65361, 81]
KEY_RIGHT = [2555904, 65363, 83]
KEY_ENTER = [13, 10]
KEY_SPACE = [32]


# ── Utilitaires squelette ──────────────────────────────────────────────────

def _lm_to_px(
    x: float, y: float,
    x_min: float, y_min: float,
    x_rng: float, y_rng: float,
    w: int, h: int,
    pad: int = 20,
) -> tuple[int, int]:
    """Normalise un landmark dans les bounds et retourne ses coordonnées pixel."""
    nx = (x - x_min) / x_rng if x_rng > 0 else 0.5
    ny = (y - y_min) / y_rng if y_rng > 0 else 0.5
    px = int(pad + nx * (w - 2 * pad))
    py = int(pad + ny * (h - 2 * pad))
    return (int(np.clip(px, 0, w - 1)), int(np.clip(py, 0, h - 1)))


def _draw_hand(
    canvas: np.ndarray,
    lm: np.ndarray,           # (21, 3)
    bounds: tuple,
    c_w: int, c_h: int,
    line_col: tuple,
    dot_col: tuple,
) -> None:
    x_min, y_min, x_rng, y_rng = bounds

    def px(i: int) -> tuple[int, int]:
        return _lm_to_px(lm[i, 0], lm[i, 1], x_min, y_min, x_rng, y_rng, c_w, c_h)

    for i, j in HAND_CONN:
        if not (np.all(lm[i] == 0) or np.all(lm[j] == 0)):
            cv2.line(canvas, px(i), px(j), line_col, 3, cv2.LINE_AA)

    for i in range(21):
        if not np.all(lm[i] == 0):
            r = 8 if i == 0 else 5   # poignet plus gros
            cv2.circle(canvas, px(i), r, dot_col, -1, cv2.LINE_AA)


def _draw_skeleton(
    canvas: np.ndarray,
    frame_data: np.ndarray,   # (318,)
    bounds: tuple,            # (x_min, y_min, x_rng, y_rng)
) -> None:
    """Dessine le squelette sur le canvas en place."""
    c_h, c_w = canvas.shape[:2]
    x_min, y_min, x_rng, y_rng = bounds

    pose  = frame_data[:132].reshape(33, 4)    # x, y, z, vis
    left  = frame_data[132:195].reshape(21, 3)
    right = frame_data[195:258].reshape(21, 3)

    def px_pose(i: int) -> tuple[int, int]:
        return _lm_to_px(pose[i, 0], pose[i, 1], x_min, y_min, x_rng, y_rng, c_w, c_h)

    # Connexions de pose
    for i, j in POSE_CONN:
        if pose[i, 3] > 0.2 and pose[j, 3] > 0.2:
            cv2.line(canvas, px_pose(i), px_pose(j), C_POSE_LINE, 2, cv2.LINE_AA)

    # Points de pose
    for i in range(33):
        if pose[i, 3] > 0.2:
            cv2.circle(canvas, px_pose(i), 4, C_POSE_DOT, -1, cv2.LINE_AA)

    # Mains
    if not np.all(left == 0):
        _draw_hand(canvas, left, bounds, c_w, c_h, C_HAND_L_LINE, C_HAND_L_DOT)
    if not np.all(right == 0):
        _draw_hand(canvas, right, bounds, c_w, c_h, C_HAND_R_LINE, C_HAND_R_DOT)


def _compute_bounds(samples: list[np.ndarray]) -> tuple:
    """
    Calcule les bounds (x_min, y_min, x_rng, y_rng) stables pour tous les samples
    d'un signe, permettant un zoom centré sur le signeur sans saut entre frames.
    """
    xs, ys = [], []
    for frames in samples:
        for frame in frames:
            pose  = frame[:132].reshape(33, 4)
            left  = frame[132:195].reshape(21, 3)
            right = frame[195:258].reshape(21, 3)
            for i in range(33):
                if pose[i, 3] > 0.25:
                    xs.append(pose[i, 0])
                    ys.append(pose[i, 1])
            for hand in [left, right]:
                if not np.all(hand == 0):
                    for lm in hand:
                        if lm[0] > 0.01 or lm[1] > 0.01:
                            xs.append(lm[0])
                            ys.append(lm[1])

    if not xs:
        return 0.0, 0.0, 1.0, 1.0

    pad = 0.08
    x_min = max(0.0, float(np.percentile(xs, 2))  - pad)
    x_max = min(1.0, float(np.percentile(xs, 98)) + pad)
    y_min = max(0.0, float(np.percentile(ys, 2))  - pad)
    y_max = min(1.0, float(np.percentile(ys, 98)) + pad)

    x_rng = max(x_max - x_min, 0.05)
    y_rng = max(y_max - y_min, 0.05)

    # Force un ratio carré pour que le squelette ne soit pas déformé
    if x_rng > y_rng:
        diff = x_rng - y_rng
        y_min -= diff / 2
        y_rng = x_rng
    else:
        diff = y_rng - x_rng
        x_min -= diff / 2
        x_rng = y_rng

    return x_min, y_min, x_rng, y_rng


# ── Rendu ──────────────────────────────────────────────────────────────────

def _render_list(
    canvas: np.ndarray,
    signs: list[str],
    sign_samples: dict[str, list],
    state: dict,
) -> None:
    """Panneau gauche : liste des signes avec navigation."""
    cv2.rectangle(canvas, (0, 0), (LIST_W, WIN_H), C_BG_LIST, -1)
    cv2.line(canvas, (LIST_W - 1, 0), (LIST_W - 1, WIN_H), C_DGRAY, 1)

    # En-tête
    cv2.putText(canvas, "SIGNES", (12, 32), FONT, 0.72, C_ACCENT, 2, cv2.LINE_AA)
    cv2.line(canvas, (0, 42), (LIST_W, 42), C_DGRAY, 1)

    # Liste scrollable centrée sur le curseur
    row_h = 30
    max_visible = (WIN_H - 100) // row_h
    cursor = state["list_cursor"]
    start = max(0, cursor - max_visible // 2)
    start = min(start, max(0, len(signs) - max_visible))

    for i, idx in enumerate(range(start, min(start + max_visible, len(signs)))):
        sign = signs[idx]
        y = 54 + i * row_h
        is_cursor   = idx == cursor
        is_selected = sign == state["sign"]

        # Fond surligné
        if is_cursor:
            cv2.rectangle(canvas, (3, y - 1), (LIST_W - 4, y + row_h - 4), C_CURSOR, -1)

        # Indicateur de sélection active
        if is_selected:
            cv2.circle(canvas, (10, y + 11), 4, C_SELECT, -1)

        color = C_SELECT if is_cursor else (C_WHITE if is_selected else C_GRAY)
        cv2.putText(canvas, sign.replace("_", " ")[:20], (22, y + 16),
                    FONT, 0.46, color, 1, cv2.LINE_AA)

        # Nombre de samples (droite)
        n = len(sign_samples.get(sign, []))
        cv2.putText(canvas, str(n), (LIST_W - 28, y + 16),
                    FONT, 0.4, C_GRAY if not is_cursor else (150, 150, 150),
                    1, cv2.LINE_AA)

    # Barre de défilement
    if len(signs) > max_visible:
        sb_h = WIN_H - 100
        thumb_h = max(20, int(sb_h * max_visible / len(signs)))
        thumb_y = 54 + int((sb_h - thumb_h) * cursor / max(1, len(signs) - 1))
        cv2.rectangle(canvas, (LIST_W - 5, 54), (LIST_W - 2, 54 + sb_h), C_DGRAY, -1)
        cv2.rectangle(canvas, (LIST_W - 5, thumb_y), (LIST_W - 2, thumb_y + thumb_h), C_GRAY, -1)

    # Footer : raccourcis
    cv2.line(canvas, (0, WIN_H - 72), (LIST_W, WIN_H - 72), C_DGRAY, 1)
    for i, hint in enumerate(["haut/bas : naviguer", "ENTREE  : choisir", "Q  : quitter"]):
        cv2.putText(canvas, hint, (8, WIN_H - 55 + i * 19),
                    FONT, 0.38, C_GRAY, 1, cv2.LINE_AA)


def _render_skeleton_panel(
    canvas: np.ndarray,
    sign_samples: dict[str, list],
    state: dict,
    bounds_cache: dict,
) -> None:
    """Panneau droit : animation du squelette + barre d'info."""
    x0 = LIST_W
    INFO_H = 95   # hauteur de la barre d'info en bas

    # Fond
    cv2.rectangle(canvas, (x0, 0), (WIN_W, WIN_H), C_BG_SKEL, -1)

    sign    = state["sign"]
    samples = sign_samples.get(sign, [])
    if not samples:
        cv2.putText(canvas, "Aucun sample disponible",
                    (x0 + 30, WIN_H // 2), FONT, 0.7, C_GRAY, 1, cv2.LINE_AA)
        return

    s_idx  = min(state["sample_idx"], len(samples) - 1)
    frames = samples[s_idx]
    f_idx  = state["frame_idx"] % len(frames)

    # Zone squelette
    skel_h = WIN_H - INFO_H
    skel_canvas = np.full((skel_h, SKEL_W, 3), C_BG_SKEL, dtype=np.uint8)

    bounds = bounds_cache.get(sign)
    if bounds is None:
        bounds = _compute_bounds(samples)
        bounds_cache[sign] = bounds

    _draw_skeleton(skel_canvas, frames[f_idx], bounds)

    # Légende mains
    cv2.circle(skel_canvas, (SKEL_W - 120, skel_h - 30), 6, C_HAND_L_DOT, -1)
    cv2.putText(skel_canvas, "main gauche (image)", (SKEL_W - 108, skel_h - 25),
                FONT, 0.36, C_HAND_L_DOT, 1, cv2.LINE_AA)
    cv2.circle(skel_canvas, (SKEL_W - 120, skel_h - 12), 6, C_HAND_R_DOT, -1)
    cv2.putText(skel_canvas, "main droite (image)", (SKEL_W - 108, skel_h - 7),
                FONT, 0.36, C_HAND_R_DOT, 1, cv2.LINE_AA)

    canvas[0:skel_h, x0:WIN_W] = skel_canvas

    # ── Barre d'info en bas ──────────────────────────────────────────────
    info_y = WIN_H - INFO_H
    cv2.rectangle(canvas, (x0, info_y), (WIN_W, WIN_H), (14, 14, 26), -1)
    cv2.line(canvas, (x0, info_y), (WIN_W, info_y), C_DGRAY, 1)

    # Nom du signe (grand)
    display_name = sign.replace("_", "'").upper()
    cv2.putText(canvas, display_name, (x0 + 14, info_y + 32),
                FONT, 1.0, C_SELECT, 2, cv2.LINE_AA)

    # Sample info
    cv2.putText(canvas,
                f"sample {s_idx + 1} / {len(samples)}   |   frame {f_idx + 1} / {len(frames)}",
                (x0 + 14, info_y + 52), FONT, 0.46, C_GRAY, 1, cv2.LINE_AA)

    # Barre de progression de la frame
    bx = x0 + 14
    bw = SKEL_W - 28
    by = info_y + 62
    cv2.rectangle(canvas, (bx, by), (bx + bw, by + 7), C_DGRAY, -1)
    fill = int(bw * (f_idx + 1) / max(len(frames), 1))
    cv2.rectangle(canvas, (bx, by), (bx + fill, by + 7), C_ACCENT, -1)

    # État lecture / pause
    play_lbl = "|>  LECTURE" if state["playing"] else "||  PAUSE"
    play_col = C_GREEN if state["playing"] else C_GRAY
    cv2.putText(canvas, play_lbl, (WIN_W - 115, info_y + 52),
                FONT, 0.44, play_col, 1, cv2.LINE_AA)

    # Raccourcis
    cv2.putText(canvas, "← → : changer sample     ESPACE : pause / lecture",
                (x0 + 14, WIN_H - 8), FONT, 0.38, (58, 58, 75), 1, cv2.LINE_AA)


# ── Boucle principale ──────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Visualiseur de signes LSF")
    parser.add_argument("--dataset", default="dataset",
                        help="Chemin vers le dossier dataset/ (défaut : dataset/)")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    if not (dataset_dir / "approved").exists():
        print(f"Dossier dataset/approved/ introuvable : {dataset_dir}")
        sys.exit(1)

    print("Chargement du dataset...", flush=True)
    all_samples = load_dataset(dataset_dir)

    # Regroupe les frames par signe
    sign_samples: dict[str, list[np.ndarray]] = defaultdict(list)
    for s in all_samples:
        sign_samples[s["label"]].append(s["frames"])   # (64, 318)

    signs = sorted(sign_samples.keys())
    print(f"{len(signs)} signes, {len(all_samples)} samples — fenêtre prête.")

    state: dict = {
        "list_cursor": 0,
        "sign":        signs[0],
        "sample_idx":  0,
        "frame_idx":   0,
        "playing":     True,
    }
    bounds_cache: dict[str, tuple] = {}

    cv2.namedWindow("LSF — Visualiseur", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("LSF — Visualiseur", WIN_W, WIN_H)

    while True:
        # ── Rendu ────────────────────────────────────────────────────────
        canvas = np.zeros((WIN_H, WIN_W, 3), dtype=np.uint8)
        _render_list(canvas, signs, sign_samples, state)
        _render_skeleton_panel(canvas, sign_samples, state, bounds_cache)
        cv2.imshow("LSF — Visualiseur", canvas)

        # ── Avance l'animation ────────────────────────────────────────────
        if state["playing"]:
            samples = sign_samples[state["sign"]]
            if samples:
                n_f = len(samples[min(state["sample_idx"], len(samples) - 1)])
                state["frame_idx"] = (state["frame_idx"] + 1) % n_f

        # ── Gestion clavier ───────────────────────────────────────────────
        key = cv2.waitKeyEx(FRAME_DELAY)
        if key == -1:
            continue

        lower = key & 0xFF

        if lower == ord("q"):
            break

        elif key in KEY_UP or lower == ord("k"):
            state["list_cursor"] = (state["list_cursor"] - 1) % len(signs)

        elif key in KEY_DOWN or lower == ord("j"):
            state["list_cursor"] = (state["list_cursor"] + 1) % len(signs)

        elif lower in KEY_ENTER:
            new_sign = signs[state["list_cursor"]]
            if new_sign != state["sign"]:
                state["sign"]       = new_sign
                state["sample_idx"] = 0
                state["frame_idx"]  = 0

        elif key in KEY_LEFT or lower == ord("h"):
            samples = sign_samples[state["sign"]]
            state["sample_idx"] = (state["sample_idx"] - 1) % max(1, len(samples))
            state["frame_idx"]  = 0

        elif key in KEY_RIGHT or lower == ord("l"):
            samples = sign_samples[state["sign"]]
            state["sample_idx"] = (state["sample_idx"] + 1) % max(1, len(samples))
            state["frame_idx"]  = 0

        elif lower in KEY_SPACE:
            state["playing"] = not state["playing"]

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

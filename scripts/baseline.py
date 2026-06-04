"""
Baseline chiffrée du modèle LSF actuel — Étape 0 du prompt d'amélioration.

Entraîne l'état COURANT du modèle sur les 37 signes (split par contributeur,
seed fixe pour la reproductibilité), puis logge :
  - accuracy globale (val)
  - accuracy par classe (val)
  - matrice de confusion (val)
  - les paires de signes les plus confondues

Ne modifie NI le modèle NI le preprocessing : il réutilise les fonctions
existantes de model.train (mêmes dataloaders, même split seed=42, mêmes
hyperparamètres). Seule différence avec scripts/train.py : on fixe les seeds
globales et on calcule la matrice de confusion sur le meilleur checkpoint.

Usage :
    .\.venv\Scripts\python.exe scripts/baseline.py --dataset src/data
    .\.venv\Scripts\python.exe scripts/baseline.py --epochs 3   # smoke test
"""

import argparse
import copy
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.loader import get_label_map, load_dataset
from model.classifier import make_classifier
from model.train import (
    LABEL_SMOOTHING,
    LR,
    N_EPOCHS,
    WARMUP_EPOCHS,
    WEIGHT_DECAY,
    compute_class_weights,
    eval_epoch,
    make_dataloaders,
    train_epoch,
)

logger = logging.getLogger("baseline")

# Seed unique pour que cette baseline soit reproductible et comparable aux
# itérations suivantes (poids initiaux, augmentation, ordre des batchs).
SEED: int = 42

# Nombre de paires confondues à afficher dans le résumé.
TOP_CONFUSED_PAIRS: int = 12


def set_global_seed(seed: int) -> None:
    """Fixe toutes les sources d'aléa (python, numpy, torch)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def collect_predictions(
    model: nn.Module,
    loader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Retourne (y_true, y_pred) concaténés sur tout le loader."""
    model.eval()
    trues: list[int] = []
    preds: list[int] = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            logits = model(X_batch.to(device))
            batch_preds = logits.argmax(dim=1).cpu().numpy()
            trues.extend(y_batch.numpy().tolist())
            preds.extend(batch_preds.tolist())
    return np.array(trues), np.array(preds)


def most_confused_pairs(
    cm: np.ndarray,
    idx_to_slug: dict[int, str],
    top_k: int,
) -> list[tuple[str, str, int, float]]:
    """
    Extrait les paires (vrai → prédit) les plus confondues hors diagonale.

    Returns:
        Liste de (slug_vrai, slug_predit, n_erreurs, part_de_la_classe_vraie).
    """
    pairs: list[tuple[str, str, int, float]] = []
    n_classes = cm.shape[0]
    row_totals = cm.sum(axis=1)
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j and cm[i, j] > 0:
                share = cm[i, j] / row_totals[i] if row_totals[i] > 0 else 0.0
                pairs.append((idx_to_slug[i], idx_to_slug[j], int(cm[i, j]), share))
    pairs.sort(key=lambda t: (t[2], t[3]), reverse=True)
    return pairs[:top_k]


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline chiffrée LSF (Étape 0)")
    parser.add_argument("--dataset", default="src/data",
                        help="Dossier contenant approved/ (défaut : src/data)")
    parser.add_argument("--epochs", type=int, default=N_EPOCHS,
                        help=f"Nombre d'epochs (défaut : {N_EPOCHS})")
    parser.add_argument("--save", default="models/lsf_v1.pt",
                        help="Chemin du meilleur checkpoint (défaut : models/lsf_v1.pt)")
    parser.add_argument("--log", default="IMPROVE_LOG.md",
                        help="Fichier journal où écrire le résumé baseline")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    set_global_seed(SEED)
    device = torch.device("cpu")

    dataset_dir = Path(args.dataset)
    logger.info("Chargement du dataset depuis %s", dataset_dir)
    samples = load_dataset(dataset_dir)
    label_map = get_label_map(dataset_dir)
    n_classes = len(label_map)
    idx_to_slug = {v: k for k, v in label_map.items()}
    logger.info("%d samples, %d classes", len(samples), n_classes)

    # --- Dataloaders (split par contributeur, identique à scripts/train.py) ---
    train_loader, val_loader = make_dataloaders(samples, label_map)

    # --- Modèle + optim + scheduler (mêmes réglages que model.train.train) ---
    model = make_classifier(n_classes=n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPOCHS)
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - WARMUP_EPOCHS), eta_min=1e-5)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[WARMUP_EPOCHS])

    train_labels = [int(y) for _, ys in train_loader for y in ys]
    class_weights = compute_class_weights(train_labels, n_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)

    logger.info("Modèle : %d paramètres", sum(p.numel() for p in model.parameters()))
    logger.info("Entraînement baseline : %d epochs, seed=%d", args.epochs, SEED)

    best_val_acc = 0.0
    best_state: dict | None = None

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()
        logger.info(
            "Epoch %3d/%d — train_loss=%.4f | val_loss=%.4f | val_acc=%.2f%% (lr=%.2e)",
            epoch, args.epochs, train_loss, val_loss, val_acc * 100,
            optimizer.param_groups[0]["lr"],
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    # --- Recharge le meilleur modèle pour les diagnostics ---
    if best_state is not None:
        model.load_state_dict(best_state)

    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state_dict": model.state_dict(), "n_classes": n_classes,
         "label_map": label_map, "val_acc": best_val_acc},
        save_path,
    )
    with (save_path.parent / "label_map.json").open("w", encoding="utf-8") as f:
        import json
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    logger.info("Meilleur modèle sauvegardé : %s (val_acc=%.2f%%)", save_path, best_val_acc * 100)

    # --- Matrice de confusion (val) ---
    from sklearn.metrics import confusion_matrix

    y_true, y_pred = collect_predictions(model, val_loader, device)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))

    # Accuracy par classe (rappel)
    per_class: list[tuple[str, float, int, int]] = []
    for idx in range(n_classes):
        total = int(cm[idx].sum())
        correct = int(cm[idx, idx])
        acc = correct / total if total > 0 else float("nan")
        per_class.append((idx_to_slug[idx], acc, correct, total))

    pairs = most_confused_pairs(cm, idx_to_slug, TOP_CONFUSED_PAIRS)

    # --- Log console ---
    logger.info("════════ BASELINE — résultats (val, split par contributeur) ════════")
    logger.info("Accuracy globale (val) : %.2f%%", best_val_acc * 100)
    logger.info("─── Accuracy par classe (triée croissante) ───")
    for slug, acc, correct, total in sorted(per_class, key=lambda t: (np.nan_to_num(t[1], nan=-1))):
        acc_str = f"{acc * 100:5.1f}%" if total > 0 else "  N/A "
        logger.info("  %-18s %s  (%d/%d)", slug, acc_str, correct, total)
    logger.info("─── Paires les plus confondues (vrai → prédit) ───")
    for true_s, pred_s, n, share in pairs:
        logger.info("  %-18s → %-18s  %d fois (%.0f%% de la classe)", true_s, pred_s, n, share * 100)

    # --- Écriture du journal IMPROVE_LOG.md ---
    write_log(args.log, best_val_acc, per_class, pairs, args.epochs, len(y_true))
    logger.info("Résumé écrit dans %s", args.log)


def write_log(
    log_path: str,
    val_acc: float,
    per_class: list[tuple[str, float, int, int]],
    pairs: list[tuple[str, str, int, float]],
    epochs: int,
    n_val: int,
) -> None:
    """Écrit (ou crée) le journal d'amélioration avec la baseline en première entrée."""
    lines: list[str] = []
    lines.append("# Journal d'amélioration du modèle LSF\n")
    lines.append("## Itération 0 — BASELINE (état courant, non modifié)\n")
    lines.append(f"- Config : 37 signes, split par contributeur (seed=42), {epochs} epochs, {n_val} samples de validation.")
    lines.append(f"- **Accuracy globale (val) : {val_acc * 100:.2f}%**\n")
    lines.append("### Accuracy par classe (triée croissante)\n")
    lines.append("| Signe | Rappel | Correct/Total |")
    lines.append("|-------|--------|---------------|")
    for slug, acc, correct, total in sorted(per_class, key=lambda t: (np.nan_to_num(t[1], nan=-1))):
        acc_str = f"{acc * 100:.1f}%" if total > 0 else "N/A (0 en val)"
        lines.append(f"| {slug} | {acc_str} | {correct}/{total} |")
    lines.append("\n### Paires les plus confondues (vrai → prédit)\n")
    lines.append("| Vrai | Prédit | Erreurs | % de la classe |")
    lines.append("|------|--------|---------|----------------|")
    for true_s, pred_s, n, share in pairs:
        lines.append(f"| {true_s} | {pred_s} | {n} | {share * 100:.0f}% |")
    lines.append("")
    Path(log_path).write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

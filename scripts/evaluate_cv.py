"""
Harness d'évaluation par validation croisée GROUPÉE PAR CONTRIBUTEUR.

Pourquoi : un split unique par contributeur (scripts/train.py) laisse ~16 classes
entièrement côté train → non mesurées. Avec un GroupKFold à k folds, chaque
sample passe exactement une fois en validation : en poolant les prédictions des
k folds, on couvre les 37 classes. C'est le protocole d'évaluation fiable
réutilisé à chaque itération d'amélioration.

Garde model/ et data/ INTACTS : on réimporte la même augmentation, le même
preprocessing et les mêmes hyperparamètres que model.train, on ne fait que
changer le schéma de découpe train/val et l'agrégation des métriques.

Deux chiffres rapportés :
  - accuracy globale poolée (37 classes)
  - accuracy sur le SOUS-ENSEMBLE ÉVALUABLE (classes ayant >= MIN_CONTRIB
    contributeurs — les seules où la généralisation à un nouveau signeur a un
    sens statistique).

Usage :
    .\.venv\Scripts\python.exe scripts/evaluate_cv.py --dataset src/data --tag "0bis baseline GroupKFold"
    .\.venv\Scripts\python.exe scripts/evaluate_cv.py --folds 5 --epochs 60
"""

import argparse
import copy
import logging
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.loader import get_label_map, load_dataset
from data.preprocessing import augment_sample, build_feature_vector
from model.classifier import _DROPOUT, make_classifier
from model.train import (
    BATCH_SIZE,
    LABEL_SMOOTHING,
    LR,
    WARMUP_EPOCHS,
    WEIGHT_DECAY,
    _compute_n_augments,
    _LSFDataset,
    compute_class_weights,
    eval_epoch,
    train_epoch,
)

logger = logging.getLogger("evaluate_cv")

SEED: int = 42
DEFAULT_FOLDS: int = 5
DEFAULT_EPOCHS: int = 60          # la val plafonne dès ~30-40 epochs (cf. baseline)
MIN_CONTRIB: int = 3              # seuil "classe évaluable" en généralisation signeur
TOP_CONFUSED_PAIRS: int = 15


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_fold_features(
    train_samples: list[dict],
    val_samples: list[dict],
) -> tuple[list[np.ndarray], list[int], list[np.ndarray], list[int]]:
    """
    Construit les features train (augmentées) et val (non augmentées) d'un fold.

    Réplique exactement la logique de model.train.make_dataloaders :
    augmentation pondérée par classe avec cible = classe la plus représentée.
    """
    counts: Counter = Counter(s["label_idx"] for s in train_samples)
    target = max(counts.values()) if counts else 1

    train_X: list[np.ndarray] = []
    train_y: list[int] = []
    for s in train_samples:
        n_aug = _compute_n_augments(counts[s["label_idx"]], target)
        for v in augment_sample(s["frames"], n_augments=n_aug):
            train_X.append(build_feature_vector(v))
            train_y.append(s["label_idx"])

    val_X = [build_feature_vector(s["frames"]) for s in val_samples]
    val_y = [s["label_idx"] for s in val_samples]
    return train_X, train_y, val_X, val_y


def train_one_fold(
    train_X: list[np.ndarray],
    train_y: list[int],
    val_X: list[np.ndarray],
    val_y: list[int],
    n_classes: int,
    n_epochs: int,
    device: torch.device,
    hp: dict,
) -> tuple[nn.Module, float]:
    """Entraîne un modèle neuf sur un fold, retourne (modèle au meilleur val_acc, best_val_acc).

    hp : dict d'hyperparamètres {lr, weight_decay, label_smoothing, dropout}.
    """
    train_loader = DataLoader(_LSFDataset(train_X, train_y), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(_LSFDataset(val_X, val_y), batch_size=BATCH_SIZE, shuffle=False)

    model = make_classifier(n_classes=n_classes, dropout=hp["dropout"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPOCHS)
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, n_epochs - WARMUP_EPOCHS), eta_min=1e-5)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[WARMUP_EPOCHS])

    class_weights = compute_class_weights(train_y, n_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=hp["label_smoothing"])

    best_acc = 0.0
    best_state = copy.deepcopy(model.state_dict())
    for _ in range(1, n_epochs + 1):
        train_epoch(model, train_loader, optimizer, criterion, device)
        _, val_acc = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model, best_acc


def predict(model: nn.Module, X: list[np.ndarray], device: torch.device) -> np.ndarray:
    model.eval()
    preds: list[int] = []
    loader = DataLoader(
        _LSFDataset(X, [0] * len(X)), batch_size=BATCH_SIZE, shuffle=False
    )
    with torch.no_grad():
        for X_batch, _ in loader:
            preds.extend(model(X_batch.to(device)).argmax(dim=1).cpu().numpy().tolist())
    return np.array(preds)


def contributor_counts(samples: list[dict], label_map: dict[str, int]) -> dict[int, int]:
    """Nombre de contributeurs distincts par index de classe."""
    by_class: dict[int, set] = defaultdict(set)
    for s in samples:
        by_class[s["label_idx"]].add(s["contributor_id"])
    return {idx: len(by_class.get(idx, set())) for idx in label_map.values()}


def filter_by_min_contrib(
    samples: list[dict],
    label_map: dict[str, int],
    min_contrib: int,
) -> tuple[list[dict], dict[str, int]]:
    """
    Ne conserve que les classes ayant >= min_contrib contributeurs distincts,
    et reconstruit un label_map contigu (0..N-1) trié alphabétiquement.

    Les classes à très peu de contributeurs sont ineval​uables en généralisation
    à un nouveau signeur : les retirer donne une métrique honnête sur le
    périmètre réellement apprenable (décision V1).
    """
    counts = contributor_counts(samples, label_map)
    keep_slugs = sorted(slug for slug, idx in label_map.items() if counts[idx] >= min_contrib)
    new_map = {slug: i for i, slug in enumerate(keep_slugs)}
    keep_set = set(keep_slugs)

    filtered: list[dict] = []
    for s in samples:
        if s["label"] in keep_set:
            s2 = dict(s)
            s2["label_idx"] = new_map[s["label"]]
            filtered.append(s2)
    return filtered, new_map


def most_confused_pairs(cm: np.ndarray, idx_to_slug, top_k):
    pairs = []
    row_totals = cm.sum(axis=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[0]):
            if i != j and cm[i, j] > 0:
                share = cm[i, j] / row_totals[i] if row_totals[i] > 0 else 0.0
                pairs.append((idx_to_slug[i], idx_to_slug[j], int(cm[i, j]), share))
    pairs.sort(key=lambda t: (t[2], t[3]), reverse=True)
    return pairs[:top_k]


def main() -> None:
    parser = argparse.ArgumentParser(description="Évaluation CV groupée par contributeur")
    parser.add_argument("--dataset", default="src/data")
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--tag", default="run", help="Étiquette de l'itération pour le journal")
    parser.add_argument("--log", default="IMPROVE_LOG.md")
    parser.add_argument("--min-contrib-keep", type=int, default=0,
                        help="Ne garder que les classes ayant >= N contributeurs (0 = toutes)")
    # Overrides d'hyperparamètres (défaut = valeurs de model.train / classifier)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--label-smoothing", type=float, default=LABEL_SMOOTHING)
    parser.add_argument("--dropout", type=float, default=_DROPOUT)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    set_global_seed(SEED)
    device = torch.device("cpu")

    samples = load_dataset(Path(args.dataset))
    label_map = get_label_map(Path(args.dataset))
    if args.min_contrib_keep > 0:
        before = len(label_map)
        samples, label_map = filter_by_min_contrib(samples, label_map, args.min_contrib_keep)
        logger.info("Filtre min_contrib=%d : %d → %d classes, %d samples conservés",
                    args.min_contrib_keep, before, len(label_map), len(samples))
    n_classes = len(label_map)
    idx_to_slug = {v: k for k, v in label_map.items()}

    groups = np.array([s["contributor_id"] for s in samples])
    y = np.array([s["label_idx"] for s in samples])
    contrib = contributor_counts(samples, label_map)
    evaluable = {idx for idx, c in contrib.items() if c >= MIN_CONTRIB}
    logger.info("%d classes, %d évaluables (>=%d contributeurs)", n_classes, len(evaluable), MIN_CONTRIB)

    hp = {
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "dropout": args.dropout,
    }
    logger.info("Hyperparams : %s", hp)

    gkf = GroupKFold(n_splits=args.folds)
    all_true = np.full(len(samples), -1, dtype=int)
    all_pred = np.full(len(samples), -1, dtype=int)
    fold_accs: list[float] = []

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(samples, y, groups), start=1):
        tr = [samples[i] for i in tr_idx]
        va = [samples[i] for i in va_idx]
        tX, tY, vX, vY = build_fold_features(tr, va)
        model, best = train_one_fold(tX, tY, vX, vY, n_classes, args.epochs, device, hp)
        preds = predict(model, vX, device)
        all_true[va_idx] = np.array(vY)
        all_pred[va_idx] = preds
        fold_accs.append(best)
        logger.info("Fold %d/%d — %d train(+aug) / %d val — best_val_acc=%.2f%%",
                    fold, args.folds, len(tX), len(vX), best * 100)

    # --- Métriques poolées ---
    global_acc = float((all_true == all_pred).mean())
    eval_mask = np.array([t in evaluable for t in all_true])
    eval_acc = float((all_true[eval_mask] == all_pred[eval_mask]).mean()) if eval_mask.any() else float("nan")

    cm = confusion_matrix(all_true, all_pred, labels=list(range(n_classes)))
    per_class = []
    for idx in range(n_classes):
        total = int(cm[idx].sum())
        correct = int(cm[idx, idx])
        acc = correct / total if total > 0 else float("nan")
        per_class.append((idx_to_slug[idx], acc, correct, total, contrib[idx]))
    pairs = most_confused_pairs(cm, idx_to_slug, TOP_CONFUSED_PAIRS)

    logger.info("════════ %s ════════", args.tag)
    logger.info("Accuracy GLOBALE poolée (37 classes)       : %.2f%%", global_acc * 100)
    logger.info("Accuracy SOUS-ENSEMBLE ÉVALUABLE (%d cls)  : %.2f%%", len(evaluable), eval_acc * 100)
    logger.info("Moyenne best_val_acc par fold              : %.2f%% (+/- %.2f)",
                np.mean(fold_accs) * 100, np.std(fold_accs) * 100)

    append_log(args.log, args.tag, global_acc, eval_acc, fold_accs, per_class, pairs, args)


def append_log(log_path, tag, global_acc, eval_acc, fold_accs, per_class, pairs, args) -> None:
    """Ajoute une entrée d'itération au journal (sans écraser l'existant)."""
    L: list[str] = []
    L.append(f"\n## Itération — {tag}\n")
    L.append(f"- Protocole : GroupKFold k={args.folds} par contributeur, {args.epochs} epochs/fold, seed={SEED}.")
    L.append(f"- Hyperparams : lr={args.lr}, weight_decay={args.weight_decay}, "
             f"label_smoothing={args.label_smoothing}, dropout={args.dropout}, min_contrib_keep={args.min_contrib_keep}.")
    L.append(f"- **Accuracy globale poolée (37 classes) : {global_acc * 100:.2f}%**")
    L.append(f"- **Accuracy sous-ensemble évaluable (>= {MIN_CONTRIB} contrib) : {eval_acc * 100:.2f}%**")
    L.append(f"- Moyenne best_val_acc/fold : {np.mean(fold_accs) * 100:.2f}% (+/- {np.std(fold_accs) * 100:.2f})\n")
    L.append("### Rappel par classe (trié croissant) — `n_contrib` = nb contributeurs\n")
    L.append("| Signe | Rappel | Correct/Total | n_contrib |")
    L.append("|-------|--------|---------------|-----------|")
    for slug, acc, correct, total, nc in sorted(per_class, key=lambda t: (np.nan_to_num(t[1], nan=-1))):
        acc_str = f"{acc * 100:.1f}%" if total > 0 else "N/A"
        flag = " ⚠️" if nc < MIN_CONTRIB else ""
        L.append(f"| {slug}{flag} | {acc_str} | {correct}/{total} | {nc} |")
    L.append("\n### Paires les plus confondues (vrai → prédit)\n")
    L.append("| Vrai | Prédit | Erreurs | % de la classe |")
    L.append("|------|--------|---------|----------------|")
    for true_s, pred_s, n, share in pairs:
        L.append(f"| {true_s} | {pred_s} | {n} | {share * 100:.0f}% |")
    L.append("")

    existing = Path(log_path).read_text(encoding="utf-8") if Path(log_path).exists() else "# Journal d'amélioration du modèle LSF\n"
    Path(log_path).write_text(existing + "\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()

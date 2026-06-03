"""
Entraînement et évaluation du modèle LSFClassifier.

Toutes les fonctions sont indépendantes — pas de classe Trainer.
Le meilleur modèle (val accuracy) est sauvegardé automatiquement.
"""

import json
import logging
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset

from data.preprocessing import augment_sample, build_feature_vector
from model.classifier import make_classifier

logger = logging.getLogger(__name__)

# --- Constantes ---
BATCH_SIZE: int = 32
N_EPOCHS: int = 100
LR: float = 1e-3
WEIGHT_DECAY: float = 1e-4   # L2 régularisation — réduit l'overfit sur petit dataset
LABEL_SMOOTHING: float = 0.1  # empêche la sur-confiance, utile avec classes rares
WARMUP_EPOCHS: int = 5        # rampe LR linéaire avant le cosinus
DROPOUT: float = 0.4

# Plafond d'augmentation pour les classes très rares (1-3 samples).
# Au-delà, les variantes sont trop similaires et l'overfit empire.
_N_AUGMENTS_MAX: int = 12


def _compute_n_augments(count: int, target: int) -> int:
    """
    Calcule le nombre d'augmentations pour une classe à partir de son count.

    Cible : count * (n_augments + 1) ≈ target (classe la plus représentée).
    Plafonné à _N_AUGMENTS_MAX pour éviter l'overfit sur les classes isolées.

    Exemples avec target=51 :
        count=1  → 12 (plafonné)   →  13 samples
        count=3  → 12 (plafonné)   →  36 samples
        count=5  → 9              →  50 samples
        count=10 → 4              →  50 samples
        count=20 → 2              →  60 samples
        count=50 → 0              →  50 samples
    """
    if count >= target:
        return 0
    n_aug = round(target / count) - 1
    return max(0, min(n_aug, _N_AUGMENTS_MAX))


# ---------------------------------------------------------------------------
# Dataset PyTorch interne (non exporté)
# ---------------------------------------------------------------------------

class _LSFDataset(Dataset):
    """Dataset minimal qui stocke les feature vectors en mémoire."""

    def __init__(self, features: list[np.ndarray], labels: list[int]) -> None:
        # Pré-conversion en tenseurs pour éviter la copie répétée dans __getitem__
        self.X = [torch.tensor(fv, dtype=torch.float32) for fv in features]
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# ---------------------------------------------------------------------------
# 1. make_dataloaders
# ---------------------------------------------------------------------------

def make_dataloaders(
    samples: list[dict],
    label_map: dict[str, int],
    batch_size: int = BATCH_SIZE,
) -> tuple[DataLoader, DataLoader]:
    """
    Construit les DataLoaders train/val avec split par contributeur.

    Le split est fait sur les contributeurs (pas sur les samples) pour éviter
    que les frames d'une même personne apparaissent dans train ET val,
    ce qui gonflerait artificiellement les métriques.

    L'augmentation (augment_sample) est appliquée uniquement sur le train :
    - n_augments=8 pour les classes avec < 10 samples dans le train
    - n_augments=3 sinon

    Args:
        samples:    Liste de dicts issus de loader.load_dataset().
        label_map:  Dict {slug: idx} — pour le logging des classes.
        batch_size: Taille de batch (défaut BATCH_SIZE).

    Returns:
        (train_loader, val_loader)
    """
    contributor_ids = [s["contributor_id"] for s in samples]
    label_indices = [s["label_idx"] for s in samples]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(
        splitter.split(samples, label_indices, groups=contributor_ids)
    )

    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]

    # Distribution des classes dans le train pour calibrer l'augmentation
    train_class_counts: Counter = Counter(s["label_idx"] for s in train_samples)

    # Cible d'équilibrage = count de la classe la plus représentée dans le train
    target_per_class: int = max(train_class_counts.values()) if train_class_counts else 1

    idx_to_slug = {v: k for k, v in label_map.items()}
    logger.info("Split : %d train / %d val samples (avant augmentation)", len(train_samples), len(val_samples))
    logger.info("Distribution train (cible=%d samples/classe) :", target_per_class)
    for idx, count in sorted(train_class_counts.items(), key=lambda x: x[1]):
        slug = idx_to_slug.get(idx, str(idx))
        n_aug = _compute_n_augments(count, target_per_class)
        logger.info("  %-22s : %2d → ×%-2d = %d", slug, count, n_aug + 1, count * (n_aug + 1))

    # Construction des features train avec augmentation pondérée par classe
    train_X: list[np.ndarray] = []
    train_y: list[int] = []
    for s in train_samples:
        count = train_class_counts[s["label_idx"]]
        n_aug = _compute_n_augments(count, target_per_class)

        # augment_sample retourne [original] + n_aug variantes (toutes en (64, 318))
        variants = augment_sample(s["frames"], n_augments=n_aug)
        for v in variants:
            train_X.append(build_feature_vector(v))
            train_y.append(s["label_idx"])

    # Construction des features val sans augmentation
    val_X: list[np.ndarray] = []
    val_y: list[int] = []
    for s in val_samples:
        val_X.append(build_feature_vector(s["frames"]))
        val_y.append(s["label_idx"])

    logger.info("Après augmentation : %d train samples, %d val samples", len(train_X), len(val_X))

    train_loader = DataLoader(
        _LSFDataset(train_X, train_y),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        _LSFDataset(val_X, val_y),
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# 2. compute_class_weights
# ---------------------------------------------------------------------------

def compute_class_weights(labels: list[int], n_classes: int) -> torch.Tensor:
    """
    Calcule les poids de classe balancés pour CrossEntropyLoss.

    Utilise sklearn 'balanced' : poids = n_samples / (n_classes × bincount(y)).
    Les classes absentes des labels reçoivent un poids 1.0 (cas dégénéré où
    une classe entière passe en val, possible avec bonjour=2 samples).

    Args:
        labels:    Liste des label_idx du jeu d'entraînement (après augmentation).
        n_classes: Nombre total de classes.

    Returns:
        Tensor float32 de shape (n_classes,) pour CrossEntropyLoss(weight=...).
    """
    all_classes = np.arange(n_classes)
    labels_arr = np.array(labels)

    # Certaines classes peuvent manquer du train (ex: bonjour avec 2 samples)
    present_classes = np.unique(labels_arr)
    missing = set(range(n_classes)) - set(present_classes.tolist())
    if missing:
        missing_slugs = [str(c) for c in sorted(missing)]
        logger.warning("Classes absentes du train set : %s → poids=1.0", missing_slugs)

    weights = np.ones(n_classes, dtype=np.float32)
    if len(present_classes) > 0:
        partial_weights = compute_class_weight(
            class_weight="balanced",
            classes=present_classes,
            y=labels_arr,
        )
        for cls, w in zip(present_classes, partial_weights):
            weights[cls] = float(w)

    return torch.tensor(weights, dtype=torch.float32)


# ---------------------------------------------------------------------------
# 3. train_epoch
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    Effectue une epoch d'entraînement complète.

    Args:
        model:     Modèle en mode train().
        loader:    DataLoader du jeu d'entraînement.
        optimizer: Optimiseur (Adam recommandé).
        criterion: Fonction de loss (CrossEntropyLoss avec weights).
        device:    Dispositif de calcul (cpu).

    Returns:
        Loss moyenne sur l'epoch (pondérée par la taille de chaque batch).
    """
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(X_batch)

    return total_loss / len(loader.dataset)


# ---------------------------------------------------------------------------
# 4. eval_epoch
# ---------------------------------------------------------------------------

def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    Évalue le modèle sur un DataLoader (sans gradient).

    Args:
        model:     Modèle en mode eval().
        loader:    DataLoader du jeu de validation.
        criterion: Même fonction de loss que l'entraînement.
        device:    Dispositif de calcul.

    Returns:
        (loss_moyenne, accuracy) — accuracy en [0, 1].
    """
    model.eval()
    total_loss = 0.0
    correct = 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            total_loss += loss.item() * len(X_batch)
            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()

    n = len(loader.dataset)
    return total_loss / n, correct / n


# ---------------------------------------------------------------------------
# 5. _log_per_class_accuracy  (interne)
# ---------------------------------------------------------------------------

def _log_per_class_accuracy(
    model: nn.Module,
    loader: DataLoader,
    label_map: dict[str, int],
    device: torch.device,
) -> None:
    """
    Calcule et logue la précision par classe sur un DataLoader.
    Appelée une fois après l'entraînement sur le meilleur checkpoint.
    """
    model.eval()
    correct: Counter = Counter()
    total: Counter = Counter()

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).argmax(dim=1).cpu().numpy()
            for true_idx, pred_idx in zip(y_batch.numpy(), preds):
                total[int(true_idx)] += 1
                if true_idx == pred_idx:
                    correct[int(true_idx)] += 1

    idx_to_slug = {v: k for k, v in label_map.items()}
    logger.info("─── Précision par classe (meilleur modèle, val) ───")
    for idx in sorted(total.keys()):
        slug = idx_to_slug.get(idx, str(idx))
        acc = correct[idx] / total[idx] if total[idx] > 0 else 0.0
        bar = "#" * int(acc * 12) + "." * (12 - int(acc * 12))
        logger.info(
            "  %-22s [%s] %3.0f%%  (%d/%d)",
            slug, bar, acc * 100, correct[idx], total[idx],
        )
    logger.info("─" * 52)


# ---------------------------------------------------------------------------
# 6. train
# ---------------------------------------------------------------------------

def train(
    samples: list[dict],
    label_map: dict[str, int],
    n_epochs: int = N_EPOCHS,
    lr: float = LR,
    save_path: str = "models/lsf_v1.pt",
) -> None:
    """
    Orchestre l'entraînement complet du LSFClassifier.

    À chaque epoch, logue loss et accuracy train/val.
    Sauvegarde le meilleur modèle (val accuracy) dans save_path et
    le label_map dans models/label_map.json (même dossier que save_path).

    Args:
        samples:   Liste de dicts issus de loader.load_dataset().
        label_map: Dict {slug: idx} issu de loader.get_label_map().
        n_epochs:  Nombre d'epochs (défaut N_EPOCHS).
        lr:        Learning rate pour Adam (défaut LR).
        save_path: Chemin de sauvegarde du meilleur checkpoint.
    """
    device = torch.device("cpu")
    n_classes = len(label_map)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Dataloaders ---
    train_loader, val_loader = make_dataloaders(samples, label_map)

    # --- Modèle, optimiseur et scheduler ---
    model = make_classifier(n_classes=n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    # Warmup linéaire (lr/10 → lr) puis cosinus (lr → eta_min).
    # Le warmup évite les grands gradients en début d'entraînement quand les
    # poids sont encore aléatoires.
    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPOCHS)
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, n_epochs - WARMUP_EPOCHS), eta_min=1e-5)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[WARMUP_EPOCHS])

    # --- Loss avec class weights + label smoothing ---
    # label_smoothing=0.1 : distribue 10% de la probabilité sur les autres classes,
    # évite que le modèle soit trop sûr de lui sur les classes rares sur-augmentées.
    train_labels = [int(y) for _, ys in train_loader for y in ys]
    class_weights = compute_class_weights(train_labels, n_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)

    logger.info("Modèle : %d paramètres", sum(p.numel() for p in model.parameters()))
    logger.info("Début entraînement : %d epochs, lr=%.4f, device=%s", n_epochs, lr, device)

    best_val_acc = 0.0

    for epoch in range(1, n_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        logger.info(
            "Epoch %3d/%d — train_loss=%.4f | val_loss=%.4f | val_acc=%.2f%%  (lr=%.2e)",
            epoch, n_epochs, train_loss, val_loss, val_acc * 100,
            optimizer.param_groups[0]["lr"],
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_acc": val_acc,
                    "n_classes": n_classes,
                    "label_map": label_map,
                },
                save_path,
            )
            logger.info("  → Meilleur modèle sauvegardé (val_acc=%.2f%%)", val_acc * 100)

    # Sauvegarde séparée du label_map pour l'inférence
    label_map_path = save_path.parent / "label_map.json"
    with label_map_path.open("w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    logger.info("label_map sauvegardé dans %s", label_map_path)

    logger.info("Entraînement terminé. Meilleure val_acc : %.2f%%", best_val_acc * 100)

    # Recharge le meilleur modèle pour le diagnostic final
    best_ckpt = torch.load(save_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    _log_per_class_accuracy(model, val_loader, label_map, device)

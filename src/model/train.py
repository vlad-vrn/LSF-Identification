"""
Entraînement et évaluation du modèle LSFClassifier.

Toutes les fonctions sont indépendantes — pas de classe Trainer.
Le meilleur modèle (val accuracy) est sauvegardé automatiquement.
"""

import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset

from data.preprocessing import augment_sample, build_feature_vector
from model.classifier import make_classifier

logger = logging.getLogger(__name__)

# --- Constantes ---
BATCH_SIZE: int = 32
N_EPOCHS: int = 50
LR: float = 1e-3
DROPOUT: float = 0.3   # utilisé par LSFClassifier, rappelé ici pour cohérence

# Seuil de classes sous-représentées : n_augments plus agressif en dessous
_SMALL_CLASS_THRESHOLD: int = 10
_N_AUGMENTS_SMALL: int = 8
_N_AUGMENTS_NORMAL: int = 3


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

    idx_to_slug = {v: k for k, v in label_map.items()}
    logger.info("Split : %d train / %d val samples (avant augmentation)", len(train_samples), len(val_samples))
    logger.info("Distribution train par classe :")
    for idx, count in sorted(train_class_counts.items()):
        slug = idx_to_slug.get(idx, str(idx))
        n_aug = _N_AUGMENTS_SMALL if count < _SMALL_CLASS_THRESHOLD else _N_AUGMENTS_NORMAL
        logger.info("  %-20s : %2d samples → ×%d (total %d)", slug, count, n_aug + 1, count * (n_aug + 1))

    # Construction des features train avec augmentation
    train_X: list[np.ndarray] = []
    train_y: list[int] = []
    for s in train_samples:
        count = train_class_counts[s["label_idx"]]
        n_aug = _N_AUGMENTS_SMALL if count < _SMALL_CLASS_THRESHOLD else _N_AUGMENTS_NORMAL

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
# 5. train
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

    # --- Modèle et optimiseur ---
    model = make_classifier(n_classes=n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # --- Loss avec class weights (calculés sur les labels train augmentés) ---
    train_labels = [int(y) for _, ys in train_loader for y in ys]
    class_weights = compute_class_weights(train_labels, n_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    logger.info("Modèle : %d paramètres", sum(p.numel() for p in model.parameters()))
    logger.info("Début entraînement : %d epochs, lr=%.4f, device=%s", n_epochs, lr, device)

    best_val_acc = 0.0

    for epoch in range(1, n_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)

        logger.info(
            "Epoch %3d/%d — train_loss=%.4f | val_loss=%.4f | val_acc=%.2f%%",
            epoch, n_epochs, train_loss, val_loss, val_acc * 100,
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

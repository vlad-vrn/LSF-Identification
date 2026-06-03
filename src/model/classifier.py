"""
Définition du modèle 1D-CNN pour la classification de signes LSF.

Input : (B, 64, 320) — batch de séquences de 64 frames × 320 features.
Output : (B, n_classes) — logits bruts (pas de softmax).
"""

import torch
import torch.nn as nn

# Dimensions fixes du pipeline
SEQUENCE_LENGTH = 64
FEATURE_DIM = 320


class LSFClassifier(nn.Module):
    """
    1D-CNN pour classer des séquences de keypoints LSF.

    Architecture choisie :
      Deux blocs Conv1d → ReLU → MaxPool1d, suivis d'un MLP.

    Choix de conception :
    - kernel_size=5 sur le premier bloc : fenêtre temporelle de 5 frames
      (~160 ms à 30fps) pour capturer les débuts de mouvement.
    - kernel_size=3 sur le second : affine la représentation sur une fenêtre
      plus courte une fois les features compressées.
    - MaxPool1d(2) × 2 : divise la longueur temporelle par 4 (64 → 16),
      rendant le modèle robuste aux décalages temporels légers.
    - Pas de BatchNorm : dataset trop petit pour que la statistique de batch
      soit stable ; le z-score du preprocessing remplace cette normalisation.
    - Dropout(0.3) avant la couche finale : régularisation principale.
    - Pas de Softmax en sortie : CrossEntropyLoss l'inclut dans PyTorch.
    """

    def __init__(self, n_classes: int) -> None:
        """
        Args:
            n_classes: Nombre de signes à classifier.
        """
        super().__init__()

        # Bloc 1 : extrait les patterns locaux sur 5 frames
        # (B, 320, 64) → (B, 128, 64) → (B, 128, 32)
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=FEATURE_DIM, out_channels=128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        # Bloc 2 : combine les patterns locaux sur 3 frames
        # (B, 128, 32) → (B, 64, 32) → (B, 64, 16)
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        # Tête de classification
        # (B, 64*16) → (B, 128) → (B, n_classes)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape (B, 64, 320) — batch de séquences.

        Returns:
            Logits shape (B, n_classes).
        """
        # Conv1d attend (B, C, L) : on transpose features ↔ frames
        x = x.transpose(1, 2)   # (B, 320, 64)
        x = self.conv1(x)        # (B, 128, 32)
        x = self.conv2(x)        # (B, 64, 16)
        return self.classifier(x)  # (B, n_classes)


def make_classifier(n_classes: int) -> LSFClassifier:
    """
    Crée une instance de LSFClassifier avec les poids initialisés par défaut.

    Args:
        n_classes: Nombre de classes (signes) à classifier.

    Returns:
        Modèle non entraîné prêt pour l'entraînement.
    """
    return LSFClassifier(n_classes=n_classes)


if __name__ == "__main__":
    x = torch.randn(4, 64, 320)
    model = make_classifier(12)
    out = model(x)
    assert out.shape == (4, 12), f"Expected (4, 12), got {out.shape}"
    print("Shape OK:", out.shape)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parametres total : {total_params:,}")

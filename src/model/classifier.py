"""
Définition du modèle 1D-CNN pour la classification de signes LSF.

Input  : (B, 64, 320)  — batch de séquences normalisées
Output : (B, n_classes) — logits bruts (pas de softmax)
"""

import torch
import torch.nn as nn

SEQUENCE_LENGTH = 64
FEATURE_DIM = 320

# Constantes d'architecture — modifier ici uniquement
_CHANNELS = (64, 128, 128)   # filtres par bloc (bloc1, bloc2, bloc3)
_KERNELS   = (7,   5,   3)   # kernel_size par bloc (large → fin)
_FC_HIDDEN = 64              # unités couche FC intermédiaire
_DROPOUT   = 0.4             # dropout avant la couche finale


class LSFClassifier(nn.Module):
    """
    3 blocs Conv1d + BatchNorm1d + ReLU + MaxPool1d, suivi de GlobalMaxPool et d'un MLP.

    Changements vs V1 :
    - BatchNorm1d après chaque conv : stabilise les gradients, permet un LR plus élevé
      et réduit le surapprentissage même sur petit dataset (les stats de batch sur
      32×64=2048 activations sont suffisantes).
    - 3e bloc conv (k=3) : features plus fines après deux compressions temporelles.
    - GlobalMaxPool au lieu de Flatten : robuste aux décalages temporels,
      ramène la tête de 1024→128 (362k params) à 128→64 (~150k params).
    - LayerNorm dans la tête FC : normalise les activations post-GlobalMax.
    - Dropout porté à 0.4 pour compenser l'expressivité accrue du 3e bloc.
    """

    def __init__(self, n_classes: int) -> None:
        super().__init__()
        self.n_classes = n_classes

        # (B, 320, 64) → (B, 64,  32)
        self.conv1 = self._conv_block(FEATURE_DIM,   _CHANNELS[0], _KERNELS[0])
        # (B, 64,  32) → (B, 128, 16)
        self.conv2 = self._conv_block(_CHANNELS[0],   _CHANNELS[1], _KERNELS[1])
        # (B, 128, 16) → (B, 128,  8)
        self.conv3 = self._conv_block(_CHANNELS[1],   _CHANNELS[2], _KERNELS[2])

        # GlobalMaxPool → (B, 128) puis tête de classification
        self.head = nn.Sequential(
            nn.Linear(_CHANNELS[2], _FC_HIDDEN),
            nn.LayerNorm(_FC_HIDDEN),
            nn.ReLU(),
            nn.Dropout(_DROPOUT),
            nn.Linear(_FC_HIDDEN, n_classes),
        )

    @staticmethod
    def _conv_block(in_ch: int, out_ch: int, kernel: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel, padding=kernel // 2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape (B, 64, 320)

        Returns:
            Logits shape (B, n_classes).
        """
        x = x.transpose(1, 2)    # (B, 320, 64) — Conv1d attend (B, C, L)
        x = self.conv1(x)         # (B, 64,  32)
        x = self.conv2(x)         # (B, 128, 16)
        x = self.conv3(x)         # (B, 128,  8)
        x = x.max(dim=2).values   # GlobalMaxPool → (B, 128)
        return self.head(x)       # (B, n_classes)


def make_classifier(n_classes: int) -> LSFClassifier:
    return LSFClassifier(n_classes=n_classes)


if __name__ == "__main__":
    x = torch.randn(4, 64, 320)
    model = make_classifier(12)
    out = model(x)
    assert out.shape == (4, 12), f"Expected (4, 12), got {out.shape}"
    print("Shape OK:", out.shape)
    print(f"Paramètres : {sum(p.numel() for p in model.parameters()):,}")

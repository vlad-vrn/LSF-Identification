# LSF-Direct — Recognition module

Projet académique ESILV. Module de reconnaissance de Langue des Signes Française en temps réel à partir de keypoints MediaPipe. Repo séparé du site de collecte (partie collègues).

## Stack

- Python 3.11+
- MediaPipe Holistic
- PyTorch (CPU only — cibler laptop standard, pas de CUDA)
- OpenCV (fenêtre de visualisation)
- pandas + pyarrow (lecture dataset parquet)
- scikit-learn (preprocessing, métriques)

## Architecture du repo

```
lsf-direct-recognition/
├── CLAUDE.md
├── requirements.txt
├── dataset/
│   └── approved/              # Données brutes — NE PAS MODIFIER
│       ├── bonjour/
│       │   ├── <uuid>.parquet
│       │   └── <uuid>.meta.json
│       ├── aide/
│       └── ...                # 12 signes au total
├── models/                    # Modèles entraînés sauvegardés
│   └── .gitkeep
├── src/
│   ├── data/
│   │   ├── loader.py          # Chargement dataset parquet
│   │   └── preprocessing.py  # Normalisation, augmentation
│   ├── model/
│   │   ├── train.py           # Entraînement + évaluation
│   │   └── classifier.py     # Définition du modèle 1D-CNN
│   ├── inference/
│   │   ├── pipeline.py        # Pipeline temps réel (buffer, prédiction, dédup)
│   │   └── mediapipe_extractor.py  # Extraction keypoints depuis webcam
│   └── display/
│       └── overlay.py         # Fenêtre OpenCV + affichage résultat
├── scripts/
│   ├── train.py               # Point d'entrée entraînement
│   └── run.py                 # Point d'entrée démonstration temps réel
└── notebooks/
    └── explore_dataset.ipynb  # Exploration data (optionnel)
```

## Format du dataset

Chaque sample = fichier `.parquet` + `.meta.json` dans `dataset/approved/<slug>/`.

Le dossier `dataset/` peut aussi contenir un sous-dossier `raw/pending/` avec des samples non encore validés — **les ignorer complètement** au chargement. Ne charger que `approved/`.

Un parquet contient exactement **64 frames** avec les colonnes :

| Colonne      | Type     | Détail                                       |
|--------------|----------|----------------------------------------------|
| `frame_idx`  | int32    | 0 → 63                                       |
| `pose`       | float[]  | 132 valeurs = 33 landmarks × (x, y, z, vis)  |
| `left_hand`  | float[]  | 63 valeurs = 21 landmarks × (x, y, z)        |
| `right_hand` | float[]  | 63 valeurs = 21 landmarks × (x, y, z)        |
| `face`       | float[]  | 60 valeurs = 15 landmarks × (x, y, z, ?)     |

**Vecteur feature par frame : 318 floats.** Input modèle : (64, 318).

Les mains absentes sont stockées comme vecteurs de zéros (pas de None/NaN).

Le fichier `.meta.json` ne contient qu'un seul champ : `{"contributor_id": "<uuid>"}`.
Les scores de qualité (movement_score, visibility_score, etc.) sont dans la base PostgreSQL
du site de collecte — ils ne sont pas disponibles ici. Ne pas en dépendre.

## Signes disponibles (V1) — distribution réelle

12 signes (273 samples approuvés au total, dataset évolutif) :

| Signe            | Samples | Main dominante détectée |
|------------------|---------|-------------------------|
| soif             | 50      | droite majoritaire      |
| boire            | 50      | droite majoritaire      |
| aujourd_hui      | 39      | mixte                   |
| aimer            | 34      | gauche majoritaire*     |
| aller            | 21      | mixte                   |
| je               | 20      | droite majoritaire      |
| aide             | 17      | gauche majoritaire*     |
| au_revoir        | 13      | gauche uniquement*      |
| toi              | 12      | droite majoritaire      |
| comprendre       | 12      | gauche majoritaire*     |
| s_il_vous_plait  | 3       | très sous-représenté    |
| bonjour          | 2       | très sous-représenté    |

*= effet miroir webcam confirmé (voir problème #1 ci-dessous)

**Déséquilibre sévère** : `bonjour` (2) et `s_il_vous_plait` (3) sont quasi-inutilisables
sans augmentation agressive. Prévoir class weights dans la loss ou oversampling.

**Contributeurs** : 29 contributeurs distincts identifiés par `contributor_id`.
Utiliser cet ID pour faire une split train/val **par contributeur** (pas aléatoire),
afin d'éviter que les frames d'une même personne se retrouvent dans les deux sets.

## Problèmes connus à gérer — CRITIQUE

### 1. Main dominante vs côté image

La webcam en mode miroir (comportement navigateur par défaut) inverse gauche/droite.
Un droitier signant de la main droite apparaît à gauche dans le flux → enregistré comme
`left_hand`. Cet effet est **confirmé par le dataset** : pour certains signes (aide, aimer,
comprendre, au_revoir), la main gauche est dominante pour 80-100% des samples, ce qui
est incohérent biologiquement (majorité de droitiers) et s'explique par le miroir.

**Solution implémentée** : définir "main dominante" comme la main avec le plus de
mouvement sur la séquence (norme L2 des déplacements frame-à-frame), indépendamment
du côté. Renommer en `dominant_hand` / `passive_hand` avant toute normalisation.
Appliquer la même logique en inférence.

```python
# Calcul mouvement total d'une main sur la séquence
def hand_movement(frames: np.ndarray) -> float:
    # frames : (64, 63)
    diffs = np.diff(frames, axis=0)           # (63, 63) déplacements frame-à-frame
    return float(np.linalg.norm(diffs))        # scalaire : mouvement total
```

### 2. Frames de repos en début/fin de signe

Les samples commencent et finissent mains au repos (le long du corps). Cela ne se produit
pas en flux continu où les signes s'enchaînent.

**Solution implémentée** :
- Trim automatique : calculer le mouvement frame-à-frame (norme déplacement keypoints),
  supprimer les frames en dessous d'un seuil au début et à la fin.
- Augmentation : générer des variantes avec trim aléatoire du début (0 à 10 frames).

### 3. Main passive à zéros — biais de poids

Pour les signes unimanuels, la main passive est à zéros sur toute la séquence
(~50% des frames toutes mains confondues).

**Solution implémentée** : ajouter un flag binaire par frame par main (1 = détectée,
0 = absente). Le vecteur feature devient 318 + 2 = 320 floats. Ne pas supprimer la main
passive — garder ses zéros mais donner au modèle l'info explicite qu'elle est absente.

Le flag est calculé ainsi :
```python
left_present  = 0.0 if np.all(left_hand_frame == 0) else 1.0
right_present = 0.0 if np.all(right_hand_frame == 0) else 1.0
```

### 4. Déséquilibre de classes — CRITIQUE pour l'entraînement

Avec `bonjour`=2 et `s_il_vous_plait`=3, un entraînement naïf ignorera ces classes.

**Solutions** :
- `class_weight='balanced'` dans la loss PyTorch (via `torch.nn.CrossEntropyLoss(weight=...)`)
- Calculer les poids depuis `sklearn.utils.class_weight.compute_class_weight`
- Augmentation x5 minimum sur les classes sous-représentées

### 5. Split train/val par contributeur

Ne pas faire de split aléatoire sur les samples — cela ferait fuiter les données d'un même
contributeur dans train ET val, gonflant artificiellement les métriques.

**Solution** : récupérer les `contributor_id` depuis les `.meta.json`, puis split 80/20
**au niveau des contributeurs** (GroupShuffleSplit de scikit-learn).

## Pipeline inférence temps réel

```
webcam frame
  → MediaPipe Holistic extraction (pose + mains + face)
  → Normalisation (même preprocessing que training)
  → Buffer circulaire 64 frames (stride 15 frames ≈ 0.5s @ 30fps)
  → Si buffer plein → inférence modèle → (label, confiance)
  → Déduplication : si même label N=3 fois consécutives ET confiance > seuil → confirmer signe
  → Ajouter à la séquence de signes détectés
  → Affichage OpenCV : flux vidéo + signe courant + phrase accumulée
```

**Pas de fenêtre fixe de 3s** : le buffer glissant avec stride permet une détection en
continu sans pause perceptible.

**Stride = 15** : à 30fps, stride=15 = nouvelle inférence toutes les 0.5s. Assez réactif
sans saturer le CPU. Ajustable via constante `INFERENCE_STRIDE` dans `pipeline.py`.

**Seuil confiance = 0.6** : valeur de départ raisonnable pour réduire les faux positifs.
La déduplication N=3 ajoute un filtre temporel supplémentaire.

## Conventions de code

- Écrire du code que je peux lire et expliquer — pas de one-liners cryptiques
- Commenter les choix non-évidents (pourquoi stride=15, pourquoi seuil=0.6, etc.)
- Typage explicite (type hints Python)
- Pas de classes sur-engineered pour la V1 — fonctions simples suffisent
- Logging via `logging` standard (pas de print sauf scripts)
- Constantes nommées en MAJUSCULES en tête de fichier (pas de magic numbers)

## Ce qui est hors scope V1

- Post-traitement LM pour assembler la phrase (simple concaténation suffit)
- GPU / CUDA
- Gestion des gauchers en inférence (la normalisation par dominance gère déjà le cas)
- Interface web
- Modèle pré-entraîné type transfer learning
- Lecture des scores de qualité depuis la DB PostgreSQL (hors scope recognition)

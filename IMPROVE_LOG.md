# Journal d'amélioration du modèle LSF

## Itération 0 — BASELINE (état courant, non modifié)

- Config : 37 signes, split par contributeur (seed=42), 100 epochs, 265 samples de validation.
- **Accuracy globale (val) : 49.81%**

### Accuracy par classe (triée croissante)

| Signe | Rappel | Correct/Total |
|-------|--------|---------------|
| aller | N/A (0 en val) | 0/0 |
| aujourd_hui | N/A (0 en val) | 0/0 |
| avoir | N/A (0 en val) | 0/0 |
| comprendre | N/A (0 en val) | 0/0 |
| dormir | N/A (0 en val) | 0/0 |
| etre | N/A (0 en val) | 0/0 |
| je | N/A (0 en val) | 0/0 |
| maison | N/A (0 en val) | 0/0 |
| merci | N/A (0 en val) | 0/0 |
| ou | N/A (0 en val) | 0/0 |
| quand | N/A (0 en val) | 0/0 |
| quoi | N/A (0 en val) | 0/0 |
| toi | N/A (0 en val) | 0/0 |
| travail | N/A (0 en val) | 0/0 |
| vouloir | N/A (0 en val) | 0/0 |
| vous | N/A (0 en val) | 0/0 |
| au_revoir | 0.0% | 0/13 |
| bonjour | 0.0% | 0/1 |
| bonne_nuit | 0.0% | 0/5 |
| demain | 0.0% | 0/7 |
| excuse_moi | 0.0% | 0/1 |
| faim | 0.0% | 0/12 |
| maintenant | 0.0% | 0/6 |
| nous | 0.0% | 0/11 |
| salut | 0.0% | 0/5 |
| savoir | 0.0% | 0/12 |
| comment | 14.3% | 1/7 |
| aide | 19.0% | 8/42 |
| s_il_vous_plait | 60.0% | 6/10 |
| manger | 64.0% | 16/25 |
| pardon | 82.6% | 19/23 |
| a_bientot | 88.5% | 23/26 |
| aimer | 100.0% | 26/26 |
| boire | 100.0% | 5/5 |
| ecole | 100.0% | 9/9 |
| pourquoi | 100.0% | 4/4 |
| soif | 100.0% | 15/15 |

### Paires les plus confondues (vrai → prédit)

| Vrai | Prédit | Erreurs | % de la classe |
|------|--------|---------|----------------|
| aide | pardon | 12 | 29% |
| au_revoir | comprendre | 8 | 62% |
| aide | comment | 7 | 17% |
| aide | travail | 7 | 17% |
| comment | ou | 6 | 86% |
| savoir | merci | 6 | 50% |
| manger | aide | 6 | 24% |
| demain | pourquoi | 5 | 71% |
| au_revoir | pourquoi | 5 | 38% |
| bonne_nuit | merci | 3 | 60% |
| maintenant | vouloir | 3 | 50% |
| faim | bonjour | 3 | 25% |

## Itération — 0bis - baseline GroupKFold honnete (37 classes)

- Protocole : GroupKFold k=5 par contributeur, 60 epochs/fold, seed=42.
- **Accuracy globale poolée (37 classes) : 60.10%**
- **Accuracy sous-ensemble évaluable (>= 3 contrib) : 68.50%**
- Moyenne best_val_acc/fold : 60.09% (+/- 11.37)

### Rappel par classe (trié croissant) — `n_contrib` = nb contributeurs

| Signe | Rappel | Correct/Total | n_contrib |
|-------|--------|---------------|-----------|
| comprendre | 0.0% | 0/29 | 4 |
| demain | 0.0% | 0/31 | 3 |
| faim ⚠️ | 0.0% | 0/12 | 1 |
| je ⚠️ | 0.0% | 0/53 | 2 |
| maintenant ⚠️ | 0.0% | 0/6 | 1 |
| maison ⚠️ | 0.0% | 0/12 | 2 |
| ou ⚠️ | 0.0% | 0/21 | 2 |
| quand | 0.0% | 0/13 | 5 |
| salut ⚠️ | 0.0% | 0/5 | 1 |
| savoir ⚠️ | 0.0% | 0/12 | 2 |
| travail ⚠️ | 0.0% | 0/4 | 2 |
| vouloir ⚠️ | 0.0% | 0/10 | 1 |
| vous ⚠️ | 0.0% | 0/12 | 1 |
| etre | 11.1% | 1/9 | 3 |
| merci | 26.9% | 7/26 | 3 |
| au_revoir | 28.0% | 14/50 | 6 |
| toi | 40.0% | 20/50 | 5 |
| excuse_moi ⚠️ | 50.0% | 1/2 | 2 |
| quoi | 56.5% | 13/23 | 3 |
| aujourd_hui | 62.0% | 31/50 | 4 |
| bonjour | 72.7% | 32/44 | 11 |
| nous ⚠️ | 72.7% | 8/11 | 2 |
| soif | 74.0% | 37/50 | 4 |
| pardon | 75.7% | 28/37 | 4 |
| manger | 75.9% | 41/54 | 4 |
| aimer | 76.5% | 39/51 | 6 |
| pourquoi | 77.9% | 53/68 | 5 |
| aide | 79.7% | 47/59 | 5 |
| bonne_nuit | 82.6% | 38/46 | 5 |
| aller | 84.0% | 42/50 | 4 |
| s_il_vous_plait | 84.4% | 27/32 | 5 |
| comment | 86.0% | 43/50 | 4 |
| ecole | 88.2% | 45/51 | 8 |
| boire | 90.0% | 45/50 | 4 |
| avoir | 93.0% | 40/43 | 4 |
| a_bientot | 93.7% | 59/63 | 8 |
| dormir | 100.0% | 9/9 | 3 |

### Paires les plus confondues (vrai → prédit)

| Vrai | Prédit | Erreurs | % de la classe |
|------|--------|---------|----------------|
| demain | merci | 23 | 74% |
| je | excuse_moi | 17 | 32% |
| au_revoir | pourquoi | 16 | 32% |
| aujourd_hui | etre | 15 | 30% |
| ou | quoi | 11 | 52% |
| pourquoi | au_revoir | 11 | 16% |
| je | bonjour | 10 | 19% |
| je | toi | 10 | 19% |
| vouloir | maintenant | 9 | 90% |
| comprendre | au_revoir | 9 | 31% |
| soif | manger | 8 | 16% |
| toi | manger | 8 | 16% |
| aller | etre | 7 | 14% |
| maison | comment | 6 | 50% |
| vous | aujourd_hui | 6 | 50% |

## Itération — 1 - normalisation geometrique corps-relative (remplace z-score)

- Protocole : GroupKFold k=5 par contributeur, 60 epochs/fold, seed=42.
- **Accuracy globale poolée (37 classes) : 65.61%**
- **Accuracy sous-ensemble évaluable (>= 3 contrib) : 75.24%**
- Moyenne best_val_acc/fold : 65.59% (+/- 13.25)

### Rappel par classe (trié croissant) — `n_contrib` = nb contributeurs

| Signe | Rappel | Correct/Total | n_contrib |
|-------|--------|---------------|-----------|
| comprendre | 0.0% | 0/29 | 4 |
| demain | 0.0% | 0/31 | 3 |
| etre | 0.0% | 0/9 | 3 |
| excuse_moi ⚠️ | 0.0% | 0/2 | 2 |
| faim ⚠️ | 0.0% | 0/12 | 1 |
| je ⚠️ | 0.0% | 0/53 | 2 |
| maintenant ⚠️ | 0.0% | 0/6 | 1 |
| ou ⚠️ | 0.0% | 0/21 | 2 |
| salut ⚠️ | 0.0% | 0/5 | 1 |
| savoir ⚠️ | 0.0% | 0/12 | 2 |
| travail ⚠️ | 0.0% | 0/4 | 2 |
| vouloir ⚠️ | 0.0% | 0/10 | 1 |
| vous ⚠️ | 0.0% | 0/12 | 1 |
| quand | 7.7% | 1/13 | 5 |
| maison ⚠️ | 8.3% | 1/12 | 2 |
| merci | 19.2% | 5/26 | 3 |
| au_revoir | 36.0% | 18/50 | 6 |
| nous ⚠️ | 36.4% | 4/11 | 2 |
| toi | 62.0% | 31/50 | 5 |
| aide | 74.6% | 44/59 | 5 |
| bonjour | 75.0% | 33/44 | 11 |
| manger | 77.8% | 42/54 | 4 |
| quoi | 78.3% | 18/23 | 3 |
| pourquoi | 79.4% | 54/68 | 5 |
| aimer | 82.4% | 42/51 | 6 |
| comment | 88.0% | 44/50 | 4 |
| pardon | 89.2% | 33/37 | 4 |
| a_bientot | 90.5% | 57/63 | 8 |
| avoir | 90.7% | 39/43 | 4 |
| bonne_nuit | 91.3% | 42/46 | 5 |
| aller | 92.0% | 46/50 | 4 |
| s_il_vous_plait | 93.8% | 30/32 | 5 |
| aujourd_hui | 94.0% | 47/50 | 4 |
| soif | 94.0% | 47/50 | 4 |
| boire | 98.0% | 49/50 | 4 |
| ecole | 98.0% | 50/51 | 8 |
| dormir | 100.0% | 9/9 | 3 |

### Paires les plus confondues (vrai → prédit)

| Vrai | Prédit | Erreurs | % de la classe |
|------|--------|---------|----------------|
| je | bonjour | 33 | 62% |
| ou | quoi | 17 | 81% |
| comprendre | pourquoi | 17 | 59% |
| merci | pourquoi | 15 | 58% |
| au_revoir | pourquoi | 14 | 28% |
| je | toi | 13 | 25% |
| comprendre | au_revoir | 11 | 38% |
| aide | maison | 11 | 19% |
| vouloir | maintenant | 10 | 100% |
| pourquoi | au_revoir | 10 | 15% |
| demain | comprendre | 9 | 29% |
| demain | merci | 9 | 29% |
| au_revoir | a_bientot | 9 | 18% |
| vous | faim | 8 | 67% |
| faim | bonjour | 7 | 58% |

## Itération — 2 - ajout features de velocite (positions+vitesses, dim 638) — ❌ REVERTÉ

> Décision : REVERTÉ. −0,9 pt global / −1,2 pt évaluable vs axe 1. N'a pas sauvé
> le cluster ciblé (comprendre/demain/quand restent à 0 %, comprendre→pourquoi
> empire à 83 %) et a introduit une confusion aimer↔toi. Meilleur courant = axe 1.

- Protocole : GroupKFold k=5 par contributeur, 60 epochs/fold, seed=42.
- **Accuracy globale poolée (37 classes) : 64.69%**
- **Accuracy sous-ensemble évaluable (>= 3 contrib) : 74.08%**
- Moyenne best_val_acc/fold : 64.67% (+/- 13.10)

### Rappel par classe (trié croissant) — `n_contrib` = nb contributeurs

| Signe | Rappel | Correct/Total | n_contrib |
|-------|--------|---------------|-----------|
| comprendre | 0.0% | 0/29 | 4 |
| demain | 0.0% | 0/31 | 3 |
| excuse_moi ⚠️ | 0.0% | 0/2 | 2 |
| faim ⚠️ | 0.0% | 0/12 | 1 |
| je ⚠️ | 0.0% | 0/53 | 2 |
| maintenant ⚠️ | 0.0% | 0/6 | 1 |
| maison ⚠️ | 0.0% | 0/12 | 2 |
| ou ⚠️ | 0.0% | 0/21 | 2 |
| quand | 0.0% | 0/13 | 5 |
| salut ⚠️ | 0.0% | 0/5 | 1 |
| savoir ⚠️ | 0.0% | 0/12 | 2 |
| travail ⚠️ | 0.0% | 0/4 | 2 |
| vouloir ⚠️ | 0.0% | 0/10 | 1 |
| vous ⚠️ | 0.0% | 0/12 | 1 |
| etre | 11.1% | 1/9 | 3 |
| merci | 19.2% | 5/26 | 3 |
| aimer | 33.3% | 17/51 | 6 |
| au_revoir | 40.0% | 20/50 | 6 |
| nous ⚠️ | 54.5% | 6/11 | 2 |
| toi | 66.0% | 33/50 | 5 |
| quoi | 73.9% | 17/23 | 3 |
| a_bientot | 76.2% | 48/63 | 8 |
| bonjour | 79.5% | 35/44 | 11 |
| aide | 81.4% | 48/59 | 5 |
| aujourd_hui | 82.0% | 41/50 | 4 |
| manger | 85.2% | 46/54 | 4 |
| dormir | 88.9% | 8/9 | 3 |
| bonne_nuit | 89.1% | 41/46 | 5 |
| s_il_vous_plait | 90.6% | 29/32 | 5 |
| comment | 92.0% | 46/50 | 4 |
| avoir | 93.0% | 40/43 | 4 |
| pourquoi | 94.1% | 64/68 | 5 |
| pardon | 94.6% | 35/37 | 4 |
| boire | 96.0% | 48/50 | 4 |
| soif | 96.0% | 48/50 | 4 |
| aller | 98.0% | 49/50 | 4 |
| ecole | 98.0% | 50/51 | 8 |

### Paires les plus confondues (vrai → prédit)

| Vrai | Prédit | Erreurs | % de la classe |
|------|--------|---------|----------------|
| je | bonjour | 32 | 60% |
| aimer | toi | 26 | 51% |
| comprendre | pourquoi | 24 | 83% |
| demain | comprendre | 23 | 74% |
| je | toi | 18 | 34% |
| ou | quoi | 17 | 81% |
| vous | faim | 11 | 92% |
| au_revoir | a_bientot | 11 | 22% |
| au_revoir | comprendre | 11 | 22% |
| a_bientot | comprendre | 11 | 17% |
| savoir | vous | 10 | 83% |
| maison | aide | 8 | 67% |
| merci | toi | 8 | 31% |
| vouloir | maintenant | 7 | 70% |
| toi | aimer | 7 | 14% |

## Itération — V1 baseline 25 classes (axe1, >=3 contrib)

- Protocole : GroupKFold k=5 par contributeur, 60 epochs/fold, seed=42.
- **Accuracy globale poolée (37 classes) : 74.95%**
- **Accuracy sous-ensemble évaluable (>= 3 contrib) : 74.95%**
- Moyenne best_val_acc/fold : 74.95% (+/- 12.24)

### Rappel par classe (trié croissant) — `n_contrib` = nb contributeurs

| Signe | Rappel | Correct/Total | n_contrib |
|-------|--------|---------------|-----------|
| demain | 0.0% | 0/31 | 3 |
| etre | 0.0% | 0/9 | 3 |
| quoi | 0.0% | 0/23 | 3 |
| quand | 7.7% | 1/13 | 5 |
| comprendre | 10.3% | 3/29 | 4 |
| merci | 26.9% | 7/26 | 3 |
| aimer | 31.4% | 16/51 | 6 |
| au_revoir | 66.0% | 33/50 | 6 |
| toi | 70.0% | 35/50 | 5 |
| pourquoi | 77.9% | 53/68 | 5 |
| manger | 79.6% | 43/54 | 4 |
| aide | 81.4% | 48/59 | 5 |
| bonjour | 88.6% | 39/44 | 11 |
| a_bientot | 88.9% | 56/63 | 8 |
| bonne_nuit | 89.1% | 41/46 | 5 |
| pardon | 89.2% | 33/37 | 4 |
| aujourd_hui | 90.0% | 45/50 | 4 |
| avoir | 93.0% | 40/43 | 4 |
| aller | 94.0% | 47/50 | 4 |
| ecole | 96.1% | 49/51 | 8 |
| comment | 98.0% | 49/50 | 4 |
| soif | 98.0% | 49/50 | 4 |
| boire | 100.0% | 50/50 | 4 |
| dormir | 100.0% | 9/9 | 3 |
| s_il_vous_plait | 100.0% | 32/32 | 5 |

### Paires les plus confondues (vrai → prédit)

| Vrai | Prédit | Erreurs | % de la classe |
|------|--------|---------|----------------|
| aimer | merci | 26 | 51% |
| comprendre | pourquoi | 17 | 59% |
| demain | a_bientot | 14 | 45% |
| quoi | comment | 13 | 57% |
| quoi | quand | 10 | 43% |
| demain | merci | 10 | 32% |
| pourquoi | comprendre | 10 | 15% |
| merci | aimer | 9 | 35% |
| au_revoir | a_bientot | 8 | 16% |
| comprendre | au_revoir | 7 | 24% |
| quand | quoi | 6 | 46% |
| etre | aujourd_hui | 5 | 56% |
| quand | pardon | 5 | 38% |
| aimer | bonjour | 5 | 10% |
| manger | quoi | 5 | 9% |

## Itération — 3A - regularisation+ (dropout=0.5, wd=3e-4) — ❌ NON RETENU (74.18% < 74.95%)

- Protocole : GroupKFold k=5 par contributeur, 60 epochs/fold, seed=42.
- Hyperparams : lr=0.001, weight_decay=0.0003, label_smoothing=0.1, dropout=0.5, min_contrib_keep=3.
- **Accuracy globale poolée (37 classes) : 74.18%**
- **Accuracy sous-ensemble évaluable (>= 3 contrib) : 74.18%**
- Moyenne best_val_acc/fold : 74.17% (+/- 12.72)

### Rappel par classe (trié croissant) — `n_contrib` = nb contributeurs

| Signe | Rappel | Correct/Total | n_contrib |
|-------|--------|---------------|-----------|
| demain | 0.0% | 0/31 | 3 |
| etre | 0.0% | 0/9 | 3 |
| quoi | 0.0% | 0/23 | 3 |
| quand | 7.7% | 1/13 | 5 |
| comprendre | 10.3% | 3/29 | 4 |
| merci | 23.1% | 6/26 | 3 |
| aimer | 33.3% | 17/51 | 6 |
| au_revoir | 44.0% | 22/50 | 6 |
| toi | 74.0% | 37/50 | 5 |
| manger | 77.8% | 42/54 | 4 |
| pourquoi | 80.9% | 55/68 | 5 |
| bonjour | 81.8% | 36/44 | 11 |
| aide | 83.1% | 49/59 | 5 |
| a_bientot | 87.3% | 55/63 | 8 |
| bonne_nuit | 89.1% | 41/46 | 5 |
| pardon | 91.9% | 34/37 | 4 |
| avoir | 93.0% | 40/43 | 4 |
| aller | 96.0% | 48/50 | 4 |
| aujourd_hui | 96.0% | 48/50 | 4 |
| soif | 96.0% | 48/50 | 4 |
| ecole | 96.1% | 49/51 | 8 |
| s_il_vous_plait | 96.9% | 31/32 | 5 |
| comment | 98.0% | 49/50 | 4 |
| boire | 100.0% | 50/50 | 4 |
| dormir | 100.0% | 9/9 | 3 |

### Paires les plus confondues (vrai → prédit)

| Vrai | Prédit | Erreurs | % de la classe |
|------|--------|---------|----------------|
| aimer | merci | 23 | 45% |
| quoi | quand | 18 | 78% |
| demain | a_bientot | 13 | 42% |
| pourquoi | comprendre | 12 | 18% |
| comprendre | pourquoi | 11 | 38% |
| demain | merci | 10 | 32% |
| au_revoir | a_bientot | 10 | 20% |
| au_revoir | pourquoi | 10 | 20% |
| comprendre | a_bientot | 9 | 31% |
| demain | s_il_vous_plait | 7 | 23% |
| toi | aimer | 7 | 14% |
| merci | aimer | 6 | 23% |
| manger | soif | 6 | 11% |
| etre | aujourd_hui | 5 | 56% |
| quand | quoi | 5 | 38% |

## Itération — 3B - pooling avg+max concat — ❌ NON RETENU (74.47% < 74.95%)

- Protocole : GroupKFold k=5 par contributeur, 60 epochs/fold, seed=42.
- Hyperparams : lr=0.001, weight_decay=0.0001, label_smoothing=0.1, dropout=0.4, min_contrib_keep=3.
- **Accuracy globale poolée (37 classes) : 74.47%**
- **Accuracy sous-ensemble évaluable (>= 3 contrib) : 74.47%**
- Moyenne best_val_acc/fold : 74.46% (+/- 12.71)

### Rappel par classe (trié croissant) — `n_contrib` = nb contributeurs

| Signe | Rappel | Correct/Total | n_contrib |
|-------|--------|---------------|-----------|
| quoi | 0.0% | 0/23 | 3 |
| merci | 3.8% | 1/26 | 3 |
| quand | 7.7% | 1/13 | 5 |
| demain | 9.7% | 3/31 | 3 |
| etre | 11.1% | 1/9 | 3 |
| comprendre | 13.8% | 4/29 | 4 |
| aimer | 33.3% | 17/51 | 6 |
| au_revoir | 46.0% | 23/50 | 6 |
| toi | 72.0% | 36/50 | 5 |
| manger | 77.8% | 42/54 | 4 |
| pourquoi | 80.9% | 55/68 | 5 |
| aide | 83.1% | 49/59 | 5 |
| bonjour | 86.4% | 38/44 | 11 |
| a_bientot | 88.9% | 56/63 | 8 |
| bonne_nuit | 89.1% | 41/46 | 5 |
| comment | 90.0% | 45/50 | 4 |
| pardon | 91.9% | 34/37 | 4 |
| avoir | 93.0% | 40/43 | 4 |
| aller | 98.0% | 49/50 | 4 |
| aujourd_hui | 98.0% | 49/50 | 4 |
| boire | 98.0% | 49/50 | 4 |
| soif | 98.0% | 49/50 | 4 |
| ecole | 98.0% | 50/51 | 8 |
| dormir | 100.0% | 9/9 | 3 |
| s_il_vous_plait | 100.0% | 32/32 | 5 |

### Paires les plus confondues (vrai → prédit)

| Vrai | Prédit | Erreurs | % de la classe |
|------|--------|---------|----------------|
| aimer | merci | 26 | 51% |
| comprendre | pourquoi | 15 | 52% |
| quoi | quand | 14 | 61% |
| demain | a_bientot | 14 | 45% |
| au_revoir | comprendre | 12 | 24% |
| merci | aimer | 10 | 38% |
| au_revoir | a_bientot | 10 | 20% |
| pourquoi | comprendre | 9 | 13% |
| quoi | comment | 8 | 35% |
| comprendre | au_revoir | 7 | 24% |
| toi | aimer | 7 | 14% |
| demain | merci | 6 | 19% |
| manger | soif | 6 | 11% |
| etre | aujourd_hui | 5 | 56% |
| merci | a_bientot | 5 | 19% |

---

## Synthèse Partie A — plateau atteint

| Itération | Évaluable | Décision |
|---|---|---|
| 0bis baseline (37 cls) | 68.50% | référence |
| **1 — normalisation corps-relative** | **75.24%** | ✅ **GARDÉ** (seul gain net) |
| 2 — vélocité | 74.08% | ❌ reverté |
| V1 baseline 25 cls (axe 1) | 74.95% | référence V1 |
| 3A — régularisation+ | 74.18% | ❌ non retenu |
| 3B — pooling avg+max | 74.47% | ❌ non retenu |

**Modèle retenu : axe 1 seul (normalisation géométrique corps-relative).**
Meilleur V1 25 classes = **74.95%** (GroupKFold k=5 par contributeur).

### Self-refine — 3 points faibles restants
1. **Plafond de diversité (data, non-modèle)** : variance inter-fold ±12%. Le
   cluster `comprendre`/`demain`/`quoi` (3-4 contributeurs) reste mal séparé.
   Levier réel = plus de signeurs, pas plus de modèle.
2. **Confusions sémantiques** près du visage (`comprendre↔pourquoi`,
   `demain↔a_bientot/merci`, `aimer↔merci`) — gestes visuellement proches en
   keypoints. Piste future : meilleure exploitation des landmarks faciaux.
3. **3 leviers modèle testés sans gain** (vélocité, régularisation, pooling) →
   l'architecture 1D-CNN actuelle est adéquate pour ce volume ; le gain
   supplémentaire viendra des données, pas du tuning.

# Prompt — Amélioration du modèle LSF (précision + capture cadencée)

> À me renvoyer tel quel pour lancer le travail. Conçu pour que tu **boucles**
> sur des axes d'amélioration mesurables jusqu'à plateau, et non pour un
> one-shot.

---

<role>
Tu es ingénieur ML/vision sur ce repo (reconnaissance LSF temps réel à partir de
keypoints MediaPipe, 1D-CNN PyTorch CPU). Tu travailles de façon agentique :
tu explores, tu mesures, tu décides, tu itères. Tu as le droit — et le devoir —
d'exprimer ton incertitude plutôt que de deviner : si une hypothèse n'est pas
validée par un chiffre, dis-le.
</role>

<verites_terrain>
CLAUDE.md est partiellement périmé. Prends CECI comme source de vérité :

- Dataset RÉEL : 37 signes, 1198 samples, 133 contributeurs.
  Emplacement : `src/data/approved/<slug>/*.parquet` (+ `.meta.json`).
  Le loader attend `<dataset_dir>/approved/` → lancer avec `--dataset src/data`.
- Environnement PRÊT : venv `.venv` en Python 3.11, toutes les deps installées
  et vérifiées (torch 2.12 CPU, mediapipe 0.10.9, opencv, pandas, pyarrow,
  scikit-learn, anthropic). Utiliser `.\.venv\Scripts\python.exe`. CPU only.
- Format : 64 frames, vecteur 318/frame = pose(132)+left(63)+right(63)+face(60).
  Mains absentes = zéros. Le preprocessing produit (64, 320) avec 2 flags présence.
- `models/label_map.json` committé est PÉRIMÉ (5 signes) → à régénérer (37).
- Classes faibles (décider : garder, augmenter agressivement, ou exclure — et
  justifier par un chiffre) : excuse_moi=2, travail=4, salut=5, maintenant=6,
  dormir=9, etre=9, vouloir=10, nous/vous/faim/maison/savoir=11-12.
</verites_terrain>

<problemes_connus_a_verifier>
Hypothèses déjà repérées dans le code — à confirmer/infirmer par la mesure, pas à
appliquer aveuglément :
1. `augment_mirror` (preprocessing.py) serait un no-op : `detect_dominant_hand`
   re-trie les mains par mouvement APRÈS le miroir, donc la variante miroir
   produirait des features identiques à l'original → augmentation inutile.
2. `build_feature_vector` z-score TOUTES les features par séquence, y compris les
   2 flags binaires (perte du sens 0/1) et les coordonnées (perte de la position
   absolue corps/mains).
3. Aucune normalisation géométrique invariante (translation/échelle) : les
   keypoints ne sont pas recentrés/redimensionnés sur un repère corporel.
</problemes_connus_a_verifier>

<contraintes>
- CPU only, pas de CUDA. Ne JAMAIS modifier les données de `src/data/approved/`.
- Split train/val PAR CONTRIBUTEUR (jamais aléatoire) — sinon les métriques
  fuitent. Fixer la seed pour comparer les itérations entre elles.
- Code lisible et commenté en français, type hints, constantes nommées en tête,
  logging standard. Pas de sur-ingénierie.
- Cohérence train ↔ inférence : tout changement de preprocessing s'applique des
  DEUX côtés, sinon la précision réelle s'effondre.
- Implémente les changements, ne te contente pas de les proposer. Mais avant
  chaque gros changement, expose un plan court et attends validation.
</contraintes>

<objectif_A_precision>
Maximiser l'accuracy de validation (split par contributeur) ET la précision par
classe sur les 37 signes, en réduisant les confusions inter-signes.

Étape 0 — BASELINE honnête, AVANT toute modif :
- Entraîner l'état actuel sur les 37 signes, logger : accuracy globale,
  accuracy par classe, et la MATRICE DE CONFUSION.
- Lister les 5–10 paires de signes les plus confondues. C'est ta carte.
</objectif_A_precision>

<objectif_B_capture_cadencee>
Remplacer (PAS ajouter) la détection continue « signe / pas de signe » (gate
mouvement + dédup N + cooldown) par une CAPTURE CADENCÉE type métronome qui
synchronise l'utilisateur sur la fenêtre attendue par le modèle.

Comportement cible :
- Boucle rythmée : « préparez-vous » (décompte 3-2-1) → « SIGNEZ » pendant
  laquelle on capture exactement SEQUENCE_LENGTH frames (64 ≈ 2s @30fps) →
  1 inférence → 1 résultat → courte pause → battement suivant.
- UNE prédiction par battement. Si confiance < seuil → afficher « non reconnu »
  et ne rien ajouter à la phrase. Sinon, ajouter le signe.
- Même preprocessing qu'à l'entraînement sur la fenêtre capturée (trim repos +
  `build_feature_vector`).
- UI : phase courante claire + décompte visuel du battement.
- Intervalle et durée de capture configurables (constantes nommées + arg CLI).
- Conserver les touches S (phrase NLP) et C (effacer). L'ancien mode continu est
  retiré (ou laissé non câblé — préciser lequel et pourquoi).
</objectif_B_capture_cadencee>

<boucle_amelioration>
Travaille en boucle, façon générer→classer→tester→garder/annuler, jusqu'à
plateau. NE T'ARRÊTE PAS au premier gain.

1. GÉNÉRER (couverture, pas filtrage) : à partir de la matrice de confusion et du
   code, liste TOUS les axes d'amélioration plausibles, même incertains. Pour
   chacun : hypothèse, coût d'implémentation (S/M/L), gain attendu, risque.
   Exemples d'axes possibles (non exhaustif, ne pas s'y limiter) :
   - normalisation géométrique (recentrage milieu épaules + échelle largeur
     d'épaules) avant z-score ;
   - traitement séparé des flags de présence (ne pas les z-scorer) ;
   - correction/suppression de l'augmentation miroir ; augmentation ciblée des
     classes confondues ou rares ;
   - architecture (profondeur, kernels, dropout, attention temporelle légère,
     pooling) ; hyperparams (LR, epochs, early stopping, weight decay,
     label smoothing, class weights) ;
   - exploitation pose/face actuellement sous-utilisées ; vélocités/accélérations
     en features dérivées ; gestion de la longueur de fenêtre.
2. CLASSER : trie les axes par (gain attendu / coût). Annonce le prochain testé.
3. TESTER UN SEUL axe à la fois, mesure sur la val (même split, même seed),
   compare à la meilleure config courante.
4. GARDER si l'amélioration est nette et non bruitée ; sinon ANNULER et noter
   pourquoi. Un changement « élégant » mais sans gain mesuré ne reste pas.
5. JOURNALISER chaque itération dans `IMPROVE_LOG.md` : axe, hypothèse, chiffres
   avant/après (accuracy globale + classes ciblées), décision (gardé/annulé),
   apprentissage.
6. RÉPÉTER jusqu'à : soit 2–3 itérations consécutives sans gain net, soit budget
   atteint. Puis applique une passe self-refine : « quels sont les 3 points
   faibles du modèle/pipeline actuel ? » et traite-les si rentable.
</boucle_amelioration>

<criteres_succes>
- Baseline chiffrée produite et committée dans le journal AVANT les modifs.
- Chaque changement conservé est justifié par un delta de métrique sur la val
  par contributeur (pas par intuition ni élégance).
- Meilleur modèle sauvegardé + `models/label_map.json` régénéré (37 classes).
- Mode capture cadencée fonctionnel, même preprocessing qu'à l'entraînement.
- `IMPROVE_LOG.md` retrace la boucle (axes testés, gardés, annulés, pourquoi).
</criteres_succes>

<methode>
- Parallélise les lectures/inspections indépendantes.
- Avant un gros refactor : plan court (quoi, pourquoi, quels fichiers) → validation.
- Petits incréments mesurables. Si un chiffre te surprend, creuse-le avant de
  continuer (un gain « trop beau » cache souvent une fuite de split).
</methode>

Commence par l'Étape 0 (baseline chiffrée + matrice de confusion), puis présente
ta liste d'axes classés avant de toucher au modèle.

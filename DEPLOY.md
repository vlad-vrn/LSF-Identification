# Déploiement LSF Direct sur Coolify

Application web de reconnaissance LSF : la webcam + MediaPipe tournent dans le
**navigateur**, un backend **FastAPI** (Docker) fait l'inférence du modèle et la
génération de phrase (NLP Nous). Image Docker légère (torch CPU, pas de
mediapipe/opencv côté serveur).

```
Navigateur (webcam + MediaPipe JS + cadence)
        │  POST /api/predict  (64×318 keypoints)
        │  POST /api/sentence (liste de signes)
        ▼
FastAPI (Docker, Coolify)  →  lsf_v1.pt + build_feature_vector + NLP Nous
```

---

## 0. Pré-requis

- Une instance **Coolify** opérationnelle (avec son proxy Traefik/Caddy).
- Un **nom de domaine** (ou sous-domaine) pointant vers le serveur Coolify.
  ⚠️ **HTTPS obligatoire** : l'accès webcam (`getUserMedia`) ne fonctionne que sur
  une origine sécurisée (https) ou `localhost`. Coolify fournit Let's Encrypt.
- Un dépôt Git (GitHub/GitLab) accessible par Coolify.
- Ta clé **nouslabs** (pour le NLP) — elle sera mise en variable d'env Coolify,
  **jamais** committée.

---

## 1. Inclure le modèle dans le dépôt

Coolify build depuis Git : le modèle doit être présent dans le dépôt. La
`.gitignore` est configurée pour ignorer tous les `.pt` **sauf** le modèle de
déploiement `models/lsf_v1.pt` (règle `!models/lsf_v1.pt`). Il suffit donc de :

```bash
git add models/lsf_v1.pt models/label_map.json
git commit -m "Ajout du modèle V1 (24 classes) pour le déploiement"
```

> `lsf_v1.pt` fait ~1 Mo : l'inclure dans le dépôt est acceptable.
> Alternative : utiliser un **volume persistant** Coolify monté sur `/app/models`
> et y déposer le `.pt` (puis retirer le `COPY models/...` du Dockerfile).

Pousse le tout :

```bash
git add Dockerfile .dockerignore requirements-server.txt webapp/ src/ DEPLOY.md
git commit -m "App web LSF + Docker"
git push
```

---

## 2. Créer l'application dans Coolify

1. **Projects → ton projet → + New → Application**.
2. Source : **Public/Private Repository** → sélectionne ton dépôt + branche (`main`).
3. **Build Pack : Dockerfile** (Coolify détecte le `Dockerfile` à la racine).
4. **Port exposé : `8000`** (le `EXPOSE 8000` / `--port 8000` du conteneur).
   Renseigne « Ports Exposes » = `8000` dans les réglages réseau si demandé.

---

## 3. Variables d'environnement (NLP nouslabs)

Dans **l'onglet Environment Variables** de l'application, ajoute (en **secret**
pour la clé) :

| Clé | Valeur |
|-----|--------|
| `LSF_NLP_PROVIDER` | `openai` |
| `LSF_NLP_BASE_URL` | `https://inference-api.nousresearch.com/v1` |
| `LSF_NLP_MODEL`    | `nousresearch/hermes-4-70b` |
| `LSF_NLP_API_KEY`  | `‹ta clé nouslabs›` (Secret) |

> Le serveur lit ces variables au runtime (priorité à l'environnement sur le
> `.env`). La clé reste côté serveur, jamais envoyée au navigateur.

---

## 4. Domaine + HTTPS

1. Dans **Domains**, mets ton domaine : `https://lsf.tondomaine.fr`.
2. Active **HTTPS / Let's Encrypt** (Coolify gère le certificat).
3. Vérifie que le DNS du domaine pointe bien vers l'IP du serveur Coolify.

> Sans HTTPS, le navigateur refusera l'accès caméra → écran figé sur « Démarrer ».

---

## 5. Déployer

1. Clique **Deploy**. Coolify build l'image (premier build ~quelques minutes :
   téléchargement de torch CPU) puis lance le conteneur.
2. Suis les **Logs** : tu dois voir
   `Modèle chargé — 24 classes : [...]` puis `Uvicorn running on http://0.0.0.0:8000`.

---

## 6. Vérifier

- Ouvre `https://lsf.tondomaine.fr/api/health` → `{"status":"ok","n_classes":24,...}`.
- Ouvre `https://lsf.tondomaine.fr/` → l'interface ; autorise la caméra.
- **Démarrer la caméra** → décompte 3-2-1 → **SIGNEZ** → résultat → les signes
  s'ajoutent en bas.
- **Générer la phrase** : met la détection en pause et appelle Nous ; **Reprendre**
  pour relancer.

---

## 7. Mettre à jour

- Réentraîner le modèle ? Régénère `models/lsf_v1.pt`, `git add -f`, commit, push,
  puis **Redeploy** dans Coolify.
- Modifier l'UI (`webapp/static/*`) : commit + push + Redeploy.
  (Coolify peut aussi auto-déployer sur push si tu actives le webhook Git.)

---

## 8. Test local (optionnel, avant de pousser)

```powershell
# Backend (sert aussi l'UI sur http://localhost:8000)
.\.venv\Scripts\python.exe -m uvicorn webapp.server:app --host 0.0.0.0 --port 8000
```
Puis ouvre `http://localhost:8000` (localhost = origine sécurisée, la webcam marche).

Build/run Docker en local :
```bash
docker build -t lsf-web .
docker run --rm -p 8000:8000 \
  -e LSF_NLP_PROVIDER=openai \
  -e LSF_NLP_BASE_URL=https://inference-api.nousresearch.com/v1 \
  -e LSF_NLP_MODEL=nousresearch/hermes-4-70b \
  -e LSF_NLP_API_KEY=‹ta_cle› \
  lsf-web
```

---

## Dépannage

| Symptôme | Cause probable | Solution |
|----------|----------------|----------|
| Écran figé sur « Démarrer » | pas de HTTPS | active Let's Encrypt / sert en https |
| `Backend injoignable` (pastille rouge) | port mal mappé | vérifie port `8000` côté Coolify |
| Build échoue `models/lsf_v1.pt not found` | modèle absent du repo | `git add -f models/lsf_v1.pt` |
| Phrase = signes bruts au lieu d'une phrase | clé/URL/modèle NLP KO | vérifie les 4 variables d'env nouslabs |
| Prédictions incohérentes | extraction navigateur ≠ entraînement | normal en conditions réelles ; baisse le seuil |
| Image Docker trop longue à build | torch télécharge | normal au 1er build, mis en cache ensuite |

"""
Reconstruction de phrase française à partir de signes LSF accumulés.

Deux backends possibles pour transformer la glose LSF en phrase :

1. **anthropic**  — API Claude (Anthropic), via le package `anthropic`.
2. **openai**     — n'importe quel provider compatible OpenAI (format
                    /chat/completions), dont **nouslabs / Nous Research**.
                    Implémenté avec la lib standard (urllib) — aucune dépendance.

Le provider est choisi automatiquement :
  - si une base_url OpenAI-compatible est fournie (arg ou env) → backend openai ;
  - sinon, si une clé Anthropic est dispo → backend anthropic ;
  - sinon → fallback texte brut (signes concaténés, hors-ligne).

Configuration possible par variables d'environnement :
    LSF_NLP_PROVIDER   = anthropic | openai
    LSF_NLP_BASE_URL   = https://.../v1        (provider OpenAI-compatible)
    LSF_NLP_MODEL      = <id du modèle>
    LSF_NLP_API_KEY    = <clé>                  (sinon ANTHROPIC_API_KEY)
"""

import json
import logging
import os
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

# Modèle par défaut du backend Anthropic.
_MODEL_ANTHROPIC = "claude-haiku-4-5-20251001"

# Valeurs par défaut du backend OpenAI-compatible (surchargées par arg/env).
# La base_url et le modèle dépendent de TON compte nouslabs : récupère-les dans
# ton dashboard et passe-les via --nlp-base-url / --nlp-model (ou les env vars).
_DEFAULT_OPENAI_BASE_URL = "https://inference-api.nousresearch.com/v1"
# IDs du catalogue Nous Portal préfixés par provider. hermes-4-70b = le moins cher
# des modèles natifs Nous. Mets l'ID exact de TON catalogue via .env / --nlp-model.
_DEFAULT_OPENAI_MODEL = "nousresearch/hermes-4-70b"

_SYSTEM_PROMPT = (
    "Tu es un assistant de traduction de glose LSF (Langue des Signes Française) "
    "vers le français oral.\n"
    "On te donne une liste de signes LSF dans l'ordre où ils ont été effectués.\n"
    "Reconstitue une phrase française naturelle et grammaticalement correcte.\n\n"
    "Règles :\n"
    "- Respecte l'ordre et le sens des signes\n"
    "- Ajoute les mots grammaticaux nécessaires (conjugaisons, articles, prépositions)\n"
    "- Phrase courte et naturelle, avec une majuscule initiale et un point final\n"
    "- Réponds UNIQUEMENT avec la phrase, sans guillemets ni explication"
)


def _slugs_to_gloss(signs: list[str]) -> str:
    """
    Convertit les slugs de signes en glose lisible pour le modèle.
    Ex: ["s_il_vous_plait", "boire", "eau"] → "s il vous plait / boire / eau"
    """
    return " / ".join(s.replace("_", " ") for s in signs)


def build_sentence(
    signs: list[str],
    api_key: str | None = None,
    provider: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
) -> str:
    """
    Reconstruit une phrase française à partir d'une liste de signes LSF.

    Args:
        signs:    Slugs des signes confirmés dans l'ordre (ex: ["je", "vouloir", "boire"]).
        api_key:  Clé API. Si None, lue depuis LSF_NLP_API_KEY puis ANTHROPIC_API_KEY.
        provider: "anthropic" | "openai" | None (auto). None → LSF_NLP_PROVIDER puis auto.
        base_url: URL de base OpenAI-compatible (provider openai). None → LSF_NLP_BASE_URL.
        model:    ID du modèle. None → LSF_NLP_MODEL puis défaut du backend.

    Returns:
        Phrase française reconstruite, ou les signes joints en texte brut si
        aucun backend n'est configuré / en cas d'erreur réseau.
    """
    if not signs:
        return ""

    gloss = _slugs_to_gloss(signs)
    logger.info("Construction de phrase — glose : %s", gloss)

    # --- Résolution de la configuration (args > env) ---
    provider = provider or os.environ.get("LSF_NLP_PROVIDER")
    base_url = base_url or os.environ.get("LSF_NLP_BASE_URL")
    model = model or os.environ.get("LSF_NLP_MODEL")
    key = api_key or os.environ.get("LSF_NLP_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")

    # Choix automatique du backend si non imposé
    if provider is None:
        provider = "openai" if base_url else "anthropic"

    if not key:
        logger.warning("Aucune clé API configurée — fallback texte brut")
        return gloss.replace(" / ", " ")

    try:
        if provider == "openai":
            return _build_openai_compatible(
                gloss,
                api_key=key,
                base_url=base_url or _DEFAULT_OPENAI_BASE_URL,
                model=model or _DEFAULT_OPENAI_MODEL,
            )
        return _build_anthropic(gloss, api_key=key, model=model or _MODEL_ANTHROPIC)
    except Exception as exc:
        logger.error("Erreur backend NLP '%s' : %s", provider, exc)
        return gloss.replace(" / ", " ")


def _build_anthropic(gloss: str, api_key: str, model: str) -> str:
    """Backend Anthropic (package `anthropic`)."""
    from anthropic import Anthropic

    client = Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=120,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": gloss}],
    )
    sentence = response.content[0].text.strip()
    logger.info("Phrase construite (anthropic) : %s", sentence)
    return sentence


def _build_openai_compatible(gloss: str, api_key: str, base_url: str, model: str) -> str:
    """
    Backend compatible OpenAI (/chat/completions), pour nouslabs / Nous Research
    ou tout provider exposant le format OpenAI. Utilise urllib (aucune dépendance).
    """
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": gloss},
        ],
        "max_tokens": 120,
        "temperature": 0.3,
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        # Le corps de la réponse contient le vrai message (modèle inconnu, auth, ...)
        body = exc.read().decode("utf-8", "replace")[:600]
        raise RuntimeError(f"HTTP {exc.code} {exc.reason} sur {url} — {body}") from None

    sentence = data["choices"][0]["message"]["content"].strip()
    logger.info("Phrase construite (openai/%s) : %s", model, sentence)
    return sentence

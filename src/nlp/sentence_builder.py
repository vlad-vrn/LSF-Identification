"""
Reconstruction de phrase française à partir de signes LSF accumulés.

Envoie la glose LSF (liste de signes) à Claude Haiku via l'API Anthropic
et retourne une phrase française grammaticalement correcte.

Dépendance : package `anthropic` + variable d'environnement ANTHROPIC_API_KEY.
En l'absence de clé, retourne les signes joints en texte brut (fallback offline).
"""

import logging
import os

logger = logging.getLogger(__name__)

_MODEL = "claude-haiku-4-5-20251001"

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
    On laisse Claude interpréter — il reconnaît "s il vous plait" → "s'il vous plaît".
    """
    return " / ".join(s.replace("_", " ") for s in signs)


def build_sentence(
    signs: list[str],
    api_key: str | None = None,
) -> str:
    """
    Reconstruit une phrase française à partir d'une liste de signes LSF.

    Args:
        signs:   Slugs des signes confirmés dans l'ordre (ex: ["je", "vouloir", "boire"]).
        api_key: Clé API Anthropic. Si None, utilise ANTHROPIC_API_KEY (env var).

    Returns:
        Phrase française reconstruite.
        Retourne les signes joints en texte brut si l'API est indisponible.
    """
    if not signs:
        return ""

    gloss = _slugs_to_gloss(signs)
    logger.info("Construction de phrase — glose : %s", gloss)

    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        logger.warning("ANTHROPIC_API_KEY non configurée — fallback texte brut")
        return gloss.replace(" / ", " ")

    try:
        from anthropic import Anthropic

        client = Anthropic(api_key=key)
        response = client.messages.create(
            model=_MODEL,
            max_tokens=120,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": gloss}],
        )
        sentence = response.content[0].text.strip()
        logger.info("Phrase construite : %s", sentence)
        return sentence

    except ImportError:
        logger.warning("Package 'anthropic' non installé — pip install anthropic")
        return gloss.replace(" / ", " ")
    except Exception as exc:
        logger.error("Erreur API Anthropic : %s", exc)
        return gloss.replace(" / ", " ")

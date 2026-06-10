"""Pré-processamento compartilhado entre o notebook clássico, o treino do baseline e a interface.

Mantém exatamente o mesmo pipeline do notebooks/poc_fake_news.ipynb:
lowercase -> remove URL/email/número/pontuação -> lematização spaCy -> remove stopwords.
"""

import re
import string

import nltk
import spacy
from nltk.corpus import stopwords

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

_nlp = spacy.load("pt_core_news_sm", disable=["parser", "ner"])

STOPWORDS_PT = set(stopwords.words("portuguese"))

URL_RE = re.compile(r"http\S+|www\.\S+")
EMAIL_RE = re.compile(r"\S+@\S+")
NUMERO_RE = re.compile(r"\d+")


def limpar(texto: str) -> str:
    texto = texto.lower()
    texto = URL_RE.sub(" ", texto)
    texto = EMAIL_RE.sub(" ", texto)
    texto = NUMERO_RE.sub(" ", texto)
    texto = texto.translate(str.maketrans("", "", string.punctuation))
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def preprocessar(texto: str) -> str:
    texto = limpar(texto)
    doc = _nlp(texto)
    tokens = [
        token.lemma_
        for token in doc
        if token.lemma_ not in STOPWORDS_PT
        and len(token.lemma_) > 2
        and not token.is_space
        and token.is_alpha
    ]
    return " ".join(tokens)

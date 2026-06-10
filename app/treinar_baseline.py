"""Treina o baseline (LogReg + TF-IDF) e salva os artefatos para a interface.

Reproduz o notebook clássico: mesmo carregamento, mesmo shuffle (random_state=42),
mesmo split 80/20 estratificado e mesma vetorização. Rodar uma vez antes do app:

    python app/treinar_baseline.py

A célula de pré-processamento é o gargalo (~5 a 10 min). O resultado vai para
models/baseline.joblib (vetorizador + modelo + métricas no holdout).
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from preprocessamento import preprocessar

RANDOM_STATE = 42
RAIZ = Path(__file__).resolve().parent.parent
DATA_DIR = RAIZ / "data" / "Fake.br-Corpus" / "full_texts"
MODELS_DIR = RAIZ / "models"


def carregar_textos(pasta: Path, label: int) -> list[dict]:
    registros = []
    for arquivo in sorted(pasta.glob("*.txt")):
        texto = arquivo.read_text(encoding="utf-8").strip()
        registros.append({"texto": texto, "label": label, "arquivo": arquivo.name})
    return registros


def main() -> None:
    fake = carregar_textos(DATA_DIR / "fake", label=1)
    true = carregar_textos(DATA_DIR / "true", label=0)
    df = (
        pd.DataFrame(fake + true)
        .sample(frac=1, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )
    print(f"Total de notícias: {len(df)}")

    print("Pré-processando (5 a 10 min)...")
    df["texto_proc"] = df["texto"].apply(preprocessar)

    X_train, X_test, y_train, y_test = train_test_split(
        df["texto_proc"], df["label"],
        test_size=0.2, stratify=df["label"], random_state=RANDOM_STATE,
    )

    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)

    modelo = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    modelo.fit(X_train_vec, y_train)

    pred = modelo.predict(X_test_vec)
    metricas = {
        "accuracy": accuracy_score(y_test, pred),
        "f1": f1_score(y_test, pred),
    }
    print(f"Holdout: accuracy={metricas['accuracy']:.4f} f1={metricas['f1']:.4f}")

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(
        {"vetorizador": tfidf, "modelo": modelo, "metricas": metricas},
        MODELS_DIR / "baseline.joblib",
    )
    print(f"Artefatos salvos em {MODELS_DIR / 'baseline.joblib'}")


if __name__ == "__main__":
    main()

"""Teste rápido: os dois modelos sobre notícias completas reais do conjunto de teste."""
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
from app import prever_baseline, prever_bertimbau  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data" / "Fake.br-Corpus" / "full_texts"


def carregar(p, l):
    return [{"texto": a.read_text(encoding="utf-8").strip(), "label": l} for a in sorted(p.glob("*.txt"))]


df = pd.DataFrame(carregar(DATA / "fake", 1) + carregar(DATA / "true", 0)).sample(
    frac=1, random_state=42
).reset_index(drop=True)
Xtr, Xte, ytr, yte = train_test_split(
    df["texto"], df["label"], test_size=0.2, stratify=df["label"], random_state=42
)
teste = pd.DataFrame({"texto": Xte, "label": yte})
para_testar = pd.concat([teste[teste.label == 1].head(3), teste[teste.label == 0].head(3)])

acertos = {"classico": 0, "bertimbau": 0}
for _, r in para_testar.iterrows():
    real = "FAKE" if r["label"] == 1 else "VERDADEIRA"
    rb = prever_baseline(r["texto"])
    rt = prever_bertimbau(r["texto"])
    acertos["classico"] += int(rb.startswith(real))
    acertos["bertimbau"] += int(rt.startswith(real))
    print(f"[real: {real}] {r['texto'][:70]}...")
    print(f"  classico : {rb}")
    print(f"  bertimbau: {rt}")

print(f"\nAcertos em 6 noticias reais: classico {acertos['classico']}/6, bertimbau {acertos['bertimbau']}/6")

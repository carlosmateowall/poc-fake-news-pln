"""Avaliação local do LLM generativo (mesma lógica do notebook, sem plots).

Requer ANTHROPIC_API_KEY no ambiente. Salva models/llm_metricas.json e
models/llm_resultados.csv (com as justificativas, úteis na apresentação).

Uso:  .venv\\Scripts\\python.exe scripts\\avaliar_llm.py
"""

import json
import os
import time
from pathlib import Path

import anthropic
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
LLM_MODELO = os.environ.get("LLM_MODELO", "claude-opus-4-8")
N_AMOSTRA = 200
RAIZ = Path(__file__).resolve().parent.parent
DATA_DIR = RAIZ / "data" / "Fake.br-Corpus" / "full_texts"

PROMPT = """Você é um verificador de fatos especializado em notícias brasileiras.
Classifique a notícia abaixo como fake news ou verdadeira. Considere sinais como:
alarmismo e urgência artificial, apelo ao compartilhamento, ausência de fontes
verificáveis, promessas implausíveis, erros de ortografia e estilo não editorial.
Importante: avalie o estilo e a estrutura, não a sua opinião sobre o tema.

<noticia>
{texto}
</noticia>"""

SCHEMA = {
    "type": "object",
    "properties": {
        "classificacao": {"type": "string", "enum": ["fake", "verdadeira"]},
        "justificativa": {"type": "string"},
    },
    "required": ["classificacao", "justificativa"],
    "additionalProperties": False,
}


def carregar_textos(pasta: Path, label: int) -> list[dict]:
    return [
        {"texto": a.read_text(encoding="utf-8").strip(), "label": label}
        for a in sorted(pasta.glob("*.txt"))
    ]


def main() -> None:
    assert os.environ.get("ANTHROPIC_API_KEY"), "Defina ANTHROPIC_API_KEY antes de rodar"
    client = anthropic.Anthropic()

    fake = carregar_textos(DATA_DIR / "fake", label=1)
    true = carregar_textos(DATA_DIR / "true", label=0)
    df = pd.DataFrame(fake + true).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    X_train, X_test, y_train, y_test = train_test_split(
        df["texto"], df["label"],
        test_size=0.2, stratify=df["label"], random_state=RANDOM_STATE,
    )
    teste = pd.DataFrame({"texto": X_test, "label": y_test})
    amostra = (
        teste.groupby("label", group_keys=False)
        .apply(lambda g: g.sample(n=N_AMOSTRA // 2, random_state=RANDOM_STATE))
        .reset_index(drop=True)
    )
    print(f"Modelo: {LLM_MODELO} | Amostra: {len(amostra)} notícias")

    def classificar(texto: str) -> dict:
        resposta = client.messages.create(
            model=LLM_MODELO,
            max_tokens=512,
            output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
            messages=[{"role": "user", "content": PROMPT.format(texto=texto[:6000])}],
        )
        return json.loads(next(b.text for b in resposta.content if b.type == "text"))

    resultados = []
    for i, linha in amostra.iterrows():
        try:
            r = classificar(linha["texto"])
        except Exception as e:
            print(f"[{i}] erro: {e}, repetindo em 10s")
            time.sleep(10)
            r = classificar(linha["texto"])
        resultados.append({
            "label": int(linha["label"]),
            "pred": 1 if r["classificacao"] == "fake" else 0,
            "justificativa": r["justificativa"],
        })
        if (i + 1) % 20 == 0:
            print(f"{i + 1}/{len(amostra)}")

    res = pd.DataFrame(resultados)
    print(classification_report(res["label"], res["pred"], target_names=["Verdadeira", "Fake"]))

    metricas = {
        "modelo": LLM_MODELO,
        "n_amostra": len(res),
        "accuracy": accuracy_score(res["label"], res["pred"]),
        "precision": precision_score(res["label"], res["pred"]),
        "recall": recall_score(res["label"], res["pred"]),
        "f1": f1_score(res["label"], res["pred"]),
        "matriz_confusao": confusion_matrix(res["label"], res["pred"]).tolist(),
    }
    (RAIZ / "models").mkdir(exist_ok=True)
    (RAIZ / "models" / "llm_metricas.json").write_text(json.dumps(metricas, indent=2), encoding="utf-8")
    res.to_csv(RAIZ / "models" / "llm_resultados.csv", index=False)
    print("Métricas em models/llm_metricas.json, resultados em models/llm_resultados.csv")


if __name__ == "__main__":
    main()

# %% [markdown]
# # POC Fake News, Etapa 3: LLM Generativo como Classificador
#
# Trabalho da disciplina de Processamento de Linguagem Natural, 7º semestre de Ciência de Dados, UniCEUB.
#
# Grupo: Carlos Mateo Wall Bruno, Lucas Wall e Jamile.
#
# Terceira frente da POC: usar um **modelo de linguagem generativo** (Claude, da Anthropic)
# para dizer se uma notícia é verdadeira ou falsa, **sem nenhum treinamento**, só com um
# prompt estruturado (zero-shot). Avaliamos numa **amostra do mesmo conjunto de teste** dos
# outros dois experimentos para a comparação ser justa.
#
# Requisito: chave de API no ambiente (`ANTHROPIC_API_KEY`). No Colab, use o ícone de chave
# (Secrets) e habilite o acesso para este notebook.

# %% [markdown]
# ## 1. Setup

# %%
# !pip -q install anthropic scikit-learn pandas

# %%
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

RANDOM_STATE = 42

# modelo generativo usado na avaliação.
# claude-opus-4-8 é o mais capaz; claude-haiku-4-5 corta o custo (~5x menor por token)
# com pouca perda nessa tarefa. Deixamos configurável.
LLM_MODELO = "claude-opus-4-8"

# tamanho da amostra do teste (1440 notícias completas custariam caro; 200 dá um
# intervalo de confiança razoável para a comparação)
N_AMOSTRA = 200

# no Colab: carrega a chave dos Secrets
try:
    from google.colab import userdata  # type: ignore
    os.environ["ANTHROPIC_API_KEY"] = userdata.get("ANTHROPIC_API_KEY")
except ImportError:
    pass  # rodando local, a variável de ambiente já deve existir

assert os.environ.get("ANTHROPIC_API_KEY"), "Defina ANTHROPIC_API_KEY antes de continuar"

# %% [markdown]
# ## 2. Dataset, mesmo carregamento e mesmo split dos outros notebooks

# %%
# !git clone -q https://github.com/roneysco/Fake.br-Corpus.git

DATA_DIR = Path("Fake.br-Corpus/full_texts")
if not DATA_DIR.exists():
    DATA_DIR = Path("../data/Fake.br-Corpus/full_texts")

def carregar_textos(pasta, label):
    registros = []
    for arquivo in sorted(pasta.glob("*.txt")):
        with open(arquivo, "r", encoding="utf-8") as f:
            texto = f.read().strip()
        registros.append({"texto": texto, "label": label, "arquivo": arquivo.name})
    return registros

fake = carregar_textos(DATA_DIR / "fake", label=1)
true = carregar_textos(DATA_DIR / "true", label=0)
df = pd.DataFrame(fake + true).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

X_train, X_test, y_train, y_test = train_test_split(
    df["texto"], df["label"],
    test_size=0.2, stratify=df["label"], random_state=RANDOM_STATE,
)

# amostra estratificada do TESTE (nunca do treino, para não contaminar a comparação)
teste = pd.DataFrame({"texto": X_test, "label": y_test})
amostra = (
    teste.groupby("label", group_keys=False)
    .apply(lambda g: g.sample(n=N_AMOSTRA // 2, random_state=RANDOM_STATE))
    .reset_index(drop=True)
)
print(f"Amostra: {len(amostra)} notícias ({amostra['label'].value_counts().to_dict()})")

# %% [markdown]
# ## 3. Prompt e chamada estruturada
#
# Usamos **saída estruturada** (JSON Schema) para o modelo responder sempre no mesmo
# formato, sem parsing frágil de texto livre. O texto da notícia é truncado em ~6.000
# caracteres para controlar custo (o essencial está no início).

# %%
import anthropic

client = anthropic.Anthropic()

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

def classificar_com_llm(texto: str) -> dict:
    resposta = client.messages.create(
        model=LLM_MODELO,
        max_tokens=512,
        output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": PROMPT.format(texto=texto[:6000])}],
    )
    return json.loads(next(b.text for b in resposta.content if b.type == "text"))

# teste rápido com 1 notícia
exemplo = classificar_com_llm(amostra["texto"].iloc[0])
print(exemplo["classificacao"])
print(exemplo["justificativa"][:300])

# %% [markdown]
# ## 4. Avaliação na amostra
#
# Loop simples com retry (o SDK já faz retry de erros transitórios; aqui só protegemos o
# loop para não perder o progresso). ~200 chamadas levam alguns minutos.

# %%
resultados = []
for i, linha in amostra.iterrows():
    try:
        r = classificar_com_llm(linha["texto"])
    except Exception as e:
        print(f"[{i}] erro: {e}, tentando de novo em 10s")
        time.sleep(10)
        r = classificar_com_llm(linha["texto"])
    resultados.append({
        "label": linha["label"],
        "pred": 1 if r["classificacao"] == "fake" else 0,
        "justificativa": r["justificativa"],
    })
    if (i + 1) % 20 == 0:
        print(f"{i + 1}/{len(amostra)}")

res = pd.DataFrame(resultados)

# %%
print(classification_report(res["label"], res["pred"], target_names=["Verdadeira", "Fake"]))
print(confusion_matrix(res["label"], res["pred"]))

# %% [markdown]
# ## 5. Análise de erros
#
# A vantagem única do LLM: ele **justifica** cada decisão. Olhar os erros diz muito sobre
# o que cada abordagem captura.

# %%
erros = res[res["label"] != res["pred"]]
print(f"Erros: {len(erros)} de {len(res)}")
for _, e in erros.head(5).iterrows():
    real = "FAKE" if e["label"] == 1 else "VERDADEIRA"
    pred = "FAKE" if e["pred"] == 1 else "VERDADEIRA"
    print(f"\nReal: {real} | Previsto: {pred}")
    print(f"Justificativa: {e['justificativa'][:400]}")

# %% [markdown]
# ## 6. Tabela comparativa final das três abordagens
#
# Atualize os valores do clássico e do BERTimbau com os números exatos das suas execuções.
# Atenção à nota metodológica: clássico e BERTimbau foram avaliados no teste completo
# (1.440 notícias); o LLM, numa amostra estratificada de 200.

# %%
comparacao = pd.DataFrame([
    {"abordagem": "Clássico: LogReg + TF-IDF", "avaliado_em": "teste completo (1.440)",
     "accuracy": 0.96, "f1": 0.96, "custo": "segundos de CPU", "treino": "sim"},
    {"abordagem": "Semântico: BERTimbau fine-tunado", "avaliado_em": "teste completo (1.440)",
     "accuracy": None, "f1": None, "custo": "minutos de GPU", "treino": "sim"},
    {"abordagem": f"Generativo: {LLM_MODELO}", "avaliado_em": f"amostra ({len(res)})",
     "accuracy": accuracy_score(res["label"], res["pred"]),
     "f1": f1_score(res["label"], res["pred"]),
     "custo": "API por chamada", "treino": "não (zero-shot)"},
])
comparacao

# %% [markdown]
# ## 7. Conclusões
#
# - O LLM classifica **sem nenhum exemplo de treino**, o que elimina o problema do corpus
#   datado (2016-2018): ele julga pelo estilo e plausibilidade, não pelo vocabulário do Fake.br.
# - Em contrapartida, custa por chamada, tem latência de segundos e não dá probabilidade
#   calibrada, só uma justificativa textual (que é ótima para explicar a decisão ao usuário).
# - O trio fecha a discussão da POC: clássico (barato, preso ao vocabulário), fine-tune
#   (melhor métrica no domínio, custo de treino) e generativo (generaliza melhor fora do
#   domínio, custo por uso). A escolha certa depende do cenário de produção.

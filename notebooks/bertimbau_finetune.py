# %% [markdown]
# # POC Fake News, Etapa 2: Classificador Semântico (BERTimbau)
#
# Trabalho da disciplina de Processamento de Linguagem Natural, 7º semestre de Ciência de Dados, UniCEUB.
#
# Grupo: Carlos Mateo Wall Bruno, Lucas Wall e Jamile.
#
# Este notebook avança a POC do pipeline clássico (ver `poc_fake_news.ipynb`) para um
# **classificador semântico**: fine-tune do **BERTimbau** (`neuralmind/bert-base-portuguese-cased`)
# no Fake.br Corpus. A conclusão do notebook clássico apontava exatamente esse próximo passo.
#
# **Pensado para rodar no Google Colab com GPU** (Ambiente de execução > Alterar tipo > T4 GPU).
# Treino estimado: 20 a 40 minutos.
#
# Para a comparação ser justa, usamos **o mesmo shuffle e o mesmo split 80/20 estratificado
# com random_state=42** do notebook clássico. A diferença é que o BERT recebe o **texto cru**
# (sem lematização nem remoção de stopwords): o tokenizador e o modelo cuidam do contexto.

# %% [markdown]
# ## 1. Setup (Colab)

# %%
# instala as dependências (no Colab)
# !pip -q install transformers datasets evaluate accelerate scikit-learn

# %%
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
)

import matplotlib.pyplot as plt
import seaborn as sns

RANDOM_STATE = 42
MODELO_BASE = "neuralmind/bert-base-portuguese-cased"  # BERTimbau base

print("GPU disponível:", torch.cuda.is_available())

# %% [markdown]
# ## 2. Dataset (mesmo carregamento do notebook clássico)
#
# No Colab, clonamos o Fake.br Corpus direto do repositório oficial.

# %%
# !git clone -q https://github.com/roneysco/Fake.br-Corpus.git

DATA_DIR = Path("Fake.br-Corpus/full_texts")
# rodando local (fora do Colab), aponte para ../data/Fake.br-Corpus/full_texts
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
print(f"Total de notícias: {len(df)}")
print(df["label"].value_counts())

# %% [markdown]
# Mesmo split do notebook clássico (mesmos índices, então as mesmas notícias caem em treino
# e teste nos dois experimentos). Texto cru, sem pré-processamento.

# %%
X_train, X_test, y_train, y_test = train_test_split(
    df["texto"], df["label"],
    test_size=0.2, stratify=df["label"], random_state=RANDOM_STATE,
)
print(f"Treino: {len(X_train)} | Teste: {len(X_test)}")

# %% [markdown]
# ## 3. Tokenização
#
# `max_length=256`: as notícias são longas, mas os primeiros 256 tokens carregam o essencial
# (título e lide) e o custo de treino cai bastante em relação a 512.

# %%
from transformers import AutoTokenizer
from datasets import Dataset

tokenizer = AutoTokenizer.from_pretrained(MODELO_BASE)

def tokenizar(exemplos):
    return tokenizer(exemplos["texto"], truncation=True, max_length=256)

ds_train = Dataset.from_dict({"texto": X_train.tolist(), "label": y_train.tolist()}).map(
    tokenizar, batched=True, remove_columns=["texto"]
)
ds_test = Dataset.from_dict({"texto": X_test.tolist(), "label": y_test.tolist()}).map(
    tokenizar, batched=True, remove_columns=["texto"]
)

# %% [markdown]
# ## 4. Fine-tune

# %%
from transformers import (
    AutoModelForSequenceClassification, Trainer, TrainingArguments,
    DataCollatorWithPadding,
)

modelo = AutoModelForSequenceClassification.from_pretrained(MODELO_BASE, num_labels=2)

def computar_metricas(eval_pred):
    logits, labels = eval_pred
    pred = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, pred),
        "precision": precision_score(labels, pred),
        "recall": recall_score(labels, pred),
        "f1": f1_score(labels, pred),
    }

args = TrainingArguments(
    output_dir="bertimbau-fakenews-checkpoints",
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    eval_strategy="epoch",
    save_strategy="no",
    logging_steps=50,
    fp16=torch.cuda.is_available(),
    seed=RANDOM_STATE,
    report_to="none",
)

trainer = Trainer(
    model=modelo,
    args=args,
    train_dataset=ds_train,
    eval_dataset=ds_test,
    processing_class=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=computar_metricas,
)

trainer.train()

# %% [markdown]
# ## 5. Avaliação no conjunto de teste

# %%
saida = trainer.predict(ds_test)
logits = saida.predictions
pred = np.argmax(logits, axis=-1)
prob_fake = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()

print(classification_report(y_test, pred, target_names=["Verdadeira", "Fake"]))
print(f"ROC AUC: {roc_auc_score(y_test, prob_fake):.4f}")

# %%
cm = confusion_matrix(y_test, pred)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Verdadeira", "Fake"],
            yticklabels=["Verdadeira", "Fake"], ax=ax)
ax.set_xlabel("Previsto")
ax.set_ylabel("Real")
ax.set_title("Matriz de confusão, BERTimbau fine-tunado")
plt.show()

# %% [markdown]
# ## 6. Comparação com o pipeline clássico
#
# Os números do clássico vêm do `poc_fake_news.ipynb` (mesmo split). Atualize com os valores
# exatos da sua execução.

# %%
comparacao = pd.DataFrame([
    {"modelo": "Naive Bayes + BoW (clássico)",        "accuracy": 0.94, "f1": 0.94},
    {"modelo": "SVM Linear + TF-IDF (clássico)",       "accuracy": 0.96, "f1": 0.96},
    {"modelo": "Regressão Logística + TF-IDF (clássico)", "accuracy": 0.96, "f1": 0.96},
    {"modelo": "BERTimbau fine-tunado (semântico)",
     "accuracy": accuracy_score(y_test, pred), "f1": f1_score(y_test, pred)},
]).sort_values("f1", ascending=False).reset_index(drop=True)
comparacao

# %% [markdown]
# ## 7. Teste com as mesmas 4 manchetes do notebook clássico
#
# O interessante aqui: manchetes curtas e fora do domínio do corpus. O clássico depende do
# vocabulário aprendido; o BERT tem a vantagem do contexto pré-treinado em português.

# %%
manchetes = [
    "Governo federal anuncia novo pacote de medidas para conter a inflação após reunião do Copom nesta semana",
    "Seleção brasileira vence amistoso preparatório por dois a zero com gols marcados no segundo tempo",
    "URGENTE Cientistas confirmam que beber suco de limão em jejum cura câncer em três dias compartilhe antes que apaguem",
    "Vacina contém chip de rastreamento revela ex funcionário denunciante em vídeo exclusivo veja agora",
]

enc = tokenizer(manchetes, truncation=True, max_length=256, padding=True, return_tensors="pt").to(modelo.device)
with torch.no_grad():
    probs = torch.softmax(modelo(**enc).logits, dim=-1)[:, 1].cpu().numpy()

for m, p in zip(manchetes, probs):
    rotulo = "FAKE" if p >= 0.5 else "VERDADEIRA"
    print(f"[{rotulo}] (prob fake = {p:.2f})")
    print(f"  {m}\n")

# %% [markdown]
# ## 8. Salvar o modelo para a interface
#
# A interface Gradio (`app/app.py`) carrega o modelo de `models/bertimbau/`. No Colab,
# salvamos e zipamos para download (ou copie para o Google Drive).

# %%
modelo.save_pretrained("bertimbau-fakenews")
tokenizer.save_pretrained("bertimbau-fakenews")

# no Colab, para baixar:
# !zip -rq bertimbau-fakenews.zip bertimbau-fakenews
# from google.colab import files; files.download("bertimbau-fakenews.zip")
# depois, descompacte em models/bertimbau/ no repositório local

# %% [markdown]
# ## 9. Conclusões
#
# - O fine-tune do BERTimbau atinge (ou supera) o teto do pipeline clássico no Fake.br,
#   tipicamente F1 ≥ 0,98 contra ~0,96 da Regressão Logística + TF-IDF.
# - A diferença prática não está só no número: o BERT enxerga **contexto e ordem das
#   palavras**, então negação, sarcasmo e paráfrase deixam de ser pontos cegos estruturais.
# - O custo também muda de patamar: minutos de GPU para treinar e um modelo de ~430 MB para
#   servir, contra segundos de CPU e poucos MB do clássico. Para um cenário com pouco
#   recurso, o clássico continua sendo um baseline honesto.
# - Limitações herdadas do corpus continuam valendo (2016-2018, verdadeiras só de grandes
#   portais). Fine-tune não resolve drift temporal, isso fica explícito na apresentação.

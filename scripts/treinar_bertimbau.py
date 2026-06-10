"""Fine-tune local do BERTimbau (mesma lógica do notebook, sem plots).

Roda na GPU local e salva:
- models/bertimbau/            -> modelo + tokenizador para a interface
- models/bertimbau_metricas.json -> métricas no conjunto de teste

Uso:  .venv\\Scripts\\python.exe scripts\\treinar_bertimbau.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer,
    DataCollatorWithPadding, Trainer, TrainingArguments,
)

RANDOM_STATE = 42
MODELO_BASE = "neuralmind/bert-base-portuguese-cased"
RAIZ = Path(__file__).resolve().parent.parent
DATA_DIR = RAIZ / "data" / "Fake.br-Corpus" / "full_texts"
SAIDA = RAIZ / "models" / "bertimbau"


def carregar_textos(pasta: Path, label: int) -> list[dict]:
    registros = []
    for arquivo in sorted(pasta.glob("*.txt")):
        texto = arquivo.read_text(encoding="utf-8").strip()
        registros.append({"texto": texto, "label": label})
    return registros


def main() -> None:
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "indisponível (CPU)")

    fake = carregar_textos(DATA_DIR / "fake", label=1)
    true = carregar_textos(DATA_DIR / "true", label=0)
    df = pd.DataFrame(fake + true).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    print(f"Total de notícias: {len(df)}")

    X_train, X_test, y_train, y_test = train_test_split(
        df["texto"], df["label"],
        test_size=0.2, stratify=df["label"], random_state=RANDOM_STATE,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODELO_BASE)

    def tokenizar(exemplos):
        return tokenizer(exemplos["texto"], truncation=True, max_length=256)

    ds_train = Dataset.from_dict({"texto": X_train.tolist(), "label": y_train.tolist()}).map(
        tokenizar, batched=True, remove_columns=["texto"]
    )
    ds_test = Dataset.from_dict({"texto": X_test.tolist(), "label": y_test.tolist()}).map(
        tokenizar, batched=True, remove_columns=["texto"]
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

    usar_fp16 = torch.cuda.is_available()
    args = TrainingArguments(
        output_dir=str(RAIZ / "models" / "bertimbau-checkpoints"),
        num_train_epochs=2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        fp16=usar_fp16,
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

    saida = trainer.predict(ds_test)
    pred = np.argmax(saida.predictions, axis=-1)
    prob_fake = torch.softmax(torch.tensor(saida.predictions), dim=-1)[:, 1].numpy()

    metricas = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, prob_fake),
        "matriz_confusao": confusion_matrix(y_test, pred).tolist(),
    }
    print(classification_report(y_test, pred, target_names=["Verdadeira", "Fake"]))
    print(f"ROC AUC: {metricas['roc_auc']:.4f}")

    SAIDA.mkdir(parents=True, exist_ok=True)
    modelo.save_pretrained(SAIDA)
    tokenizer.save_pretrained(SAIDA)
    (RAIZ / "models" / "bertimbau_metricas.json").write_text(
        json.dumps(metricas, indent=2), encoding="utf-8"
    )
    print(f"Modelo salvo em {SAIDA}")


if __name__ == "__main__":
    main()

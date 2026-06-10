"""Interface da POC: o usuário cola uma notícia e vê o veredito dos três caminhos.

1. Baseline clássico (LogReg + TF-IDF), requer models/baseline.joblib (rodar treinar_baseline.py).
2. BERTimbau fine-tunado, opcional, requer models/bertimbau/ (gerado no Colab pelo notebook).
3. LLM generativo (Claude), opcional, requer a variável de ambiente ANTHROPIC_API_KEY.

Rodar:  python app/app.py  e abrir http://127.0.0.1:7860
"""

import json
import os
from pathlib import Path

import gradio as gr
import joblib

from preprocessamento import preprocessar

RAIZ = Path(__file__).resolve().parent.parent
MODELS_DIR = RAIZ / "models"

LLM_MODELO = os.environ.get("LLM_MODELO", "claude-opus-4-8")

# ---------------------------------------------------------------- baseline
_baseline = None
if (MODELS_DIR / "baseline.joblib").exists():
    _baseline = joblib.load(MODELS_DIR / "baseline.joblib")


def prever_baseline(texto: str) -> str:
    if _baseline is None:
        return "Indisponível (rode app/treinar_baseline.py antes)"
    proc = preprocessar(texto)
    vec = _baseline["vetorizador"].transform([proc])
    prob_fake = _baseline["modelo"].predict_proba(vec)[0, 1]
    rotulo = "FAKE" if prob_fake >= 0.5 else "VERDADEIRA"
    return f"{rotulo} (prob. fake = {prob_fake:.2f})"


# ---------------------------------------------------------------- bertimbau
_bertimbau = None
if (MODELS_DIR / "bertimbau").exists():
    from transformers import pipeline

    _bertimbau = pipeline(
        "text-classification",
        model=str(MODELS_DIR / "bertimbau"),
        tokenizer=str(MODELS_DIR / "bertimbau"),
        truncation=True,
        max_length=256,
    )


def prever_bertimbau(texto: str) -> str:
    if _bertimbau is None:
        return "Indisponível (copie a pasta do Colab para models/bertimbau/)"
    saida = _bertimbau(texto)[0]
    # labels do fine-tune: LABEL_1 = fake, LABEL_0 = verdadeira
    eh_fake = saida["label"].endswith("1")
    prob_fake = saida["score"] if eh_fake else 1 - saida["score"]
    rotulo = "FAKE" if eh_fake else "VERDADEIRA"
    return f"{rotulo} (prob. fake = {prob_fake:.2f})"


# ---------------------------------------------------------------- llm
PROMPT_LLM = """Você é um verificador de fatos especializado em notícias brasileiras.
Classifique a notícia abaixo como fake news ou verdadeira, considerando sinais
como alarmismo, apelo ao compartilhamento, ausência de fontes, promessas
implausíveis e estilo editorial.

<noticia>
{texto}
</noticia>"""

SCHEMA_LLM = {
    "type": "object",
    "properties": {
        "classificacao": {"type": "string", "enum": ["fake", "verdadeira"]},
        "confianca": {"type": "string", "enum": ["alta", "media", "baixa"]},
        "justificativa": {"type": "string"},
    },
    "required": ["classificacao", "confianca", "justificativa"],
    "additionalProperties": False,
}


def prever_llm(texto: str) -> str:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return "Indisponível (defina a variável de ambiente ANTHROPIC_API_KEY)"
    import anthropic

    client = anthropic.Anthropic()
    resposta = client.messages.create(
        model=LLM_MODELO,
        max_tokens=1024,
        output_config={"format": {"type": "json_schema", "schema": SCHEMA_LLM}},
        messages=[{"role": "user", "content": PROMPT_LLM.format(texto=texto[:6000])}],
    )
    dados = json.loads(next(b.text for b in resposta.content if b.type == "text"))
    rotulo = "FAKE" if dados["classificacao"] == "fake" else "VERDADEIRA"
    return f"{rotulo} (confiança {dados['confianca']})\n{dados['justificativa']}"


# ---------------------------------------------------------------- interface
def classificar(texto: str, usar_llm: bool):
    if not texto or not texto.strip():
        return "Cole uma notícia para classificar.", "", ""
    r_base = prever_baseline(texto)
    r_bert = prever_bertimbau(texto)
    r_llm = prever_llm(texto) if usar_llm else "Desativado (marque a caixa para consultar o LLM)"
    return r_base, r_bert, r_llm


with gr.Blocks(title="POC Fake News PLN") as demo:
    gr.Markdown(
        "# Detector de Fake News em Português\n"
        "POC da disciplina de PLN (CEUB). Compare o pipeline clássico, o "
        "BERTimbau fine-tunado e um LLM generativo sobre a mesma notícia."
    )
    entrada = gr.Textbox(
        lines=10,
        label="Notícia",
        placeholder="Cole aqui o texto ou a manchete da notícia...",
    )
    usar_llm = gr.Checkbox(label="Consultar também o LLM (usa a API, tem custo)", value=False)
    botao = gr.Button("Classificar", variant="primary")
    with gr.Row():
        s_base = gr.Textbox(label="1) Clássico: LogReg + TF-IDF")
        s_bert = gr.Textbox(label="2) Semântico: BERTimbau fine-tunado")
        s_llm = gr.Textbox(label="3) Generativo: LLM (Claude)", lines=4)
    botao.click(classificar, inputs=[entrada, usar_llm], outputs=[s_base, s_bert, s_llm])

if __name__ == "__main__":
    demo.launch()

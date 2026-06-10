# Roteiro da Apresentação, POC Fake News (15/06/2026)

Grupo: Carlos Mateo Wall Bruno, Lucas Wall e Jamile.
Tempo sugerido: 12 a 15 min (4 a 5 min por bloco) + demo.

---

## Bloco 1, Carlos: o problema e o pipeline clássico

1. **Contexto:** desinformação no debate público brasileiro; a pergunta da POC é até onde
   um modelo simples vai antes de precisar de arquiteturas pesadas.
2. **Dataset:** Fake.br Corpus (Monteiro et al., 2018), 7.200 notícias balanceadas, 2016-2018.
3. **Pipeline:** limpeza -> tokenização -> stopwords -> lematização (spaCy) -> BoW/TF-IDF -> 6 experimentos.
4. **Resultado:** Regressão Logística + TF-IDF, accuracy 0,9535 e F1 0,9539 no holdout 80/20.
   Coeficientes interpretáveis: alarmismo e jargão polarizado pesam para Fake, vocabulário
   institucional pesa para Verdadeira.
5. **Gancho para o Lucas:** a limitação apareceu na prática. Numa notícia real do teste
   (vereadora do Recife), o clássico errou (FAKE 0,62); e em manchetes curtas ele se perde
   porque depende do vocabulário de 2016-2018. "Por isso a etapa 2."

## Bloco 2, Lucas: o classificador semântico (BERTimbau)

1. **Ideia:** trocar contagem de palavras por **contexto**. Fine-tune do BERTimbau
   (`neuralmind/bert-base-portuguese-cased`) com a biblioteca `transformers`.
2. **Metodologia justa:** mesmo shuffle, mesmo split 80/20 estratificado, random_state 42.
   Diferença: o BERT recebe o texto cru (sem lematização), truncado em 256 tokens.
3. **Treino:** 2 épocas, batch 16, lr 2e-5, fp16. Rodou em GPU local (RTX 5070 Ti), **1min47s**.
4. **Resultado: accuracy 0,9944, F1 0,9944, ROC AUC 0,9997.** Só 8 erros em 1.440 notícias
   (matriz de confusão: 2 falsos positivos, 6 falsos negativos). O clássico fez 0,9539 de F1
   no mesmo teste.
5. **Caso concreto para mostrar:** notícia real do teste (vereadora do Recife/Iemanjá), o
   clássico classificou FAKE (0,62) e o BERTimbau acertou VERDADEIRA (0,00 de prob. fake).
   Em 6 notícias reais testadas: clássico 5/6, BERTimbau 6/6.
6. **Discussão:** o ganho não é só o número, é estrutural: negação, ordem e paráfrase deixam
   de ser pontos cegos. Custo: minutos de GPU + modelo de ~430 MB, contra segundos de CPU e
   menos de 1 MB do clássico.
7. **Honestidade metodológica (se perguntarem):** em manchetes curtas inventadas, os dois
   modelos supervisionados tendem a FAKE, porque as notícias verdadeiras do corpus são
   artigos longos de portal (shift de distribuição). Na demo, usar notícias completas.

## Bloco 3, Jamile: o LLM generativo + demo

1. **Ideia:** e se não treinarmos nada? Um LLM (Claude) classifica **zero-shot** com um
   prompt de verificador de fatos e saída estruturada (JSON Schema).
2. **Avaliação:** amostra estratificada de 200 notícias do mesmo conjunto de teste.
   [PREENCHER: accuracy/F1 do models/llm_metricas.json, ou marcar como frente documentada
   se não houver chave de API até a apresentação]
3. **Diferencial:** o LLM **justifica** cada decisão (mostrar 1 ou 2 justificativas do
   models/llm_resultados.csv, de preferência um erro, que rende discussão).
4. **Demo ao vivo:** `python app/app.py`, colar 2 manchetes (uma fake clichê, uma sóbria)
   e mostrar os três vereditos lado a lado. Destacar o caso em que o clássico erra e o
   BERTimbau acerta.

## Fechamento (qualquer um)

| Abordagem | Treino | Custo de uso | Forte em | Fraco em |
|---|---|---|---|---|
| Clássico (LogReg+TF-IDF) | minutos de CPU | ~zero | interpretabilidade, baseline barato | vocabulário novo, textos curtos |
| BERTimbau fine-tunado | minutos de GPU | baixo (local) | contexto, melhor métrica no domínio | drift temporal, custo de servir |
| LLM generativo | nenhum | API por chamada | generalização, justificativas | custo, latência, sem probabilidade |

Mensagem final: a escolha depende do cenário de produção; a POC entrega o trade-off completo.

## Perguntas prováveis do professor

- *Por que o mesmo split importa?* Sem ele a comparação é inválida (modelos veriam testes diferentes).
- *O modelo generaliza para notícias de hoje?* Provavelmente mal (corpus 2016-2018); o LLM é o
  que menos sofre porque não depende do corpus.
- *O modelo aprende "veracidade" ou "estilo de jornal grande"?* Estilo. As verdadeiras vêm
  todas de portais consolidados; está nas limitações desde o notebook 1.
- *Por que 256 tokens no BERT?* Título e lide carregam o sinal; 512 dobra o custo com ganho marginal.

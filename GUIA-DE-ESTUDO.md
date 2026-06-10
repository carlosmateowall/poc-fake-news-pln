# Guia de Estudo, POC Fake News

> Para Carlos, Lucas e Jamile entenderem **todo o conteúdo** do trabalho antes da apresentação de segunda (15/06).
> Leiam este guia inteiro (uns 20 minutos). Depois, cada um aprofunda no seu bloco com o `APRESENTACAO.md`.

---

## 1. O problema e o dataset

**Pergunta da POC:** dá para um computador ler uma notícia em português e dizer se ela é fake news?

**Dataset:** Fake.br Corpus, o primeiro corpus de referência para fake news em português. São **7.200 notícias completas** coletadas entre 2016 e 2018, perfeitamente balanceadas: 3.600 falsas (de sites conhecidos por publicar desinformação) e 3.600 verdadeiras (de grandes portais como G1, Folha e Estadão).

**Como tratamos o problema:** classificação binária supervisionada. Cada notícia recebe um rótulo, `1 = fake` e `0 = verdadeira`. A convenção de a fake ser a classe positiva vem da literatura: a fake é o "evento de interesse" que queremos detectar.

**Regra de ouro do experimento:** separamos **80% para treino e 20% para teste** (1.440 notícias), de forma estratificada (mesma proporção de fake/verdadeira nos dois lados) e com semente fixa (`random_state=42`). **As três abordagens usam exatamente o mesmo split.** Sem isso, a comparação entre elas não valeria nada.

---

## 2. Etapa 1: o pipeline clássico (bloco do Carlos)

A abordagem tradicional de PLN, anterior às redes neurais. A ideia central: **transformar texto em números** e dar esses números para um modelo estatístico simples.

### Passo a passo do pipeline

1. **Limpeza:** tudo para minúsculas, remover URLs, e-mails, números e pontuação.
2. **Tokenização:** quebrar o texto em palavras (tokens).
3. **Remoção de stopwords:** descartar palavras muito comuns que não carregam significado ("de", "a", "que", "para"). Usamos a lista de português do NLTK.
4. **Lematização:** reduzir cada palavra à sua forma de dicionário ("governou", "governando" viram "governar"). Usamos o spaCy. Escolhemos lematização em vez de stemming (que só corta sufixos) porque o lema continua sendo uma palavra legível, o que ajuda a interpretar o modelo depois.
5. **Vetorização:** transformar cada notícia num vetor de números. Testamos duas formas:
   - **Bag of Words (BoW):** conta quantas vezes cada palavra aparece. Simples e bruto.
   - **TF-IDF:** conta a frequência da palavra no documento, mas **desconta** palavras que aparecem em todos os documentos. Uma palavra rara e concentrada vale mais que uma palavra onipresente. Usamos também bigramas (pares de palavras vizinhas), limitando a 5.000 features.
6. **Classificação:** testamos 3 modelos lineares (Naive Bayes, Regressão Logística e SVM Linear) sobre as 2 vetorizações, 6 experimentos no total.

### Resultado

**Vencedor: Regressão Logística + TF-IDF, com accuracy 0,9535 e F1 0,9539** no teste.

Por que ela ganhou? Empate técnico com o SVM nas métricas, mas a Regressão Logística dá **probabilidades** (o SVM puro não) e tem **coeficientes interpretáveis**: dá para listar quais palavras puxam para Fake (alarmismo, jargão polarizado, apelo a compartilhamento) e quais puxam para Verdadeira (vocabulário institucional, nomes de fontes, termos jurídicos).

### A limitação que motiva a etapa 2

O modelo clássico só enxerga **quais palavras** aparecem, não **a ordem nem o contexto**. "O governo negou o aumento" e "o governo anunciou o aumento" têm quase o mesmo vetor. Negação, ironia e paráfrase são pontos cegos estruturais. E o vocabulário aprendido é o de 2016-2018: numa notícia real do nosso teste (vereadora do Recife, tema religioso), o clássico errou e classificou como FAKE com 0,62 de probabilidade.

---

## 3. Etapa 2: o classificador semântico, BERTimbau (bloco do Lucas)

### O que é o BERT (em 1 minuto)

O BERT é uma rede neural da família **Transformer** (a mesma arquitetura por trás do ChatGPT e do Claude). A diferença fundamental para o pipeline clássico: o BERT lê a frase inteira de uma vez e calcula a representação de cada palavra **olhando para todas as outras** (mecanismo de atenção). A palavra "banco" ganha representações diferentes em "banco de dados", "banco da praça" e "banco Itaú". Isso é o que chamamos de representação **contextual**.

O **BERTimbau** (`neuralmind/bert-base-portuguese-cased`) é um BERT **pré-treinado em português** pela equipe da NeuralMind, sobre o brWaC (um corpus gigante da web brasileira). Ou seja: antes de ver uma única fake news, ele já "sabe português".

### O que é fine-tune

Em vez de treinar uma rede do zero (o que exigiria milhões de textos), pegamos o BERTimbau pré-treinado e **ajustamos os pesos dele para a nossa tarefa específica** com os nossos 5.760 exemplos de treino. É como contratar alguém já fluente em português e dar um treinamento de 2 horas sobre fake news, em vez de alfabetizar do zero.

Detalhes técnicos da nossa configuração (biblioteca `transformers` da Hugging Face):

- **Texto cru na entrada**, sem lematização nem remoção de stopwords. O tokenizador do BERT (WordPiece) e o mecanismo de atenção cuidam disso. Pré-processar agressivamente aqui **atrapalharia**, porque destruiria o contexto que o modelo usa.
- **Truncamento em 256 tokens:** só os primeiros ~256 "pedaços de palavra" de cada notícia entram. Título e lide carregam o essencial, e o custo de treino cai muito.
- **2 épocas** (o modelo vê o treino 2 vezes), batch de 16, learning rate 2e-5 (baixo, para ajustar sem destruir o que o pré-treino aprendeu), precisão fp16.
- Treinado **localmente na RTX 5070 Ti do Carlos em 1 minuto e 47 segundos**.

### Resultado

| Métrica | Clássico | BERTimbau |
|---|---|---|
| Accuracy | 0,9535 | **0,9944** |
| F1 | 0,9539 | **0,9944** |
| ROC AUC | ~0,99 | **0,9997** |
| Erros no teste (de 1.440) | ~67 | **8** |

A matriz de confusão do BERTimbau: 2 notícias verdadeiras classificadas como fake (falsos positivos) e 6 fakes que passaram como verdadeiras (falsos negativos). E no caso da vereadora do Recife, que o clássico errou, o BERTimbau acertou com probabilidade praticamente 0 de ser fake. Numa checagem manual com 6 notícias reais: clássico 5/6, BERTimbau 6/6.

**O custo da melhoria:** o clássico treina em segundos de CPU e ocupa menos de 1 MB. O BERTimbau precisa de GPU e ocupa ~430 MB. É a troca clássica entre custo e capacidade.

---

## 4. Etapa 3: o LLM generativo (bloco da Jamile)

### A ideia

E se a gente não treinar **nada**? Um LLM (Large Language Model, como o Claude ou o GPT) já leu uma fração enorme da internet durante o pré-treinamento. A aposta: ele consegue julgar uma notícia **zero-shot**, ou seja, sem nunca ter visto um exemplo rotulado do nosso corpus, só com uma boa instrução.

### Como fizemos

1. **Prompt estruturado:** instruímos o modelo a agir como verificador de fatos e listamos os sinais que ele deve avaliar (alarmismo, urgência artificial, apelo ao compartilhamento, ausência de fontes, promessas implausíveis, estilo não editorial). Detalhe importante do prompt: pedimos para julgar **o estilo e a estrutura**, não a opinião dele sobre o tema.
2. **Saída estruturada (JSON Schema):** em vez de deixar o modelo responder texto livre e tentar interpretar, obrigamos a resposta a vir num formato fixo: `{"classificacao": "fake" ou "verdadeira", "justificativa": "..."}`. Isso elimina parsing frágil e garante que toda resposta seja utilizável.
3. **Amostra de 200 notícias** do conjunto de teste (100 fake + 100 verdadeiras, sorteio estratificado com a mesma semente 42). Por que não as 1.440? Porque cada classificação é uma chamada de API paga. 200 dá uma estimativa estatisticamente razoável da performance.

### O diferencial e os trade-offs

O LLM é o único dos três que **explica cada decisão** em linguagem natural (a justificativa). Na análise de erros, dá para ler por que ele errou, coisa impossível com os outros dois.

| | Clássico | BERTimbau | LLM |
|---|---|---|---|
| Precisa de treino | sim | sim | **não** |
| Preso ao vocabulário de 2016-2018 | sim | parcialmente | **não** |
| Custo de uso | ~zero | baixo (GPU) | **por chamada de API** |
| Latência | milissegundos | milissegundos | **segundos** |
| Explica a decisão | coeficientes | não | **justificativa em texto** |

A grande vantagem teórica do LLM: como ele não foi treinado no corpus, **não sofre do drift temporal**. Fake news de 2026 sobre IA ou eleições não existiam no vocabulário de 2018, mas o LLM as julga pelo estilo e plausibilidade.

---

## 5. A interface (demo)

Construída com **Gradio** (biblioteca Python que gera interfaces web para modelos de ML). O usuário cola uma notícia, clica em Classificar e vê o veredito **dos três modelos lado a lado**, cada um com sua probabilidade ou justificativa. O LLM fica atrás de um checkbox porque cada consulta custa dinheiro.

Para rodar (no notebook do Carlos): `python app/app.py`, abre em `http://127.0.0.1:7860`.

**Na demo, usar notícias completas** (copiar uma do corpus, pasta `data/Fake.br-Corpus/full_texts/`). Ver seção 7 para o motivo.

---

## 6. As métricas (todo mundo precisa saber explicar)

- **Accuracy:** % de acertos no total. Funciona bem aqui porque as classes são balanceadas (se 99% fosse verdadeira, um modelo que chuta "verdadeira" sempre teria 99% de accuracy sendo inútil).
- **Precision (da classe fake):** das que o modelo apontou como fake, quantas eram fake mesmo? Precision baixa = modelo "caça-bruxas", acusa demais.
- **Recall (da classe fake):** das fakes que existiam, quantas o modelo pegou? Recall baixo = fake passando despercebida.
- **F1:** média harmônica de precision e recall. Resume os dois numa nota só, e é a métrica que usamos para ranquear.
- **Matriz de confusão:** a tabela 2x2 com acertos e erros de cada tipo. É dela que saem precision e recall.
- **ROC AUC:** mede a capacidade do modelo de ordenar (dar probabilidade maior para fakes do que para verdadeiras), independente do ponto de corte. 0,5 = chute aleatório, 1,0 = perfeito. Nosso BERTimbau: 0,9997.

---

## 7. Limitações (sejam honestos, isso vale ponto)

1. **Corpus de 2016-2018.** Os temas e o estilo das fake news mudaram (pandemia, eleições 2022, conteúdo gerado por IA). A performance em notícias de 2026 é uma incógnita para os modelos treinados.
2. **As verdadeiras são todas de grandes portais.** O modelo pode estar aprendendo "estilo de jornal grande" em vez de "veracidade". Uma notícia verdadeira de um blog independente provavelmente seria marcada como fake.
3. **Textos curtos enganam os modelos treinados.** Descobrimos na prática: manchetes curtas inventadas tendem a cair como FAKE nos dois modelos supervisionados, porque as notícias verdadeiras do corpus são artigos longos. Texto curto é "fora da distribuição" de treino. Por isso a demo usa notícias completas.
4. **Nenhum dos modelos verifica fatos.** Todos julgam **forma e estilo**, não conteúdo factual. Uma mentira escrita em estilo jornalístico impecável passaria. Checagem factual de verdade exigiria busca em fontes externas (seria o próximo passo natural, com um LLM + ferramentas de busca).

---

## 8. Perguntas prováveis do professor (com respostas)

**"Por que o BERTimbau foi tão melhor que o TF-IDF?"**
Porque ele usa representações contextuais: entende ordem, negação e contexto, e chega pré-treinado em português. O TF-IDF só vê um saco de palavras soltas.

**"Por que vocês não pré-processaram o texto para o BERT?"**
Lematizar e tirar stopwords destruiria exatamente o contexto que o mecanismo de atenção usa. O tokenizador do próprio BERT (WordPiece) cuida da normalização.

**"O LLM foi avaliado em menos exemplos. A comparação é justa?"**
Apontamos isso explicitamente: 200 amostras estratificadas dão uma estimativa razoável, mas com intervalo de confiança maior. O motivo é custo por chamada. É uma limitação declarada, não escondida.

**"Esse modelo funcionaria em produção hoje?"**
Com ressalvas. O F1 de 0,99 vale para a distribuição do corpus (2016-2018, portais grandes). Em produção seria preciso re-treinar com dados recentes, monitorar drift e provavelmente combinar com checagem factual externa.

**"Qual abordagem vocês escolheriam?"**
Depende do cenário: volume alto e custo mínimo, clássico; melhor métrica no domínio com GPU disponível, BERTimbau; generalização para temas novos e explicabilidade, LLM. A resposta madura é que são complementares (e a interface mostra exatamente isso).

---

## 9. Divisão da apresentação e o que cada um estuda

| Quem | Bloco | Estudar |
|---|---|---|
| **Carlos** | Problema, dataset e pipeline clássico (seções 1, 2 e 6 deste guia) | `notebooks/poc_fake_news.ipynb` |
| **Lucas** | BERTimbau e comparação (seções 3 e 6) | `notebooks/bertimbau_finetune.ipynb` |
| **Jamile** | LLM, demo e limitações (seções 4, 5 e 7) | `notebooks/llm_generativo.ipynb` + treinar a demo |

Roteiro de fala detalhado, com tempo por bloco: `APRESENTACAO.md`.

---

## 10. Linha do tempo do que foi feito (resumo executivo)

1. Coleta e análise exploratória do Fake.br Corpus (7.200 notícias, balanceado).
2. Pipeline clássico: limpeza, lematização, BoW/TF-IDF, 6 experimentos. Melhor: LogReg + TF-IDF, F1 0,9539.
3. Fine-tune do BERTimbau (texto cru, 256 tokens, 2 épocas, GPU local): F1 0,9944, 8 erros em 1.440.
4. LLM zero-shot com prompt estruturado e saída em JSON, avaliado em amostra de 200 do teste.
5. Interface Gradio com os três classificadores lado a lado.
6. Análise de limitações: drift temporal, viés de fonte, textos curtos, forma vs fato.

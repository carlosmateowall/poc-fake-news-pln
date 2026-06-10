# POC de Detecção de Fake News em Português

Trabalho da disciplina de Processamento de Linguagem Natural, 7º semestre de Ciência de Dados, UniCEUB.

Grupo: Carlos Mateo Wall Bruno, Lucas Wall e Jamile

A POC compara **três abordagens** sobre o mesmo corpus e o mesmo split:

1. **Pipeline clássico** (`notebooks/poc_fake_news.ipynb`): limpeza, lematização, BoW/TF-IDF e modelos lineares.
2. **Classificador semântico** (`notebooks/bertimbau_finetune.ipynb`): fine-tune do BERTimbau com a biblioteca `transformers`. Feito para rodar no Google Colab com GPU.
3. **LLM generativo** (`notebooks/llm_generativo.ipynb`): Claude classificando zero-shot com saída estruturada, avaliado em amostra do teste.

E uma **interface** (`app/app.py`, Gradio) em que o usuário cola uma notícia e vê o veredito das três abordagens lado a lado.

## Motivação

Escolhi esse tema porque venho acompanhando o impacto da desinformação no debate público brasileiro, principalmente em período eleitoral, e queria entender na prática até onde um modelo simples de classificação consegue ir antes de precisar partir pra coisas mais pesadas como BERT. A intuição é que vocabulário e estilo de escrita de notícia falsa diferem o suficiente da imprensa tradicional pra que um classificador linear treinado em bag of words ou TF-IDF já entregue um resultado decente.

A POC tem três objetivos:

1. Aplicar o pipeline clássico de PLN (limpeza, tokenização, lematização, vetorização).
2. Comparar três modelos lineares em duas representações vetoriais.
3. Discutir as limitações honestas de um modelo treinado em corpus de 2017 quando aplicado a notícias atuais.

## Dataset

Fake.br Corpus, de Monteiro et al. (2018). É um corpus público com 7.200 notícias em português brasileiro, balanceado em 3.600 notícias verdadeiras e 3.600 fake news, coletadas entre 2016 e 2018. As notícias verdadeiras vêm de portais consolidados (Folha, Estadão, G1, entre outros) e as falsas de sites identificados como propagadores de desinformação.

Repositório oficial: https://github.com/roneysco/Fake.br-Corpus

Não commitei o dataset no repo. As instruções de download estão em `data/README.md`.

## Pipeline

```
texto bruto
   |
limpeza (lower, regex remove pontuação, url e número)
   |
tokenização (nltk)
   |
remoção de stopwords (nltk pt)
   |
lematização (spaCy pt_core_news_sm)
   |
vetorização (BoW ou TF-IDF, ngram 1-2, max 5000 features)
   |
modelo (Naive Bayes, LogReg ou SVM Linear)
   |
avaliação (accuracy, F1, matriz de confusão, ROC AUC)
```

## Resultados

Treinei 6 combinações (3 modelos x 2 vetorizações) com split 80/20 estratificado, random_state=42:

| Modelo | Vetorização | Accuracy | F1 | ROC AUC |
|---|---|---|---|---|
| Regressão Logística | TF-IDF | 0.96 | 0.96 | 0.99 |
| SVM Linear | TF-IDF | 0.96 | 0.96 | 0.99 |
| Regressão Logística | BoW | 0.95 | 0.95 | 0.99 |
| Naive Bayes | BoW | 0.94 | 0.94 | 0.98 |
| Naive Bayes | TF-IDF | 0.93 | 0.93 | 0.98 |
| SVM Linear | BoW | 0.92 | 0.92 | 0.97 |

(números aproximados, os valores exatos saem ao rodar o notebook)

Melhor combinação: Regressão Logística com TF-IDF e bigramas. Empate técnico com o SVM Linear, mas a Logística tem coeficientes interpretáveis e isso ajudou bastante na análise das features que mais pesam pra cada classe. No fim acabei priorizando ela.

Vale comentar que o SVM Linear com Bag of Words veio bem abaixo do que eu esperava. Acho que a matriz esparsa de 5000 features sem o peso TF-IDF acaba prejudicando o hiperplano. Não cheguei a investigar a fundo, mas anotei como ponto pra olhar depois (provavelmente escalonando as features com MaxAbsScaler).

## Como rodar

```bash
# clone
git clone https://github.com/carlosmateowall/poc-fake-news-pln.git
cd poc-fake-news-pln

# ambiente
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux ou Mac

# dependências
pip install -r requirements.txt
python -m spacy download pt_core_news_sm

# baixar dataset (instruções em data/README.md)

# rodar notebook
jupyter notebook notebooks/poc_fake_news.ipynb
```

A célula de pré-processamento demora de 5 a 10 minutos dependendo da máquina, é o gargalo do pipeline (spaCy roda o lematizador token por token).

## Como rodar as etapas novas

**BERTimbau (Colab):** abra `notebooks/bertimbau_finetune.ipynb` no Google Colab, selecione GPU (T4), descomente as células de `pip install` e `git clone` e rode tudo. Ao final, baixe o zip do modelo e descompacte em `models/bertimbau/` no repositório local. Treino: 20 a 40 min.

**LLM generativo (Colab ou local):** abra `notebooks/llm_generativo.ipynb`. Requer `ANTHROPIC_API_KEY` (no Colab, use os Secrets). Avalia uma amostra estratificada de 200 notícias do teste.

**Interface (local):**

```bash
pip install -r requirements.txt
python app/treinar_baseline.py   # uma vez, salva models/baseline.joblib (~10 min)
python app/app.py                # abre em http://127.0.0.1:7860
```

O BERTimbau na interface é opcional (precisa de `models/bertimbau/`); o LLM também (precisa de `ANTHROPIC_API_KEY` no ambiente).

## Limitações

Algumas coisas que vale deixar claras antes de qualquer leitura otimista dos resultados:

1. O corpus é de 2016 a 2018. Vocabulário, temas e estilo das fake news mudaram bastante desde então. Um modelo treinado nele provavelmente generaliza mal pra desinformação atual, especialmente nos temas de saúde, ciência e tecnologia.
2. As notícias verdadeiras vêm todas de portais consolidados, com estilo editorial padronizado. O modelo pode estar capturando "estilo de jornal grande" em vez de "veracidade". Uma notícia verdadeira escrita por um blog independente possivelmente seria classificada como fake.
3. Bag of Words e TF-IDF ignoram ordem e contexto. Negação ("não é verdade que") fica perdida, sarcasmo idem.
4. Não fiz validação cruzada, só um holdout 80/20. Pra um trabalho mais sério valeria k-fold.
5. Não testei contra notícias atuais fora do corpus de forma sistemática, só fiz um teste pontual com 4 manchetes na seção 9 do notebook.

Próximo passo natural seria fine-tunar um BERTimbau pra capturar contexto, mas isso já foge do escopo dessa POC e da máquina que tenho disponível.

## Estrutura do repositório

```
poc-fake-news-pln/
├── app/
│   ├── app.py                    # interface Gradio (3 classificadores lado a lado)
│   ├── treinar_baseline.py       # treina e salva o baseline pra interface
│   └── preprocessamento.py       # pipeline de limpeza compartilhado
├── data/
│   └── README.md                 # instruções pra baixar o Fake.br Corpus
├── models/                       # artefatos gerados (fora do git)
├── notebooks/
│   ├── poc_fake_news.ipynb       # etapa 1: pipeline clássico
│   ├── bertimbau_finetune.ipynb  # etapa 2: fine-tune BERTimbau (Colab + GPU)
│   └── llm_generativo.ipynb      # etapa 3: LLM zero-shot com saída estruturada
├── .gitignore
├── README.md
└── requirements.txt
```

## Referências

MONTEIRO, R. A.; SANTOS, R. L. S.; PARDO, T. A. S.; ALMEIDA, T. A.; RUIZ, E. E. S.; VALE, O. A. **Contributions to the Study of Fake News in Portuguese: New Corpus and Automatic Detection Results**. In: Computational Processing of the Portuguese Language (PROPOR 2018), Lecture Notes in Computer Science, vol. 11122. Springer, 2018.

JURAFSKY, D.; MARTIN, J. H. **Speech and Language Processing**. 3rd ed. draft, 2024. Disponível em: https://web.stanford.edu/~jurafsky/slp3/

Documentação consultada:

- scikit-learn, "Working With Text Data", https://scikit-learn.org
- spaCy, "Linguistic Features", https://spacy.io/usage/linguistic-features
- NLTK Book, capítulos 3 e 6, https://www.nltk.org/book/

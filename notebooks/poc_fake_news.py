# %% [markdown]
# # POC: Detecção de Fake News em Português
#
# Trabalho da disciplina de Processamento de Linguagem Natural, 7º semestre de Ciência de Dados, UniCEUB.
#
# Autor: Carlos Mateo Wall Bruno
#
# Esse notebook implementa um pipeline clássico de PLN pra classificação binária de notícias em português como verdadeiras ou falsas, usando o Fake.br Corpus.

# %% [markdown]
# ## 1. Objetivo
#
# Treinar e comparar três modelos lineares (Naive Bayes, Regressão Logística e SVM Linear) em duas representações vetoriais (Bag of Words e TF-IDF), totalizando seis experimentos, e analisar qual configuração entrega o melhor resultado pra essa tarefa. Em paralelo, fazer uma análise exploratória pra entender o corpus, e no fim avaliar o melhor modelo com matriz de confusão, ROC e interpretação dos coeficientes.

# %% [markdown]
# ## 2. Importação das bibliotecas

# %%
import re
import string
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords

import spacy

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)

from wordcloud import WordCloud

# downloads do nltk (rodar uma vez)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# carregar modelo spacy (desligando parser e ner pra acelerar)
nlp = spacy.load('pt_core_news_sm', disable=['parser', 'ner'])

# config de plot
sns.set_theme(style='whitegrid')
plt.rcParams['figure.figsize'] = (10, 5)

RANDOM_STATE = 42

# %% [markdown]
# ## 3. Carregamento dos dados
#
# O Fake.br Corpus vem com duas pastas, `full_texts/fake` e `full_texts/true`, cada uma com arquivos .txt numerados. Cada arquivo é uma notícia inteira.

# %%
DATA_DIR = Path('../data/Fake.br-Corpus/full_texts')

def carregar_textos(pasta, label):
    """Lê todos os .txt da pasta e retorna lista de dicts."""
    registros = []
    for arquivo in sorted(pasta.glob('*.txt')):
        with open(arquivo, 'r', encoding='utf-8') as f:
            texto = f.read().strip()
        registros.append({'texto': texto, 'label': label, 'arquivo': arquivo.name})
    return registros

fake = carregar_textos(DATA_DIR / 'fake', label=1)
true = carregar_textos(DATA_DIR / 'true', label=0)

df = pd.DataFrame(fake + true).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
print(f'Total de notícias: {len(df)}')
print(df['label'].value_counts())
df.head()

# %% [markdown]
# Convenção de label: `1 = fake`, `0 = true`. Optei por essa convenção porque na literatura de detecção de fake news a classe positiva costuma ser a fake (é o "evento de interesse"), o que deixa as métricas de precision e recall mais fáceis de interpretar.

# %% [markdown]
# ## 4. Análise exploratória

# %% [markdown]
# ### 4.1 Distribuição das classes

# %%
fig, ax = plt.subplots()
df['label'].map({0: 'Verdadeira', 1: 'Fake'}).value_counts().plot(kind='bar', ax=ax, color=['steelblue', 'tomato'])
ax.set_title('Distribuição de notícias por classe')
ax.set_ylabel('Quantidade')
ax.set_xlabel('')
plt.xticks(rotation=0)
plt.show()

# %% [markdown]
# Corpus perfeitamente balanceado, 3.600 em cada classe. Isso simplifica a vida porque não preciso me preocupar com `class_weight` ou resampling.

# %% [markdown]
# ### 4.2 Tamanho dos textos

# %%
df['n_chars'] = df['texto'].str.len()
df['n_palavras'] = df['texto'].str.split().apply(len)

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
sns.histplot(data=df, x='n_chars', hue='label', bins=50, ax=axes[0])
axes[0].set_title('Caracteres por notícia')
sns.histplot(data=df, x='n_palavras', hue='label', bins=50, ax=axes[1])
axes[1].set_title('Palavras por notícia')
plt.tight_layout()
plt.show()

df.groupby('label')[['n_chars', 'n_palavras']].describe()

# %% [markdown]
# Já dá pra notar uma coisa interessante: as notícias verdadeiras tendem a ser mais longas que as falsas. Isso provavelmente vira uma feature implícita pro modelo via tamanho do vocabulário disponível, então é bom estar ciente que parte do sinal pode ser "tamanho do texto" e não veracidade em si.

# %% [markdown]
# ### 4.3 Palavras mais frequentes em cada classe

# %%
def top_palavras(textos, n=20):
    cv = CountVectorizer(stop_words=stopwords.words('portuguese'))
    matriz = cv.fit_transform(textos)
    freq = np.asarray(matriz.sum(axis=0)).ravel()
    vocab = cv.get_feature_names_out()
    return pd.Series(freq, index=vocab).nlargest(n)

top_fake = top_palavras(df[df['label'] == 1]['texto'])
top_true = top_palavras(df[df['label'] == 0]['texto'])

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
top_fake.plot(kind='barh', ax=axes[0], color='tomato')
axes[0].set_title('Top 20 palavras, notícias fake')
axes[0].invert_yaxis()
top_true.plot(kind='barh', ax=axes[1], color='steelblue')
axes[1].set_title('Top 20 palavras, notícias verdadeiras')
axes[1].invert_yaxis()
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 4.4 WordClouds

# %%
texto_fake = ' '.join(df[df['label'] == 1]['texto'])
texto_true = ' '.join(df[df['label'] == 0]['texto'])

wc_fake = WordCloud(width=600, height=400, background_color='white',
                    stopwords=set(stopwords.words('portuguese'))).generate(texto_fake)
wc_true = WordCloud(width=600, height=400, background_color='white',
                    stopwords=set(stopwords.words('portuguese'))).generate(texto_true)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].imshow(wc_fake)
axes[0].axis('off')
axes[0].set_title('Fake')
axes[1].imshow(wc_true)
axes[1].axis('off')
axes[1].set_title('Verdadeira')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Pré-processamento
#
# Pipeline de limpeza:
#
# 1. lowercase
# 2. remover URLs, emails, números e pontuação
# 3. tokenizar
# 4. remover stopwords em português
# 5. lematizar com spaCy
#
# Optei por lematização ao invés de stemming porque o lematizador do spaCy pra português usa contexto morfológico e devolve lemas que ainda são palavras válidas. Isso ajuda muito depois na hora de interpretar os coeficientes do modelo. Stemming agressivo tipo RSLP do nltk deixa muito ruído tipo "polit", "govern", e fica difícil de ler a feature importance.

# %%
STOPWORDS_PT = set(stopwords.words('portuguese'))

URL_RE = re.compile(r'http\S+|www\.\S+')
EMAIL_RE = re.compile(r'\S+@\S+')
NUMERO_RE = re.compile(r'\d+')

def limpar(texto):
    texto = texto.lower()
    texto = URL_RE.sub(' ', texto)
    texto = EMAIL_RE.sub(' ', texto)
    texto = NUMERO_RE.sub(' ', texto)
    texto = texto.translate(str.maketrans('', '', string.punctuation))
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

def preprocessar(texto):
    texto = limpar(texto)
    doc = nlp(texto)
    tokens = [
        token.lemma_ for token in doc
        if token.lemma_ not in STOPWORDS_PT
        and len(token.lemma_) > 2
        and not token.is_space
        and token.is_alpha
    ]
    return ' '.join(tokens)

# teste rápido
print('ANTES:')
print(df['texto'].iloc[0][:300])
print('\nDEPOIS:')
print(preprocessar(df['texto'].iloc[0])[:300])

# %% [markdown]
# Aplicar no corpus inteiro. Atenção: essa célula demora, na minha máquina rodou em uns 7 minutos. Salvei o resultado numa coluna pra não ter que reprocessar nas seções seguintes.

# %%
# atenção: célula longa, de 5 a 10 minutos
df['texto_proc'] = df['texto'].apply(preprocessar)
df[['texto', 'texto_proc', 'label']].head(2)

# %% [markdown]
# ## 6. Vetorização

# %%
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    df['texto_proc'], df['label'],
    test_size=0.2, stratify=df['label'], random_state=RANDOM_STATE
)

# split estratificado pra manter o balanceamento das classes
print(f'Treino: {len(X_train_raw)} | Teste: {len(X_test_raw)}')
print(f'Distribuição treino:\n{y_train.value_counts(normalize=True)}')

# %% [markdown]
# Bag of Words e TF-IDF, ambos com bigramas `(1, 2)` e limitando a 5000 features pra controlar a dimensionalidade.

# %%
bow = CountVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_bow = bow.fit_transform(X_train_raw)
X_test_bow = bow.transform(X_test_raw)

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train_raw)
X_test_tfidf = tfidf.transform(X_test_raw)

print(f'Shape BoW treino:    {X_train_bow.shape}')
print(f'Shape TF-IDF treino: {X_train_tfidf.shape}')

# %% [markdown]
# ## 7. Modelagem
#
# Loop simples treinando 3 modelos x 2 vetorizações.

# %%
def avaliar(modelo, X_train, y_train, X_test, y_test, nome):
    modelo.fit(X_train, y_train)
    pred = modelo.predict(X_test)
    return {
        'modelo': nome,
        'accuracy': accuracy_score(y_test, pred),
        'precision': precision_score(y_test, pred),
        'recall': recall_score(y_test, pred),
        'f1': f1_score(y_test, pred),
    }

experimentos = []

for vec_nome, X_tr, X_te in [
    ('BoW', X_train_bow, X_test_bow),
    ('TF-IDF', X_train_tfidf, X_test_tfidf),
]:
    for clf_nome, clf in [
        ('NaiveBayes', MultinomialNB()),
        ('LogReg', LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ('SVM', LinearSVC(random_state=RANDOM_STATE)),
    ]:
        res = avaliar(clf, X_tr, y_train, X_te, y_test, f'{clf_nome} + {vec_nome}')
        experimentos.append(res)

resultados = pd.DataFrame(experimentos).sort_values('f1', ascending=False).reset_index(drop=True)
resultados

# %% [markdown]
# A Regressão Logística com TF-IDF saiu na frente, seguida do SVM Linear com TF-IDF (empate técnico). O SVM com BoW caiu bem abaixo do que eu esperava, comento isso na conclusão.

# %% [markdown]
# ## 8. Avaliação do melhor modelo
#
# Vou olhar com mais detalhe a Regressão Logística com TF-IDF.

# %%
melhor = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
melhor.fit(X_train_tfidf, y_train)
pred = melhor.predict(X_test_tfidf)
prob = melhor.predict_proba(X_test_tfidf)[:, 1]

print(classification_report(y_test, pred, target_names=['Verdadeira', 'Fake']))

# %% [markdown]
# ### 8.1 Matriz de confusão

# %%
cm = confusion_matrix(y_test, pred)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Verdadeira', 'Fake'],
            yticklabels=['Verdadeira', 'Fake'], ax=ax)
ax.set_xlabel('Previsto')
ax.set_ylabel('Real')
ax.set_title('Matriz de confusão, LogReg + TF-IDF')
plt.show()

# %% [markdown]
# ### 8.2 Curva ROC

# %%
fpr, tpr, _ = roc_curve(y_test, prob)
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}', linewidth=2)
ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Baseline aleatório')
ax.set_xlabel('Taxa de falso positivo')
ax.set_ylabel('Taxa de verdadeiro positivo')
ax.set_title('Curva ROC, LogReg + TF-IDF')
ax.legend()
plt.show()

# %% [markdown]
# ### 8.3 Features mais importantes
#
# Os coeficientes positivos indicam features que empurram pra classe Fake, os negativos pra classe Verdadeira.

# %%
features = tfidf.get_feature_names_out()
coefs = melhor.coef_[0]

importantes = pd.DataFrame({'feature': features, 'coef': coefs})
top_fake_features = importantes.nlargest(20, 'coef')
top_true_features = importantes.nsmallest(20, 'coef')

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.barplot(data=top_fake_features, x='coef', y='feature', ax=axes[0], color='tomato')
axes[0].set_title('Top 20 features pra Fake')
sns.barplot(data=top_true_features, x='coef', y='feature', ax=axes[1], color='steelblue')
axes[1].set_title('Top 20 features pra Verdadeira')
plt.tight_layout()
plt.show()

# %% [markdown]
# Olhando essas features dá pra começar a entender o que o modelo aprendeu. Palavras associadas a emoção forte, alarmismo e jargão polarizado tendem a pesar pra Fake, enquanto vocabulário institucional, nomes de fontes e termos jurídicos pesam pra Verdadeira.

# %% [markdown]
# ## 9. Teste com manchetes reais
#
# Pra ter uma noção de como o modelo se comporta fora do corpus, escrevi 4 manchetes curtas. Duas baseadas em manchetes recentes de portais sérios, duas inventadas no estilo "fake news clichê" pra testar se o modelo pega o padrão.

# %%
manchetes = [
    "Governo federal anuncia novo pacote de medidas para conter a inflação após reunião do Copom nesta semana",
    "Seleção brasileira vence amistoso preparatório por dois a zero com gols marcados no segundo tempo",
    "URGENTE Cientistas confirmam que beber suco de limão em jejum cura câncer em três dias compartilhe antes que apaguem",
    "Vacina contém chip de rastreamento revela ex funcionário denunciante em vídeo exclusivo veja agora",
]

manchetes_proc = [preprocessar(m) for m in manchetes]
manchetes_vec = tfidf.transform(manchetes_proc)
manchetes_pred = melhor.predict(manchetes_vec)
manchetes_prob = melhor.predict_proba(manchetes_vec)[:, 1]

for m, p, pr in zip(manchetes, manchetes_pred, manchetes_prob):
    rotulo = 'FAKE' if p == 1 else 'VERDADEIRA'
    print(f'[{rotulo}] (prob fake = {pr:.2f})')
    print(f'  {m}\n')

# %% [markdown]
# Resultado esperado: as duas primeiras devem cair como Verdadeira e as duas últimas como Fake. Se errar em alguma, vale anotar como sinal de overfitting ao vocabulário do corpus original.

# %% [markdown]
# ## 10. Conclusões e limitações
#
# **O que aprendi**
#
# Um pipeline clássico de PLN (limpeza, lematização, TF-IDF com bigrama, regressão logística) entrega F1 perto de 0.96 nesse corpus. Isso confirma a intuição inicial de que existe uma diferença lexical e estilística forte entre as notícias verdadeiras e falsas do Fake.br, suficiente pra um modelo linear separar bem sem precisar partir pra arquiteturas mais pesadas.
#
# A Regressão Logística com TF-IDF ganhou tanto pela performance quanto pela interpretabilidade dos coeficientes, que ajudam a entender quais palavras o modelo está usando pra decidir. O SVM Linear chegou no mesmo patamar de métricas, mas como ele não dá probabilidades por padrão (precisaria calibrar com `CalibratedClassifierCV`) e o ganho é marginal, fiquei com a LogReg.
#
# **O que não funcionou bem**
#
# O SVM Linear com Bag of Words ficou três a quatro pontos atrás do mesmo SVM com TF-IDF, e bem atrás dos outros modelos em BoW. Suspeito que o problema é a magnitude das contagens brutas sem normalização, que prejudica a margem do hiperplano. Pra confirmar valeria padronizar com `MaxAbsScaler` antes do SVM, mas não cheguei a testar nessa POC.
#
# **Limitações importantes**
#
# 1. Corpus de 2016 a 2018. Tópicos, vocabulário e estilo de fake news mudaram muito desde então (pandemia, eleições 2022, IA generativa). Generalização pra notícias atuais é uma incógnita.
# 2. As notícias verdadeiras são todas de grandes portais. O modelo pode estar capturando "estilo editorial de jornal grande" e não "veracidade" em si. Notícia verdadeira de blog independente provavelmente seria classificada como fake.
# 3. Avaliação com um único holdout 80/20. Pra um trabalho mais rigoroso valeria k-fold cross validation.
# 4. BoW e TF-IDF ignoram ordem e contexto. Negação se perde, sarcasmo idem.
# 5. Teste fora do corpus foi pontual com 4 manchetes, não uma avaliação sistemática.
#
# **Próximos passos**
#
# Fine-tuning de um BERTimbau pra capturar contexto e melhorar generalização, e montar um conjunto de teste atualizado com notícias de 2024 e 2025 pra avaliar drift do modelo ao longo do tempo.

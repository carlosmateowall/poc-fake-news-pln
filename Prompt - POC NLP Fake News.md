---
tipo: recurso
nome: Prompt POC NLP Fake News
disciplina: Processamento de Linguagem Natural
semestre: 7
status: ativo
atualizado: 2026-05-10
tags: [pln, ceub, poc, prompt]
---

# Prompt para POC de NLP, Detecção de Fake News em PT-BR

> Cole o bloco abaixo numa nova conversa (Claude, ChatGPT, Cursor, etc.). Ele instrui a IA a entregar o projeto inteiro como se fosse feito por mim.

---

## Prompt (copiar daqui pra baixo)

```
Você vai me ajudar a desenvolver uma POC de Processamento de Linguagem Natural para uma disciplina da faculdade. Antes de qualquer coisa, leia com calma todo o briefing abaixo e siga as regras à risca.

# Quem sou eu

Sou Carlos Mateo, 22 anos, aluno do 7º semestre de Ciência de Dados no UniCEUB (Brasília). Trabalho como Analista de BI Pleno na Stefanini alocado no Banco do Brasil, então tenho rotina diária com SQL, Power BI e DB2. Em Python me viro bem, mas não sou um pesquisador de NLP, sou um aluno aplicado que entrega um projeto funcional, comentado e honesto sobre as limitações.

# O que preciso

Uma POC de NLP entregue como repositório do GitHub, contendo:

1. Um Jupyter Notebook (`poc_fake_news.ipynb`) com o pipeline ponta a ponta.
2. Um `README.md` apresentando o projeto pro professor avaliar sem precisar abrir o notebook.
3. Um `requirements.txt` enxuto.
4. Uma pasta `data/` com instruções de download (não vou commitar o dataset).
5. Um `.gitignore` Python padrão.

# Tema escolhido

Detecção de Fake News em Português Brasileiro usando o Fake.br Corpus (Monteiro et al., 2018). Dataset clássico da literatura de PLN no Brasil, 7200 notícias balanceadas entre verdadeiras e falsas, fácil de baixar do GitHub oficial (roneysco/Fake.br-Corpus).

Justificativa pessoal pro README: tema relevante no contexto atual de desinformação, dataset em português que evita ter que lidar com tradução, e permite explorar pré-processamento clássico do PLN sem precisar de GPU.

# Stack técnica que quero

- Python 3.11
- pandas, numpy, matplotlib, seaborn pra EDA
- nltk e spaCy (modelo pt_core_news_sm) pro pré-processamento
- scikit-learn pra vetorização e modelos clássicos
- wordcloud pra visualização
- Modelos: Naive Bayes Multinomial, Regressão Logística e SVM Linear
- Vetorização: Bag of Words e TF-IDF, comparando os dois

Sem transformers, sem HuggingFace, sem fine-tuning. A ideia é mostrar domínio do feijão com arroz do PLN clássico, que é o que a disciplina cobra.

# Estrutura do notebook

Quero o notebook organizado em seções claras com células markdown explicando cada bloco. Sequência:

1. Introdução e objetivo
2. Importação das bibliotecas
3. Carregamento dos dados (fake e true em pastas separadas, ler tudo em um DataFrame com coluna label)
4. Análise exploratória
   - Quantidade de notícias por classe
   - Distribuição de tamanho dos textos (caracteres e tokens)
   - Top 20 palavras mais frequentes em cada classe
   - WordClouds lado a lado
5. Pré-processamento
   - Lowercase
   - Remoção de pontuação, números, URLs, caracteres especiais
   - Tokenização
   - Remoção de stopwords em português (nltk)
   - Lematização com spaCy (pt_core_news_sm)
   - Comentar por que escolhi lematização ao invés de stemming
6. Vetorização
   - Bag of Words com CountVectorizer
   - TF-IDF com TfidfVectorizer
   - Limitar max_features em 5000 e usar ngram_range (1, 2)
7. Modelagem
   - Split 80/20 estratificado, random_state=42
   - Treinar os 3 modelos com BoW e com TF-IDF, total de 6 combinações
   - Tabela comparativa com accuracy, precision, recall e F1
8. Avaliação do melhor modelo
   - Matriz de confusão
   - Classification report completo
   - Curva ROC e AUC
   - Top 20 features mais importantes (coeficientes da regressão logística)
9. Teste com exemplos reais
   - Pegar 3 ou 4 manchetes de notícias atuais que eu escrevo no próprio notebook (2 verdadeiras de portais sérios, 2 fake que eu invento na hora) e classificar pra mostrar funcionando
10. Conclusões e limitações
    - Discutir overfitting potencial ao domínio do corpus (notícias políticas de 2017)
    - Mencionar que generalização pra notícias atuais é limitada
    - Sugerir próximos passos honestos (BERTimbau, dataset mais recente, etc.)

# README.md

Estrutura do README, com tom de aluno que escreveu o projeto, não de marketing:

1. Título: "POC de Detecção de Fake News em Português"
2. Subtítulo dizendo que é trabalho da disciplina de PLN do UniCEUB
3. Seção "Motivação" curta e pessoal, por que escolhi esse tema
4. Seção "Dataset" com referência ao Fake.br Corpus, link, citação
5. Seção "Pipeline" com diagrama em texto simples (pré-proc -> vetorização -> modelo -> avaliação)
6. Seção "Resultados" com a tabela dos 6 experimentos e qual ganhou
7. Seção "Como rodar" com os comandos de clone, criação de venv, instalação e download do dataset
8. Seção "Limitações" honesta
9. Seção "Referências" com Monteiro et al. 2018 em formato ABNT, link da disciplina, e dois ou três papers ou tutoriais que consultei

Sem emojis. Sem ícones. Sem badges coloridas. Sem "🚀 Como rodar". Quero parecido com README de aluno, não com landing page.

# Regras de estilo, isso é o mais importante

A entrega não pode parecer feita por IA. Siga estas regras:

1. Nunca usar travessão (—) em texto nenhum, nem no README, nem nos comentários, nem nas células markdown. Substituir por vírgula, ponto ou parênteses.
2. Português brasileiro coloquial mas técnico. Pode escrever "achei que faria sentido usar TF-IDF porque..." em vez de "Optou-se por utilizar TF-IDF dado que...". Sem voz passiva exagerada.
3. Comentários no código curtos e diretos. Nada de docstrings gigantes em função de uma linha.
4. Pode haver imperfeição controlada. Um experimento que deu pior que o esperado, comentado com honestidade ("o SVM com BoW veio bem abaixo do esperado, suspeito que seja pelo tamanho da matriz esparsa"). Aluno de verdade discute o que não funcionou.
5. Não use as palavras "robusto", "abrangente", "comprehensive", "Adicionalmente", "Ademais", "Cumpre destacar", "É importante ressaltar", "Em suma", "Por fim, conclui-se". Essas frases gritam IA.
6. Não estruture tudo em listas perfeitas com bullets simétricos. Misture texto corrido com listas onde fizer sentido.
7. Nada de emoji em lugar nenhum, nem no README, nem em comentário.
8. Comentários no código podem ser informais às vezes, tipo "# split estratificado pra não desbalancear" em vez de "# Realizando a divisão estratificada do conjunto de dados".
9. No README, na seção de motivação, escrever em primeira pessoa de forma natural. Algo como: "escolhi esse tema porque venho acompanhando o impacto da desinformação nas eleições e queria ver se um modelo simples já entrega resultado razoável".
10. Citações acadêmicas sim, no formato ABNT, isso o professor cobra. Mas sem floreio.

# Como vou subir

Depois que você gerar tudo, vou:
1. Criar repositório público no GitHub com nome `poc-fake-news-pln`
2. Subir os arquivos
3. Mandar o link pro professor

Então o README precisa estar bem escrito, porque é a primeira coisa que ele vai ver.

# Formato da entrega da sua resposta

Entregue na ordem:

1. Estrutura de pastas do projeto em árvore
2. Conteúdo completo do `README.md`
3. Conteúdo completo do `requirements.txt`
4. Conteúdo completo do `.gitignore`
5. Conteúdo completo do notebook em formato Python script com células marcadas (`# %% [markdown]` e `# %%`), pra eu colar no VSCode e exportar pra `.ipynb`. Já com todos os comentários, células markdown explicativas, e código rodável.
6. Comandos do Git pra inicializar, commitar e subir no GitHub (assumindo que eu já criei o repo vazio em github.com/CarlosMateoBruno/poc-fake-news-pln).

Não pule nenhuma das 6 partes. Não resuma nada com "..." ou "deixo a critério do aluno". Quero o projeto pronto pra rodar.

Comece pela árvore de pastas.
```

---

## Notas de uso

- O prompt assume usuário `CarlosMateoBruno` no GitHub, se for diferente, troque na seção final.
- Depois de gerar, rodar o notebook localmente antes de subir, pra garantir que não tem erro de import nem path quebrado.
- Antes do commit, conferir manualmente se sobrou algum travessão (—) no README ou nos comentários. Buscar e substituir por vírgula.
- Conferir também se a IA não colocou emoji em nenhum lugar, principalmente no README.
- Repo público pra compartilhar com o professor, sem dados sensíveis.

## Checklist pré-entrega

- [ ] Notebook roda do começo ao fim sem erro
- [ ] README sem travessão e sem emoji
- [ ] requirements.txt com versões pinadas
- [ ] Dataset não commitado, apenas instruções de download
- [ ] Citação do Fake.br Corpus em ABNT
- [ ] Repo público no GitHub
- [ ] Link colado no Moodle ou enviado pro professor

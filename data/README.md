# Dataset

Esse projeto usa o **Fake.br Corpus**, de Monteiro et al. (2018).

## Download

1. Clonar o repositório oficial dentro dessa pasta `data/`:

```bash
cd data
git clone https://github.com/roneysco/Fake.br-Corpus.git
```

2. A estrutura final que o notebook espera é:

```
data/
└── Fake.br-Corpus/
    └── full_texts/
        ├── fake/    # 3600 arquivos .txt
        └── true/    # 3600 arquivos .txt
```

O notebook lê os textos a partir de `../data/Fake.br-Corpus/full_texts/`.

## Citação

MONTEIRO, R. A.; SANTOS, R. L. S.; PARDO, T. A. S.; ALMEIDA, T. A.; RUIZ, E. E. S.; VALE, O. A. Contributions to the Study of Fake News in Portuguese: New Corpus and Automatic Detection Results. In: Computational Processing of the Portuguese Language (PROPOR 2018), LNCS vol. 11122. Springer, 2018.

# Revisão – Limpeza e Preparação de Dados

**Autor:** Ronald Machado  
**Disciplina:** Ciência de Dados  
**Curso:** Engenharia de Software

## Sobre a atividade

Este repositório apresenta cinco exercícios de limpeza e preparação de dados, organizados de acordo com os esqueletos propostos no material da disciplina. Cada exercício possui um arquivo Python próprio, com o código, as comparações e as respostas às perguntas de reflexão nos comentários.

## Exercícios

| Arquivo | Tema | Dataset |
|---|---|---|
| `exercicio_1_valores_ausentes.py` | Identificação, remoção e imputação de valores ausentes | Melbourne Housing Snapshot |
| `exercicio_2_duplicatas.py` | Duplicatas exatas, chaves de identificação e tratamento | KPMG Customer Transactions |
| `exercicio_3_padronizacao_escalonamento.py` | Padronização de textos e escalonamento Min-Max, Z-score e robusto | Adult Census Income |
| `exercicio_4_outliers.py` | Detecção por Z-score e IQR e comparação de tratamentos | Diamonds |
| `exercicio_5_pipeline_integrado.py` | Diagnóstico e aplicação de um pipeline de limpeza | SHEIN E-Commerce (amostra de eletrodomésticos) |

## Estrutura do repositório

```text
ciencia-de-dados-limpeza-preparacao/
├── README.md
├── exercicio_1_valores_ausentes.py
├── exercicio_2_duplicatas.py
├── exercicio_3_padronizacao_escalonamento.py
├── exercicio_4_outliers.py
└── exercicio_5_pipeline_integrado.py
```

## Bibliotecas utilizadas

Os exercícios utilizam Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn e openpyxl. Para instalar as bibliotecas necessárias:

```bash
python -m pip install pandas numpy matplotlib seaborn scikit-learn openpyxl
```


## Fontes dos dados

- [Melbourne Housing Snapshot](https://www.kaggle.com/datasets/dansbecker/melbourne-housing-snapshot)
- [KPMG Customer Transaction Dataset](https://www.kaggle.com/datasets/archit9406/customer-transaction-dataset)
- [Adult Census Income – UCI](https://archive.ics.uci.edu/dataset/2/adult)
- [Diamonds – seaborn-data](https://github.com/mwaskom/seaborn-data/blob/master/diamonds.csv)
- [SHEIN E-Commerce Data](https://www.kaggle.com/datasets/oleksiimartsiuk/e-commerce-data-shein)

# IMPORTAÇÕES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

# NOMES DAS COLUNAS (CONSULTA À DOCUMENTAÇÃO DO DATASET)
colunas = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num',
    'marital-status', 'occupation', 'relationship', 'race', 'sex',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
]

# CARREGAMENTO DOS DADOS
# adult.data não tem cabeçalho no próprio arquivo.
from pathlib import Path
import zipfile
if Path('adult.data').exists():
    origem = 'adult.data'
else:
    compactado = zipfile.ZipFile('adult.zip')
    origem = compactado.open('adult.data')
df = pd.read_csv(origem, names=colunas, header=None,
                 skipinitialspace=True)
print('Tamanho da base:', df.shape)
print(df.head())

# INSPEÇÃO DAS COLUNAS CATEGÓRICAS
categoricas = df.select_dtypes(include='object').columns
for coluna in categoricas:
    print(f'\n{coluna} - valores mais comuns:')
    print(df[coluna].value_counts(dropna=False).head(5))

# PADRONIZAÇÃO DOS TEXTOS (ANTES E DEPOIS)
antes = df[categoricas].nunique(dropna=False)
df_pad = df.copy()
for coluna in categoricas:
    # Padronizo letras e tiro espaços extras para evitar categorias repetidas.
    df_pad[coluna] = (df_pad[coluna].astype('string').str.strip()
                      .str.lower().str.replace(r'\s+', ' ', regex=True))
depois = df_pad[categoricas].nunique(dropna=False)
print('\nQuantidade de categorias antes e depois:')
print(pd.DataFrame({'antes': antes, 'depois': depois}))

# CONVERSÃO DOS MARCADORES DE AUSÊNCIA
# '?' não é profissão nem país: é ausência de informação.
df_pad[categoricas] = df_pad[categoricas].replace({'?': pd.NA, '': pd.NA})
print('\nValores ausentes por coluna:')
print(df_pad.isna().sum()[df_pad.isna().sum() > 0])

# ESCALONAMENTO - MIN-MAX
numericas = df_pad.select_dtypes(include='number').columns
# Preencho eventuais valores numéricos vazios para aplicar os escalonadores.
df_pad[numericas] = df_pad[numericas].fillna(df_pad[numericas].median())
df_minmax = df_pad.copy()
df_minmax[numericas] = MinMaxScaler().fit_transform(df_pad[numericas])
print('\nMin-Max (menor e maior valor):')
print(df_minmax[numericas].agg(['min', 'max']))

# ESCALONAMENTO - Z-SCORE
df_zscore = df_pad.copy()
df_zscore[numericas] = StandardScaler().fit_transform(df_pad[numericas])
print('\nZ-score (média e desvio):')
print(df_zscore[numericas].agg(['mean', 'std']).round(2))

# ESCALONAMENTO - ROBUSTO
df_robusto = df_pad.copy()
df_robusto[numericas] = RobustScaler().fit_transform(df_pad[numericas])
print('\nRobusto (mediana):')
print(df_robusto[numericas].median().round(2))

# COMPARAÇÃO ENTRE AS TÉCNICAS
# Vou usar capital-gain porque tem muitos zeros e alguns valores muito altos.
coluna = 'capital-gain'
comparacao = pd.DataFrame({
    'original': df_pad[coluna].describe(),
    'min_max': df_minmax[coluna].describe(),
    'z_score': df_zscore[coluna].describe(),
    'robusto': df_robusto[coluna].describe()
})
print('\nComparação de capital-gain:')
print(comparacao.round(2))
print('Porcentagem de zeros em capital-gain:',
      round((df_pad[coluna] == 0).mean() * 100, 2), '%')
print('Intervalo interquartil de capital-gain:',
      df_pad[coluna].quantile(0.75) - df_pad[coluna].quantile(0.25))
fig, eixos = plt.subplots(1, 3, figsize=(12, 3))
for eixo, (nome, dados) in zip(eixos, [
    ('Min-Max', df_minmax[coluna]),
    ('Z-score', df_zscore[coluna]),
    ('Robusto', df_robusto[coluna])
]):
    eixo.hist(dados, bins=25)
    eixo.set_title(nome)
plt.tight_layout()
plt.show()

# RESPOSTAS ÀS PERGUNTAS DE REFLEXÃO
# Na base consultada, 91,67% dos valores de capital-gain são zero.
# 1) O df.isna().sum() pode mostrar zero mesmo com dados faltando porque
#    o arquivo guarda alguns campos vazios como o texto '?'. Antes de
#    trocar '?' por ausência (NA), o pandas vê uma categoria normal.
# 2) Em capital-gain, muitos valores são zero e poucos são altos. O Min-Max
#    comprime boa parte dos dados perto de zero. O Z-score também sofre
#    influência dos valores extremos. O RobustScaler usa mediana e IQR,
#    mas, se o IQR for zero, também pode não separar bem os valores.
#    Por isso imprimi as estatísticas e fiz os gráficos para comparar.
# 3) Não é obrigatório escalonar toda coluna numérica. Por exemplo,
#    fnlwgt é um peso amostral utilizado no censo para representar pessoas
#    semelhantes na população. Dependendo do modelo, pode servir como
#    peso da amostra e não como uma variável comum de previsão.


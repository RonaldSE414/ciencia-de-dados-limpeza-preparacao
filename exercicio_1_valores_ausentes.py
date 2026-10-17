# IMPORTAÇÕES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import zipfile
import io

# CARREGAMENTO DOS DADOS
arquivo = Path('melb_data.csv')
if arquivo.exists():
    df = pd.read_csv(arquivo)
else:
    arquivo = Path('melb_data.csv.zip')
    with zipfile.ZipFile(arquivo) as compactado:
        df = pd.read_csv(compactado.open('melb_data.csv'))
print('Primeiras linhas:')
print(df.head())

# DIAGNÓSTICO INICIAL
print('\nTamanho da base (linhas, colunas):', df.shape)
print('\nTipos das colunas:')
print(df.dtypes)

# RESUMO DE VALORES AUSENTES (QUANTIDADE E %)
ausentes = pd.DataFrame({
    'quantidade': df.isna().sum(),
    'porcentagem': (df.isna().mean() * 100).round(2)
}).sort_values('porcentagem', ascending=False)
print('\nResumo dos dados ausentes:')
print(ausentes)

# VISUALIZAÇÃO DO PADRÃO DE AUSÊNCIA
# O gráfico de barras mostra em quais colunas faltam mais informações.
ausentes.query('quantidade > 0')['porcentagem'].plot(kind='bar', figsize=(10, 4))
plt.title('Porcentagem de dados ausentes por coluna')
plt.ylabel('Valores ausentes (%)')
plt.tight_layout()
plt.show()

# A imagem abaixo mostra se as ausências aparecem juntas nas mesmas linhas.
plt.figure(figsize=(10, 4))
plt.imshow(df.isna(), aspect='auto', interpolation='nearest')
plt.title('Onde aparecem os dados ausentes')
plt.xlabel('Colunas')
plt.ylabel('Linhas')
plt.xticks(range(len(df.columns)), df.columns, rotation=90)
plt.tight_layout()
plt.show()

# ESTRATÉGIA 1 - REMOÇÃO
df_remocao = df.copy()
df_remocao = df_remocao.dropna()
print('\nLinhas originais:', len(df))
print('Linhas após remover registros incompletos:', len(df_remocao))
# A remoção reduz a base; por isso comparei o total de linhas.

# ESTRATÉGIA 2 - IMPUTAÇÃO
df_imputacao = df.copy()
for coluna in df_imputacao.columns:
    if df_imputacao[coluna].isna().sum() == 0:
        continue
    if pd.api.types.is_numeric_dtype(df_imputacao[coluna]):
        # Usei a mediana para os números por causa dos valores mais extremos.
        valor = df_imputacao[coluna].median()
    else:
        # Para texto, o valor mais comum preserva uma categoria já existente.
        moda = df_imputacao[coluna].mode(dropna=True)
        valor = moda.iloc[0] if not moda.empty else 'Não informado'
    if pd.notna(valor):
        df_imputacao[coluna] = df_imputacao[coluna].fillna(valor)
print('Total de valores ausentes após imputação:', df_imputacao.isna().sum().sum())

# COMPARAÇÃO ENTRE AS ESTRATÉGIAS
comparacao = pd.DataFrame({
    'estrategia': ['Original', 'Remoção', 'Imputação'],
    'linhas': [len(df), len(df_remocao), len(df_imputacao)],
    'dados_ausentes': [df.isna().sum().sum(), df_remocao.isna().sum().sum(),
                      df_imputacao.isna().sum().sum()]
})
print('\nComparação:')
print(comparacao)
for coluna in ['Price', 'Landsize']:
    if coluna in df.columns:
        print(f'\n{coluna} - médias (original, remoção, imputação):',
              round(df[coluna].mean(), 2),
              round(df_remocao[coluna].mean(), 2),
              round(df_imputacao[coluna].mean(), 2))
        print(f'{coluna} - medianas (original, remoção, imputação):',
              round(df[coluna].median(), 2),
              round(df_remocao[coluna].median(), 2),
              round(df_imputacao[coluna].median(), 2))

# RESPOSTAS ÀS PERGUNTAS DE REFLEXÃO
# BuildingArea é a coluna com mais lacunas, aproximadamente 47,5%.
# YearBuilt também tem muitos campos vazios, perto de 39,6%.
# 1) Mais de 50% de dados faltando é bastante coisa, mas não significa que
#    eu deva apagar a coluna sem pensar. Primeiro vejo se ela é importante
#    e se é possível recuperar alguma informação com segurança.
# 2) Em Price e Landsize, valores muito altos podem puxar a média.
#    Nesse caso, a mediana costuma ser mais segura. Eu compararia os dois
#    resultados antes de decidir. Se Price for o alvo de uma previsão,
#    evitaria preencher preços inventados.
# 3) A própria ausência pode ser útil quando acontece mais em determinado
#    tipo de imóvel. Isso pode indicar algum padrão de cadastro ou coleta.
#    Por isso fiz o gráfico antes de escolher como tratar os dados.


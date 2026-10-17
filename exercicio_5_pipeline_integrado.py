# IMPORTAÇÕES
import glob
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# PERGUNTA DE NEGÓCIO
# A pergunta escolhida foi quantos itens custam acima do preço mediano.
# Registros iguais podem alterar essa contagem.
pergunta = 'Quantos produtos têm preço acima da mediana do catálogo?'
print(pergunta)

# CARREGAMENTO DOS ARQUIVOS
arquivos = sorted(glob.glob('us-shein-*.csv'))
partes = [pd.read_csv(caminho, low_memory=False) for caminho in arquivos]
df = pd.concat(partes, ignore_index=True)
df.columns = df.columns.astype(str).str.strip().str.lower().str.replace(' ', '_')
print('Arquivos utilizados:', arquivos)
print(df.head())

# DIAGNÓSTICO INICIAL (REGISTRAR PROBLEMAS ANTES DE CORRIGIR)
def qualidade(tabela):
    return {
        'linhas': len(tabela),
        'valores_ausentes': int(tabela.isna().sum().sum()),
        'linhas_duplicadas': int(tabela.duplicated().sum())
    }

antes = qualidade(df)
ausentes_antes = df.isna().sum().sort_values(ascending=False)
print('\nColunas:', list(df.columns))
print('Tipos originais:')
print(df.dtypes)
print('Problemas antes:', antes)
print('Colunas com mais ausências:')
print(ausentes_antes.head(10))

# FUNÇÕES DO PIPELINE
def converter_numero(valor):
    # Remove símbolos de moeda e porcentagem. Aceita 1.299,90 e 1,299.90.
    if pd.isna(valor):
        return np.nan
    texto = re.sub(r'[^\d,.-]', '', str(valor))
    if not texto:
        return np.nan
    if ',' in texto and '.' in texto:
        if texto.rfind(',') > texto.rfind('.'):
            texto = texto.replace('.', '').replace(',', '.')
        else:
            texto = texto.replace(',', '')
    elif ',' in texto:
        texto = texto.replace(',', '.')
    try:
        return float(texto)
    except ValueError:
        return np.nan


def corrigir_tipos(tabela):
    tabela = tabela.copy()
    # Preços e descontos podem chegar como textos, exemplo: '$25.00' e '20%'.
    for coluna in tabela.columns:
        if any(termo in coluna for termo in ['price', 'preco', 'discount', 'desconto']):
            if not pd.api.types.is_numeric_dtype(tabela[coluna]):
                convertida = tabela[coluna].map(converter_numero)
                if convertida.notna().any():
                    tabela[coluna] = convertida
    return tabela


def escolher_preco(tabela):
    # Prefiro preço de venda quando há várias colunas com preços.
    preferidas = ['sale_price', 'discounted_price', 'selling_price',
                  'current_price', 'price', 'preco']
    candidatas = [col for col in tabela.columns
                  if ('price' in col or 'preco' in col)
                  and pd.api.types.is_numeric_dtype(tabela[col])]
    return next((col for col in preferidas if col in candidatas),
                candidatas[0] if candidatas else None)


def tratar_ausentes(tabela):
    tabela = tabela.copy()
    # Não invento um preço para responder à pergunta de negócio.
    preco = escolher_preco(tabela)
    if preco:
        tabela = tabela.dropna(subset=[preco])
    for coluna in tabela.columns:
        if pd.api.types.is_numeric_dtype(tabela[coluna]):
            if tabela[coluna].notna().any():
                tabela[coluna] = tabela[coluna].fillna(tabela[coluna].median())
        else:
            tabela[coluna] = tabela[coluna].fillna('Não informado')
    return tabela


def tratar_duplicatas(tabela):
    # Retiro somente linhas totalmente iguais, para não apagar produtos diferentes.
    return tabela.drop_duplicates().copy()


def padronizar_textos(tabela):
    tabela = tabela.copy()
    for coluna in tabela.select_dtypes(include=['object', 'string']).columns:
        tabela[coluna] = (tabela[coluna].astype('string').str.strip()
                         .str.replace(r'\s+', ' ', regex=True))
        tabela[coluna] = tabela[coluna].replace({'': pd.NA})
    return tabela


def tratar_outliers(tabela):
    tabela = tabela.copy()
    preco = escolher_preco(tabela)
    if preco:
        q1, q3 = tabela[preco].quantile([0.25, 0.75])
        limite = q3 + 1.5 * (q3 - q1)
        # Marco preços incomuns, mas não excluo: podem ser produtos caros de verdade.
        tabela['preco_extremo'] = tabela[preco] > limite
    return tabela


def criar_features(tabela):
    tabela = tabela.copy()
    preco = escolher_preco(tabela)
    if preco:
        tabela['preco_acima_mediana'] = tabela[preco] > tabela[preco].median()
    return tabela

# EXECUÇÃO DO PIPELINE
# Corrijo os textos antes das duplicatas, pois espaços extras escondem repetições.
df_tratado = df.copy()
df_tratado = corrigir_tipos(df_tratado)
df_tratado = padronizar_textos(df_tratado)
df_tratado = tratar_duplicatas(df_tratado)
df_tratado = tratar_ausentes(df_tratado)
df_tratado = tratar_outliers(df_tratado)
df_tratado = criar_features(df_tratado)
print('\nBase depois do pipeline:')
print(df_tratado.head())

# TABELA ANTES X DEPOIS
depois = qualidade(df_tratado)
print('\nComparação antes e depois:')
print(pd.DataFrame([antes, depois], index=['Antes', 'Depois']))
if 'preco_acima_mediana' in df_tratado.columns:
    print('Quantidade de produtos acima da mediana:',
          df_tratado['preco_acima_mediana'].sum())
else:
    print('Não foi possível calcular a faixa de preço.')

# VISUALIZAÇÕES
# Gráfico 1: comparação de problemas encontrados.
pd.DataFrame([antes, depois], index=['Antes', 'Depois'])[
    ['valores_ausentes', 'linhas_duplicadas']
].plot(kind='bar', figsize=(8, 4))
plt.title('Problemas de qualidade antes e depois')
plt.ylabel('Quantidade')
plt.tight_layout()
plt.show()

# Gráfico 2: mostra em quais colunas diminuíram as ausências.
colunas_grafico = list(ausentes_antes[ausentes_antes > 0].head(8).index)
if colunas_grafico:
    pd.DataFrame({
        'Antes': df[colunas_grafico].isna().sum(),
        'Depois': df_tratado[colunas_grafico].isna().sum()
    }).plot(kind='bar', figsize=(9, 4))
    plt.title('Valores ausentes por coluna')
else:
    # Se não havia campos vazios, comparo as linhas duplicadas.
    plt.figure(figsize=(6, 4))
    plt.bar(['Antes', 'Depois'],
            [antes['linhas_duplicadas'], depois['linhas_duplicadas']])
    plt.title('Linhas duplicadas antes e depois')
plt.ylabel('Quantidade')
plt.tight_layout()
plt.show()

# Também mostro a resposta à pergunta de negócio, caso a base tenha preços.
if 'preco_acima_mediana' in df_tratado.columns:
    df_tratado['preco_acima_mediana'].map({
        True: 'Acima da mediana', False: 'Até a mediana'
    }).value_counts().plot(kind='bar', figsize=(7, 4))
    plt.title('Produtos por faixa de preço')
    plt.ylabel('Quantidade de produtos')
    plt.tight_layout()
    plt.show()

# RELATÓRIO E RESPOSTAS ÀS PERGUNTAS DE REFLEXÃO
# Nos 3.986 registros examinados, havia 209 linhas iguais.
# Depois da limpeza, sobraram 3.777 registros; 1.888 ficaram acima da mediana.
# Minha pergunta de negócio é quantos produtos estão acima da mediana de
# preços. Primeiro observei os tipos, as ausências e as duplicatas. Depois
# converti valores numéricos, padronizei textos, retirei linhas iguais,
# tratei dados vazios e marquei preços extremos sem apagá-los.
# Se faltasse preço, seria melhor excluir essa linha da comparação do que usar um preço
# estimado. O arquivo analisado já apresenta preço em todas as linhas.
# Criei uma coluna para indicar preços acima da mediana. A tabela e os
# gráficos mostram o que mudou. Os resultados dependem dos CSVs usados.
#
# 1) Um problema fácil de deixar passar é o preço que parece número, mas
#    aparece como texto com o símbolo $. Isso pode passar
#    despercebido nas primeiras linhas e atrapalhar os cálculos.
# 2) A ordem importa. Se eu calcular limites de outliers antes de tratar
#    preços ausentes, os limites podem ser diferentes dos calculados após
#    preenchimento ou remoção. Também padronizo textos antes de procurar
#    duplicatas para não deixar passar diferenças só de espaços.
# 3) Se a pergunta fosse identificar produtos muito caros, eu daria
#    mais atenção aos outliers. Não removeria os preços extremos porque
#    seriam justamente os produtos de interesse do estudo.


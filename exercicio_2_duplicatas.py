# IMPORTAÇÕES
import pandas as pd
import numpy as np
from pathlib import Path
import zipfile
import io

# CARREGAMENTO DOS DADOS
arquivo = Path('KPMG_VI_New_raw_data_update_final.xlsx')
if arquivo.exists():
    df = pd.read_excel(arquivo, sheet_name='Transactions', header=1)
else:
    with zipfile.ZipFile('archive.zip') as compactado:
        df = pd.read_excel(io.BytesIO(compactado.read('KPMG_VI_New_raw_data_update_final.xlsx')),
                           sheet_name='Transactions', header=1)
df.columns = df.columns.astype(str).str.strip()
df = df.dropna(how='all')
print('Primeiras linhas:')
print(df.head())

# CONFERÊNCIA DO CABEÇALHO E DOS TIPOS
print('\nTamanho da base:', df.shape)
print('Colunas:', list(df.columns))
print('Tipos dos dados:')
print(df.dtypes)

# DUPLICATAS EXATAS
duplicatas_exatas = df.duplicated().sum()
print('\nQuantidade de linhas totalmente repetidas:', duplicatas_exatas)
print(df[df.duplicated(keep=False)].head(10))

# DUPLICATAS POR SUBCONJUNTO DE COLUNAS
# Duas compras parecidas não são obrigatoriamente a mesma venda.
chaves = ['customer_id', 'transaction_date', 'product_id']
if all(coluna in df.columns for coluna in chaves):
    repetidas_chaves = df.duplicated(subset=chaves).sum()
    print('\nMesmo cliente, mesma data e mesmo produto:', repetidas_chaves)
    print(df[df.duplicated(subset=chaves, keep=False)].head(10))
else:
    print('\nNão encontrei todas as colunas para comparar as compras.')

# UNICIDADE DOS IDENTIFICADORES
if 'transaction_id' in df.columns:
    print('\nIDs de transação repetidos:', df['transaction_id'].duplicated().sum())
    print(df[df.duplicated('transaction_id', keep=False)].head(10))
else:
    print('\nA coluna transaction_id não foi encontrada.')

# ESTRATÉGIA DE TRATAMENTO E IMPACTO
# Retirei as cópias exatas, mas mantive compras parecidas com IDs diferentes.
df_tratado = df.drop_duplicates().copy()
print('\nRegistros antes:', len(df))
print('Registros depois:', len(df_tratado))
print('Registros removidos:', len(df) - len(df_tratado))
if 'transaction_id' in df_tratado.columns:
    print('IDs repetidos que ainda precisam de análise:',
          df_tratado['transaction_id'].duplicated().sum())

# RESPOSTAS ÀS PERGUNTAS DE REFLEXÃO
# A planilha tem 20.000 transações e nenhum transaction_id repetido.
# Também não encontrei linhas exatamente iguais.
# 1) Não encontrei linhas idênticas nessa planilha, mas pode haver registros
#    parecidos com datas, espaços ou identificadores diferentes.
#    Comparando cliente, data e produto, apareceu um caso repetido.
# 2) Duas linhas iguais nem sempre são erro. Por exemplo, dois clientes
#    podem comprar o mesmo produto pelo mesmo valor no mesmo dia.
#    Sem uma identificação confiável, eu não apagaria esses registros.
# 3) A escolha da chave é uma decisão de negócio porque precisamos saber o
#    que representa uma transação única. Isso faz parte do Entendimento do
#    Negócio no CRISP-DM: evitar contar uma venda duas vezes e também
#    evitar eliminar uma venda verdadeira.


# IMPORTAÇÕES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# CARREGAMENTO DOS DADOS
# Dados: diamonds, da coleção seaborn-data.
from pathlib import Path
if Path('diamonds.csv').exists():
    df = pd.read_csv('diamonds.csv')
else:
    df = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/diamonds.csv')
print('Tamanho da base:', df.shape)
print(df.head())

# ESTATÍSTICAS DESCRITIVAS E VALORES IMPOSSÍVEIS
print('\nResumo das colunas numéricas:')
print(df.describe())
# Um diamante não pode ter comprimento, largura ou altura igual a zero.
erros_medida = (df[['x', 'y', 'z']] <= 0).any(axis=1)
erros_medida |= (df['carat'] <= 0) | (df['price'] <= 0)
print('\nRegistros com medidas impossíveis:', erros_medida.sum())
df_valido = df.loc[~erros_medida].copy()

# DETECÇÃO - Z-SCORE
numericas = df_valido.select_dtypes(include='number')
medias = numericas.mean()
desvios = numericas.std().replace(0, np.nan)
z = ((numericas - medias) / desvios).abs()
outliers_z = (z > 3).any(axis=1)
print('\nRegistros sinalizados pelo Z-score:', outliers_z.sum())

# DETECÇÃO - IQR
q1 = numericas.quantile(0.25)
q3 = numericas.quantile(0.75)
iqr = q3 - q1
limite_inferior = q1 - 1.5 * iqr
limite_superior = q3 + 1.5 * iqr
outliers_iqr = ((numericas < limite_inferior) |
                (numericas > limite_superior)).any(axis=1)
print('Registros sinalizados pelo IQR:', outliers_iqr.sum())
print('Casos encontrados pelos dois métodos:', (outliers_z & outliers_iqr).sum())

# VISUALIZAÇÕES
fig, eixos = plt.subplots(1, 3, figsize=(13, 4))
eixos[0].hist(df['price'], bins=40)
eixos[0].set_title('Distribuição dos preços')
sns.boxplot(x=df['price'], ax=eixos[1])
eixos[1].set_title('Preço dos diamantes')
sns.boxplot(x=df['carat'], ax=eixos[2])
eixos[2].set_title('Peso em quilates')
plt.tight_layout()
plt.show()

# ERRO X VALOR EXTREMO LEGÍTIMO
# Medidas físicas iguais a zero parecem erros de cadastro.
# Preço muito alto pode ser normal no caso de um diamante raro.
print('\nMaiores preços encontrados:')
print(df.nlargest(5, 'price')[['carat', 'price', 'x', 'y', 'z']])

# TRATAMENTO - REMOÇÃO / CAPPING / TRANSFORMAÇÃO
# Faço três alternativas para comparar, sem alterar df original.
# Remoção: tiro registros com medidas impossíveis e os sinalizados pelo IQR.
df_remocao = df_valido.loc[~outliers_iqr].copy()

# Capping: mantenho os registros, mas limito o preço ao intervalo do IQR.
# Isso é apenas um teste; preços altos podem ser verdadeiros.
df_capping = df_valido.copy()
q1_preco = df_capping['price'].quantile(0.25)
q3_preco = df_capping['price'].quantile(0.75)
limite_preco = q3_preco + 1.5 * (q3_preco - q1_preco)
df_capping['price'] = df_capping['price'].clip(upper=limite_preco)

# Transformação: crio o logaritmo do preço e guardo o preço original.
df_log = df_valido.copy()
df_log['preco_log'] = np.log1p(df_log['price'])

# COMPARAÇÃO ENTRE AS ESTRATÉGIAS
comparacao = pd.DataFrame({
    'estrategia': ['Original válido', 'Remoção', 'Capping', 'Log'],
    'linhas': [len(df_valido), len(df_remocao), len(df_capping), len(df_log)],
    'media_preco': [df_valido['price'].mean(), df_remocao['price'].mean(),
                    df_capping['price'].mean(), df_log['price'].mean()],
    'desvio_preco': [df_valido['price'].std(), df_remocao['price'].std(),
                    df_capping['price'].std(), df_log['price'].std()]
})
print('\nComparação dos tratamentos (preço original):')
print(comparacao.round(2).to_string(index=False))
print('O log foi criado em outra coluna, preservando o preço original.')
print('Média do preço em escala log:', round(df_log['preco_log'].mean(), 2))

# Comparo a distribuição do preço original com as duas alternativas.
fig, eixos = plt.subplots(1, 3, figsize=(12, 3))
eixos[0].hist(df_valido['price'], bins=35)
eixos[0].set_title('Original')
eixos[1].hist(df_capping['price'], bins=35)
eixos[1].set_title('Capping')
eixos[2].hist(df_log['preco_log'], bins=35)
eixos[2].set_title('Log do preço')
plt.tight_layout()
plt.show()

# RESPOSTAS ÀS PERGUNTAS DE REFLEXÃO
# 1) O Z-score funciona melhor quando a distribuição é aproximadamente
#    simétrica, especialmente próxima da normal. O histograma de price
#    permite ver que os preços não têm esse formato: há cauda à direita.
#    Por isso não dá para usar só o Z-score sem avaliar o contexto.
# 2) Um diamante com mais de 4 quilates pode ser só um produto raro e caro.
#    Se a análise for de diamantes raros, eu manteria esse valor. Já uma
#    medida física igual a zero parece um erro de registro.
# 3) Se uma loja quer descobrir quais são os diamantes mais caros do seu
#    catálogo, remover os maiores preços esconderia justamente esse grupo.
#    Por isso comparei remoção, capping e transformação antes de escolher.


# Na análise desta base, encontrei 20 registros com medidas impossíveis.
# Depois de separá-los, o Z-score sinalizou 2.334 linhas e o IQR, 6.396.
# Os dois métodos encontraram 2.333 casos em comum.

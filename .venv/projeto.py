import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

df_amazon = pd.read_csv('https://raw.githubusercontent.com/dastias/Arquivos-Banco-de-Dados/refs/heads/main/amazon.csv')
print(df_amazon)

# Cabeçalho
print(df_amazon.head())
print(df_amazon.info())
print(df_amazon.shape)

# Observando as colunas para saber seu tipo(type), Qtd de colunas e qtd de linhas.
print(df_amazon.columns)
print(f"Linhas : {df_amazon.shape[0]}")
print(f"Colunas : {df_amazon.shape[1]}")

# Descrição dos dados
# Variáveis Quantitativas Contínuas
quant_cont = ['discounted_price', 'actual_price', 'discount_percentage', 'rating']

# Variáveis Quantitativas Discretas
quant_dis = ['rating_count']

# Quantitativa em geral
quant = quant_cont + quant_dis

# Variáveis Qualitativas Nominais
qual_nominal = [
    'product_id', 'product_name', 'category', 'about_product',
    'user_id', 'user_name', 'review_id', 'review_title',
    'review_content', 'img_link', 'product_link'
]
print("--- Classificação das Features ---")
print("Quantitativas Contínuas:", quant_cont)
print("Quantitativas Discretas:", quant_dis)
print("Qualitativas Nominais:", qual_nominal)

# Trocando os tipos de dados de 'dicounted_price' e 'actual_price' para Float.
df_amazon['discounted_price'] = df_amazon['discounted_price'].str.replace('₹', '')
df_amazon['discounted_price'] = df_amazon['discounted_price'].str.replace(",", '')
df_amazon['discounted_price'] = df_amazon['discounted_price'].astype('float64')

df_amazon['actual_price'] = df_amazon['actual_price'].str.replace('₹', '')
df_amazon['actual_price'] = df_amazon['actual_price'].str.replace(",", '')
df_amazon['actual_price'] = df_amazon['actual_price'].astype('float64')

## Nota-se que esse dataset é o e-commerce da amazon da India, então os valores da coluna,
## 'actual_price' e 'discounted_price' tem o simbolo da moeda, '₹' que iremos remove-los.

# Alterando o tipo de dados e os valores na porcentagem de desconto
df_amazon['discount_percentage'] = df_amazon['discount_percentage'].str.replace('%', ''). astype('float64')
df_amazon['discount_percentage'] = df_amazon['discount_percentage'] / 100
df_amazon.info()

# Encontrando uma sequência incomum na coluna de classificação
df_amazon['rating'].value_counts()

# Analisando a linha 'Estranha'
df_amazon.query('rating == "|"')
# Observe que tem 1 linha que na coluna 'rating' que está errada

# Encontrei a avaliação do produto no site 'amazon.in' 
# pesquisando pelo o ID e a Avaliação é de 4.0, portanto vou atribuir esse valor na linha.

# Observe que a coluna 'rating' ainda está com o datatype de objeto
df_amazon['rating'] = df_amazon['rating'].str.replace('|', '4.0').astype('float64')

# Trocando o datatype da coluna 'rating_count' para float
df_amazon['rating_count'] = df_amazon['rating_count'].str.replace(',', '').astype('float64')
df_amazon.info()

# Valores ausentes.
df_amazon.isnull().sum().sort_values(ascending=False)

# Porcentagem de valores ausentes das colunas.
round(df_amazon.isnull().sum() / len(df_amazon) * 100, 2).sort_values(ascending=False)

# Criação de uma coluna chamada 'final_price' que vai ser o 'actual_price' - 'discounted_price'
df_amazon['final_price'] = df_amazon['actual_price'] - df_amazon['discounted_price']
df_amazon['final_price'] = df_amazon['final_price'].astype('float64')
df_amazon.info()


# Plotando os valores ausentes/nulos de cada coluna.
plt.figure(figsize=(10, 8))
sns.heatmap(df_amazon.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.show()

# Plote da porcentagem dos valores ausentes/nulos.
plt.figure(figsize=(10, 8))
missing_percentage = df_amazon.isnull().sum()/len(df_amazon)*100
missing_percentage.plot(kind='bar')
plt.xlabel('Colunas')
plt.ylabel('Porcentagem')
plt.title('Porcentagem de valores ausentes em cada coluna ')
plt.show()

# Estamos visualizando apenas as linhas onde existem valores nulos/ausentes na coluna.

df_amazon[df_amazon['rating_count'].isnull()].head()

# Imputando a mediana nos valores ausentes.
df_amazon['rating_count'] = df_amazon.rating_count.fillna(value=df_amazon['rating_count'].median())

df_amazon.isnull().sum().sort_values(ascending=False)

# Encontrando duplicatas.
df_amazon.duplicated().sum()

# Analisando as colunas do conjunto inteiro com 'any' para retornar em booleano
any_duplicates = df_amazon.duplicated(subset=['product_id', 'product_name', 'category', 'discounted_price',
       'actual_price', 'discount_percentage', 'rating', 'rating_count',
       'about_product', 'user_id', 'user_name', 'review_id', 'review_title',
       'review_content', 'img_link', 'product_link']).any()
print(f'Olhando se tem duplicadas em booleano : {any_duplicates}')

# Dropando colunas que não fazem sentido para analise no caso,  'img_link' que são imagens de produtos e  'product_link' que é o link do URL do produto .
colunas_drop = ['img_link', 'product_link']
df_amazon = df_amazon.drop(columns=colunas_drop)
df_amazon.info()

# concluida a limpeza do conjunto de dados, removendo os valores nulos, ausentes e duplicados.

# Desconbrindo se há outliers nas variavéis quantitativas.
plt.figure(figsize=(15, 10))
for i, column in enumerate(quant):
    plt.subplot(2, 3, i + 1)
    sns.boxplot(y=df_amazon[column])
    plt.title(f'Box Plot de {column}')
    plt.ylabel(column)
plt.tight_layout()
plt.show()


""" 
1- 'discounted_price' e 'actual_price': Ambos os box plots mostram uma grande quantidade de outliers na parte superior, 
indicando que existem muitos produtos com preços significativamente mais altos do que a maioria. A distribuição é altamente 
assimétrica, com a maioria dos produtos concentrados em preços mais baixos e alguns produtos caros puxando a média para cima.

2- 'discount_percentage': Este box plot parece ter outliers em ambos os extremos, mas principalmente na parte inferior 
(descontos muito baixos) e na parte superior (descontos muito altos). No entanto, a maioria dos dados parece estar concentrada
em uma faixa específica de percentuais de desconto.

3- 'rating': O box plot para 'rating' mostra uma distribuição relativamente compacta e simétrica, com a maioria das avaliações
entre 3.8 e 4.5. Há alguns outliers na parte inferior, representando produtos com avaliações muito baixas, mas são poucos.

4- 'rating_count': Similar aos preços, 'rating_count' também exibe muitos outliers na parte superior, indicando que alguns
produtos receberam um número de avaliações extremamente alto em comparação com a maioria. Isso é esperado para produtos populares..

Conclusão
A maioria dos "outliers" identificados pelos box plots são valores válidos que simplesmente representam a grande variedade de produtos
e popularidade no dataset do Amazon. Ou seja, não há evidências claras de erros de dados massivos que gerem esses outliers. Eles são parte
da natureza do seu conjunto de dados,  há outliers estatísticos, mas a maioria parece ser valores válidos dentro do contexto de um e-commerce. 
"""

# Visualização dos dados.
# Scatter plot
plt.scatter(df_amazon['actual_price'], df_amazon['rating'])
plt.xlabel('Actual Price')
plt.ylabel('Rating')
plt.title('Scatter Plot')
plt.show()

# Plot para saber a media e mediana do 'actual_price' de todos os produtos e frequencia
plt.figure(figsize=(12, 5))
plt.hist(df_amazon['actual_price'], bins = 20, color = 'skyblue', edgecolor = 'black', alpha = 0.7)
plt.axvline(df_amazon['actual_price'].mean(), color = 'red', linestyle = '--', linewidth = 2, label = f'Média: {df_amazon["actual_price"].mean():.2f}')
plt.axvline(df_amazon['actual_price'].median(), color = 'orange', linestyle = '--', linewidth =2, label = f'Mediana: {df_amazon["actual_price"].median():.2f}')
plt.axvline(df_amazon['actual_price'].mean() + df_amazon['actual_price'].std(), color = 'green', linestyle = ':', linewidth = 2, label = f' +1 DP = ₹ {df_amazon["actual_price"].mean() + df_amazon["actual_price"].std():.2f}')
plt.axvline(df_amazon['actual_price'].mean() - df_amazon['actual_price'].std(), color = 'green', linestyle = ':', linewidth = 2, label = f' -1 DP = ₹ {df_amazon["actual_price"].mean() - df_amazon["actual_price"].std():.2f}')
plt.title('Distribuição dos valores de compra')
plt.xlabel('Valor da compra')
plt.ylabel('Frequencia')
plt.legend()
plt.grid(alpha = 0.3)
plt.show()

# Visualização para variaveis quantitativas em geral.

quant_cols = ['discounted_price', 'actual_price', 'discount_percentage', 'rating', 'rating_count', 'final_price']

for col in quant_cols:
    plt.figure(figsize=(12, 5))

    # Histogram
    plt.subplot(1, 2, 1)
    sns.histplot(df_amazon[col], kde=True)
    plt.title(f'Histograma de {col}')
    plt.xlabel(col)
    plt.ylabel('Frequência')

    # Boxplot
    plt.subplot(1, 2, 2)
    sns.boxplot(y=df_amazon[col])
    plt.title(f'Boxplot de {col}')
    plt.ylabel(col)

    plt.tight_layout()
    plt.show()

# Tabela de Frequência para a coluna 'category'
category_frequency = df_amazon['category'].value_counts()
print(category_frequency.head(10)) # Mostrando as 10 categorias mais frequentes

# Grafico de barras para os 10 produtos mais vendidos
plt.figure(figsize=(15, 8))
sns.barplot(x=category_frequency.head(10).index, y=category_frequency.head(10).values, palette='viridis')
plt.title('Top 10 Categorias de Produtos Mais Frequentes')
plt.xlabel('Categoria do Produto')
plt.ylabel('Contagem')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Relação entre as variaveis "Desconto" com 'avaliação'
print("Scatter Plot: Percentual de Desconto vs. Avaliação")
plt.figure(figsize=(10, 6))
sns.scatterplot(x='discount_percentage', y='rating', data=df_amazon)
plt.title('Relação entre Percentual de Desconto e Avaliação')
plt.xlabel('Percentual de Desconto')
plt.ylabel('Avaliação')
plt.show()

# Transformação de Dados
# Usando LabelEncoder para transformar colunas categoricas, em valores numericos .

product_id = LabelEncoder()
category = LabelEncoder()
review_id = LabelEncoder()
review_content = LabelEncoder()
product_name = LabelEncoder()
user_name = LabelEncoder()
about_product = LabelEncoder()
user_id = LabelEncoder()
review_title = LabelEncoder()
final_price = LabelEncoder()


df_amazon['product_id'] = product_id.fit_transform(df_amazon['product_id'])
df_amazon['category'] = category.fit_transform(df_amazon['category'])
df_amazon['review_id'] = review_id.fit_transform(df_amazon['review_id'])
df_amazon['review_content'] = review_content.fit_transform(df_amazon['review_content'])
df_amazon['product_name'] = product_name.fit_transform(df_amazon['product_name'])
df_amazon['user_name'] = user_name.fit_transform(df_amazon['user_name'])
df_amazon['about_product'] = about_product.fit_transform(df_amazon['about_product'])
df_amazon['user_id'] = user_id.fit_transform(df_amazon['user_id'])
df_amazon['review_title'] = review_title.fit_transform(df_amazon['review_title'])
df_amazon['final_price'] = final_price.fit_transform(df_amazon['final_price'])


# Plot de Correlação entre variavéis.
corr_matrix = df_amazon.corr()
sns.heatmap(corr_matrix, annot=True)
plt.show()


# Calcular os coeficientes de correlação de Pearson.
print(corr_matrix)

# Criando mapa de calor, para ver a correlação.
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix (Pearson)")
plt.show()


# Calcular os coeficientes de correlação de Spearma.
spearman_corr_matrix = df_amazon.corr(method="spearman")

# Print da correlação de Spearma .
print(spearman_corr_matrix)

# Criando o heatmap para visualizar a correlação de Spearma.
sns.heatmap(spearman_corr_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix (Spearman)")
plt.show()

#Calculando o coeficiente
corr_coeficiente = np.corrcoef(df_amazon['actual_price'], df_amazon['rating'])[0, 1]

# Print correlation coefficient
print(corr_coeficiente)

# Resultado proximo de 0, mostra uma relação fraca entre as duas variaveis.

# Calcular média de vendas por categoria do produto.
df_group = df_amazon.groupby('category')['rating'].mean()
print(df_group)

# Chi-square test, teste de contigência
contigency_table = pd.crosstab(df_amazon['actual_price'], df_amazon['rating'])
contigency_table

df_amazon.info()

# Transformação Inversa dos dados.

df_amazon['product_id'] = product_id.inverse_transform(df_amazon['product_id'])
df_amazon['category'] = category.inverse_transform(df_amazon['category'])
df_amazon['review_id'] = review_id.inverse_transform(df_amazon['review_id'])
df_amazon['review_content'] = review_content.inverse_transform(df_amazon['review_content'])
df_amazon['product_name'] = product_name.inverse_transform(df_amazon['product_name'])
df_amazon['user_name'] = user_name.inverse_transform(df_amazon['user_name'])
df_amazon['about_product'] = about_product.inverse_transform(df_amazon['about_product'])
df_amazon['user_id'] = user_id.inverse_transform(df_amazon['user_id'])
df_amazon['review_title'] = review_title.inverse_transform(df_amazon['review_title'])
df_amazon['final_price'] = final_price.inverse_transform(df_amazon['final_price'])


df_amazon.info()

# Conclusões Finais
"""
Quais padrões e tendências foram identificados?
1. Distribuição de Preços Assimétrica: As variáveis discounted_price e actual_price (preço real) mostram uma distribuição assimétrica positiva.
Isso é comum em mercados de e-commerce, onde existem produtos de entrada e produtos premium
2. Correlação Fraca entre Preço e Avaliação: O coeficiente de correlação de Pearson entre actual_price e rating foi de aproximadamente 0.12,
indicando uma relação linear positiva muito fraca.
3. Concentração de Avaliações em Alta: A maioria dos produtos tem avaliações concentradas na faixa de 3.8 a 4.5,
com poucos produtos recebendo avaliações muito baixas. Isso sugere uma satisfação geral positiva dos clientes.

Há valores extremos ou dados faltantes que podem impactar a análise?
1. Identificamos que a única coluna com dados faltantes era rating_count, com apenas 2 valores ausentes, representando cerca de 0.14% do total.
2. Esses valores foram preenchidos com a mediana da coluna rating_count, o que é uma abordagem comum para um número tão pequeno de dados faltantes
e para evitar distorções que a média poderia causar na presença de outliers.
3. A maioria dos "outliers" identificados pelos box plots são valores válidos que simplesmente representam a grande variedade de produtos
e popularidade no dataset do Amazon. Ou seja, não há evidências claras de erros de dados massivos que gerem esses outliers.
Eles são parte da natureza do seu conjunto de dados,  há outliers estatísticos, mas a maioria parece ser valores válidos dentro do contexto
de um e-commerce.(Botei essa observação acima)

Existem correlações ou associações relevantes entre as variáveis?
Existe uma correlação positiva muito forte entre o preço com desconto (discounted_price) e o preço real (actual_price), 
(coeficiente de Pearson em torno de 0.96 e Spearman em torno de 0.93). Isso é natural, pois o preço com desconto é diretamente derivado do preço real.
"""

# Exportando dataset tratado, para ser usado no power bi.
df_amazon.to_csv('amazon_tratado.csv', index=False)
print('DataFrame exportado com sucesso para amazon_tratado.csv')
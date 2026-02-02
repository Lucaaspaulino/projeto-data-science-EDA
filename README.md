Análise Exploratória de Dados (EDA) de Produtos E-commerce da Amazon
Este projeto realiza uma Análise Exploratória de Dados (EDA) em um dataset de produtos da Amazon, com o objetivo de entender a distribuição de preços, avaliações, categorias e outras características dos produtos.

📊 Fonte dos Dados
O dataset utilizado foi carregado a partir de um arquivo CSV disponível publicamente no GitHub: https://raw.githubusercontent.com/Lucaaspaulino/Dataset-Amazon/refs/heads/main/amazon.csv

✨ Passos Realizados na Análise
Carregamento e Visão Geral dos Dados

O dataset foi carregado utilizando a biblioteca Pandas.
Verificações iniciais de cabeçalho (.head()), informações (.info()) e dimensões (.shape) foram realizadas.
Limpeza e Pré-processamento de Dados

Conversão de Tipos de Dados: As colunas discounted_price e actual_price tiveram seus símbolos monetários (₹) e vírgulas removidos, sendo convertidas para o tipo float64.
A coluna discount_percentage teve o símbolo % removido e foi convertida para float64, sendo em seguida dividida por 100 para representar um valor decimal.
Tratamento de Valores Incomuns: Um valor incorreto (|) na coluna rating foi identificado e substituído pelo valor 4.0, com base em uma pesquisa externa do produto. A coluna rating foi então convertida para float64.
rating_count: Teve vírgulas removidas e foi convertida para float64.
Tratamento de Valores Ausentes: Foi identificada a presença de 2 valores ausentes na coluna rating_count (0.14% do total). Esses valores foram imputados com a mediana da coluna para evitar distorções.
Duplicatas: Verificou-se a ausência de linhas duplicadas no dataset.
Engenharia de Features

Foi criada uma nova coluna, final_price, calculada como a diferença entre actual_price e discounted_price.
Simplificação de Categorias

A coluna category, que continha strings longas e aninhadas (ex: Computers&Accessories|Accessories&Peripherals|Cables&Accessories|Cables|USBCables), foi simplificada para extrair apenas a categoria mais específica (ex: USBCables).
O símbolo & foi substituído por - para padronização.

Análise Exploratória de Dados (EDA)
Visualização de Outliers: Box plots foram utilizados para identificar outliers em variáveis quantitativas (discounted_price, actual_price, discount_percentage, rating, rating_count). Concluiu-se que a maioria dos
outliers eram valores válidos que representam a variedade natural de produtos e sua popularidade no e-commerce.

🛠 Ferramentas Utilizadas
Python: Linguagem de programação principal.
Pandas: Para manipulação e análise de dados.
Numpy: Para operações numéricas.
Matplotlib: Para visualização de dados estática.
Seaborn: Para visualização de dados estatística e atraente.
Scipy: Para testes estatísticos (e.g., teste Qui-quadrado de independência).
Scikit-learn: Para pré-processamento (e.g., LabelEncoder).

📝 Principais Insights
Distribuição de Preços Assimétrica: As variáveis discounted_price e actual_price mostram uma distribuição assimétrica positiva, indicando a presença de produtos de baixo custo e um grupo menor de produtos premium.
Correlação Fraca entre Preço e Avaliação: O coeficiente de correlação de Pearson entre actual_price e rating foi de aproximadamente 0.12, sugerindo uma relação linear positiva muito fraca.
Concentração de Avaliações Positivas: A maioria dos produtos possui avaliações entre 3.8 e 4.5, com poucos produtos recebendo avaliações muito baixas, o que indica uma alta satisfação geral dos clientes.
rating_count: Embora rating_count tenha muitos outliers na parte superior, estes são considerados válidos, representando produtos extremamente populares.
Simplificação de Categorias: A categorização foi simplificada para facilitar a análise, mostrando USBCables, SmartWatches e Smartphones como as categorias mais frequentes.

🚀 Próximos Passos (Sugestões)
Análise aprofundada de categorias: Explorar a relação entre categorias e outras variáveis (preço, avaliação).
Análise de Sentimento: Analisar o review_content e review_title para extrair insights sobre a opinião dos clientes.
Modelagem Preditiva: Construir modelos para prever o preço com desconto, a avaliação ou a popularidade do produto com base em outras características.

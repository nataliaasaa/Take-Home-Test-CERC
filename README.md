# Take-Home-Test-CERC
Minha resolução do Take Home Test do processo seletivo da empresa CERC

Perguntas escolhidas para o desenvolvimento do projeto:
* 1\. Analyze credit data and financial statements to determine the degree of risk involved in extending credit or lending money. **Resolvido em modelos_ml.ipynb**
* 3\. Generate financial ratios, using computer programs, to evaluate customers' financial status. **Resolvido em regras_financeiras.ipynb**
* 4\. Prepare reports that include the degree of risk involved in extending credit or lending money. **Resolvido em agente_ai.ipynb**

### Os dados:

A base de dados utilizada, [Corporate Credit Rating](https://www.kaggle.com/datasets/agewerc/corporate-credit-rating), foi retirada do site Kaggle em 06/02/2026.

### O repositório:

A analise geral e desenvolvimento de workflow pode ser encontrada em diferentes jupyter notebooks, em sequencia:
  * *exploracao_de_dados.ipynb* faz a primeira leitura da base de dados, explora a presença de valores nulos, a distribuição de classes (target), a distribuição estatística de features, e uma primeria matriz de correlação entre as variaveis do dataset.
  * *modelos_ml.ipynb* converte os Ratings em categorias de risco de acordo com Standard & Poor's (S&P), lida com as colunas categoricas vs numericas, cria um Pipeline para o treinamento do modelo RandomForestClassifier e explora outros modelos com o pacote PyCaret. O modelo com melhor resultado é salvo localmente.
  * *regras_financeiras.ipynb* utiliza as features iniciais para a criação de regras financeiras, com o objetivo de diminuir o rating em duas classes finais: arriscado vs não arriscado. As regras financeiras sao complementadas com um modelo binario de classificação (XGboost). O objetivo aqui é aplicar mais um filtro para diminuir a incidencia de falsos negativos (ou falsos positivos) do modelos_ml.ipynb. O modelo treinado também é salvo localmente.
  * *predição.ipynb* importa uma base de dados de avaliação, nunca usada por nenhum dos modelos. A predição é realizada pelo modelo de ml e novamente pelas regras financeiras, o output destaca as predições em que os modelos discordam, que devem ser melhores analisadas por especialistas. O csv de predições é salvo localmente.
  * *agente_ai.ipynb* utiliza o google Gemini para criar um agente especializado em analise de creditos, que recebe o csv de predições anteriror e interage com o usuario com base em regras pre estabelecidas e apenas informações disponiveis no dataset.

### Streamlit:

A resolução desenvolvida em jupyter notebook também está disponível na plataforma streamlite. O programa roda localmente no browser.

```streamlit run app.py```

### Instalação e Execução

Comandos para Windows, no cmd:

Clone esse repositorio:

```git clone https://github.com/nataliaasaa/Take-Home-Test-CERC.git```

Para a criação do ambiente virtual, navegue até o diretório raiz Take-Home-Test-CERC. Em seguida:

``` python -m venv cerc ```

``` .\cerc\Scripts\activate ```

``` pip install -r requirements.txt ```

Para execução do dashboard em streamlit:

```streamlit run app.py```

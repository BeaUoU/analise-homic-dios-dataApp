**Análise Global e Preditiva de Homicídios Intencionais**

Este projeto consiste numa plataforma de exploração de dados e modelação estatística focada na evolução temporal e distribuição geográfica das taxas de homicídios intencionais no mundo. A divisão do projeto foi feita em duas componentes principais:
1. **Ambiente de Pesquisa (`AP_01.ipynb`)**: Um Jupyter Notebook detalhado para limpeza de dados, análise exploratória de dados (EDA) e treino do modelo preditivo de Regressão Linear utilizando a biblioteca “scikit-learn”.
2. **Aplicação Interativa (dataApp.py)**: Um dashboard web dinâmico desenvolvido em “Streamlit” e “Plotly”, que permite aos utilizadores filtrar dados hierarquicamente e visualizar previsões em tempo real.

Foi utilizado o data frame “Intentional homicide” fornecido pela UNODC (United Nations Office on Drugs and Crime). A quantização dos dados é dividida em duas partes: 

**Counts**: Número absoluto de registros de homicídios.
**Rate per 100,000 population**: Proporção normalizada para comparações justas entre diferentes densidades demográficas.

O link para acesso e download dos dados encontra-se disponível aqui: **[UNODC Data report | Data Portal UNODC](https://data.unodc.org/datareport/hom-victim)**.

---

 **Principais Funcionalidades**

- **Limpeza e Tratamento Automatizado**: Conversão automática de dados textuais e numéricos (correção de separadores de milhar/decimais de vírgula para ponto), tratamento de valores nulos e filtragem focada unicamente no indicador *Victims of intentional homicide*.
- **Filtros Hierárquicos**: Navegação fluida e estruturada na aplicação web por **Região → Sub-região → País → Ano**.
- **Análise Temporal Individualizada**: Visualização dedicada à evolução histórica das taxas de homicídio (por 100 mil habitantes) com foco em países específicos (como o Brasil) e médias globais.
- **Modelos de Previsão Integrados**: Algoritmo de Regressão Linear treinado para prever tendências futuras com base no comportamento histórico.
- **Gráficos Interativos**: Gráficos de barras horizontais e linhas dinâmicos gerados através da biblioteca Plotly Express.

---

**Principais plataformas e bibliotecas utilizadas**  

O projeto foi construído sobre o ecossistema de Data Science do Python:

- **[Python 3.8+](https://www.python.org/)** - Linguagem base do projeto.
- **[Streamlit](https://streamlit.io/)** - Criação da interface web interativa de forma rápida.
- **[Pandas](https://pandas.pydata.org/)** - Manipulação, tratamento e análise estruturada dos dados.
- **[NumPy](https://numpy.org/)** - Operações matemáticas e vetoriais eficientes.
- **[Scikit-Learn](https://scikit-learn.org/)** - Criação, treino e avaliação do modelo de Regressão Linear.
- **[Plotly Express / Graph Objects](https://plotly.com/python/)** - Gráficos dinâmicos e interativos para a web.
- **[Matplotlib & Seaborn](https://seaborn.pydata.org/)** - Visualizações estáticas e de diagnóstico utilizadas na fase de pesquisa do Notebook.

---

**Modelação Estatística (Regressão Linear)** 

A regressão linear é um modelo estatístico utilizado para analisar a relação entre variáveis. Ela busca prever o comportamento de uma variável dependente (y) com base em uma ou mais variáveis independentes (x), utilizando uma equação linear. É amplamente usada em ciência de dados, econometria e machine learning para prever valores, identificar tendências e analisar relações entre variáveis.

**Modelo de Regressão Linear Simples**

Em suma, na regressão linear simples há apenas uma variável independente (x) e uma dependente (y). A equação do modelo é representada como:

y = β₀ + β₁x + ε

- β₀: Intercepto, valor de y quando x é zero.

- β₁: Inclinação da reta, indicando a variação média de y para cada unidade de x.

- ε: Termo de erro, representando o que o modelo não explica.



Os coeficientes do modelo são estimados pelo método dos mínimos quadrados ordinários (MQO), que minimiza a soma dos quadrados dos resíduos. A fórmula para calcular β₁ e β₀ na regressão simples é:

- β₁ = Σ((xᵢ - x̄)(yᵢ - ȳ)) / Σ((xᵢ - x̄)²)

- β₀ = ȳ - β₁x̄

Para que o modelo seja válido, algumas premissas devem ser atendidas:

- Os erros devem ter média zero e variância constante.

- Não deve haver autocorrelação entre os erros.

- A relação entre as variáveis deve ser linear.

---

**Execução do data App**

É necessária a instalação prévia das bibliotecas citadas anteriormente, para isso basta utilizar o comando “pip install pandas”, por exemplo, no terminal do VScode.

Após isso, para execução do data App basta adicionar o comando “streamlit run dataApp.py” no terminal do VScode, automaticamente abrirá o endereço no navegador padrão utilizado pelo usuário.

---

Este projeto foi desenvolvido com fins acadêmicos pelas alunas do curso de Engenharia de Computação da Universidade Federal do Ceará (UFC) - Campus Sobral em 2026.1, para a disciplina de Tópicos Especiais em Computação, ministrada pelo Prof. Dr. Iális Cavalcante Paula Júnior.

---

**Autores**

- Gabriela da Silva Melo e Costa
- Isabela da Silva Melo e Costa 
- Maria Beatriz Vitorino Almeida
- Karen Stephan da Penha Sousa  



import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


# Exportar Dataset
df = pd.read_csv('https://docs.google.com/spreadsheets/d/14GjwAZoqXbdEv_l3WWpCqSQv1A9imODE/export?format=csv')


    # Retirando colunas inúteis
colunas_inuteis = ['Iso3_code', 'Source']
novo_df = df.drop(columns=colunas_inuteis, errors='ignore')

    #Concentrando apenas nas vítimas de homicidio
novo_df = novo_df[novo_df['Indicator'] == 'Victims of intentional homicide']

    # Converter vírgula em ponto ANTES de transformar VALUE em número
novo_df['VALUE'] = novo_df['VALUE'].astype(str).str.replace(',', '.')

    # Converter os dados em VALUE para o tipo númerico
novo_df['VALUE'] = pd.to_numeric(novo_df['VALUE'], errors='coerce')

# Nivelamento para obter proporções mais justas
df_taxa= novo_df[novo_df['Unit of measurement'] == 'Rate per 100,000 population']

# Analise da tendencia de homicídios por ano
df_tendencia = df_taxa.groupby('Year')['VALUE'].mean().reset_index()

# Divisão dos Dados em foco
X = df_tendencia[['Year']]
y = df_tendencia['VALUE']

regressor_linear = LinearRegression()

#Treinando o modelo
regressor_linear.fit(X, y)

# Criar dados para o futuro
futuro = pd.DataFrame({'Year': [2023, 2024, 2025, 2026]})

# Prever taxas futuras
previsoes = regressor_linear.predict(futuro)
futuro['Taxa Prevista'] = previsoes



def main():
    st.title("Análise de Homicídios no Mundo")

    # 1. Preparar os dados para o Plotly (unir histórico e previsão)
    # Criamos um DataFrame para o histórico com os mesmos nomes de coluna da previsão
    df_historico_plot = df_tendencia.rename(columns={'VALUE': 'Taxa de Homicídio', 'Year': 'Ano'})
    df_historico_plot['Tipo'] = 'Dados Históricos Reais'

    df_futuro_plot = futuro.rename(columns={'Taxa Prevista': 'Taxa de Homicídio', 'Year': 'Ano'})
    df_futuro_plot['Tipo'] = 'Previsão (Regressão Linear)'

    # Juntamos tudo em um único DataFrame que o Plotly adora trabalhar
    df_completo = pd.concat([df_historico_plot, df_futuro_plot], ignore_index=True)

    # 2. Criar o gráfico interativo com Plotly Express
    fig = px.line(
        df_completo, 
        x='Ano', 
        y='Taxa de Homicídio', 
        color='Tipo',
        title='Tendência e Previsão da Taxa de Homicídios (Global) usando Regressão Linear',
        markers=True, # Adiciona as bolinhas/pontos
        color_discrete_map={
            'Dados Históricos Reais': 'darkturquoise', # Cor próxima do ciano original
            'Previsão (Regressão Linear)': 'purple'
        }
    )

    # 3. Customizações estéticas extras (opcional, para deixar parecido com o seu)
    fig.update_layout(
        title_font=dict(size=16, family="Arial"),
        xaxis=dict(
            tickmode='linear', 
            tick0=1990, 
            dtick=2,
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.2)' # Grade sutil que aparece no claro e no escuro
        ),
        yaxis=dict(
            showgrid=True, 
            gridcolor='rgba(128, 128, 128, 0.2)', # Grade sutil transparente
            title_text='Taxa de Homicídio por 100 mil habitantes'
        ),
        # Removemos os fundos brancos fixos para que o Streamlit controle a transparência!
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )

    # 4. Exibir o gráfico interativo no Streamlit
    st.plotly_chart(fig, use_container_width=True)

    #Pergunta 01

# Encontrar o ano mais recente do Banco de Dados
    ano_atual = df_taxa['Year'].max()

    # Criar Banco de Dados restringindo últimos 5 anos
    últimos_5_anos = df_taxa[df_taxa['Year'] >= ano_atual - 4]

    # Criar Banco de Dados agrupado por 'Country' pela média de 'VALUE'
    df_media_paises = últimos_5_anos.groupby('Country')['VALUE'].mean().reset_index()

    # Criar Banco de Dados em ordem decrescente e armazenar os 10 primeiros valores
    df_top_10 = df_media_paises.sort_values(by='VALUE', ascending=False).head(10)

    # Ordenar em ordem crescente para que o país com maior taxa fique no topo do gráfico horizontal
    df_grafico = df_top_10.sort_values(by='VALUE', ascending=True)

    # 1. Criar o gráfico de barras horizontais interativo usando Plotly
    fig2 = px.bar(
        df_grafico,
        x='VALUE',
        y='Country',
        orientation='h',
        title=f'Top 10 Países com Maiores Taxas de Homicídios (Média de {ano_atual - 4} a {ano_atual})',
        text='VALUE',
        color='VALUE',  # <--- CRUCIAL: Diz ao Plotly para colorir baseado na taxa!
        color_continuous_scale=px.colors.sequential.Plasma,  # <--- Escolha a paleta aqui (Plasma, Viridis, Inferno, etc.)
        labels={'VALUE': 'Taxa Média (por 100 mil habitantes)', 'Country': 'País', 'color': 'Taxa'}
    )

    # 2. Customizações visuais e adaptação ao Tema Escuro/Claro
    fig2.update_traces(
        texttemplate='%{textTarget:.1f}',  
        textposition='outside'        
    )

    fig2.update_layout(
        title_font=dict(size=16, family="Arial"),
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.2)'  
        ),
        yaxis=dict(showgrid=False),
        
        # Remove fundos fixos para respeitar o Dark/Light mode
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=150),
        
        # Configuração da barra de legenda do gradiente
        coloraxis_showscale=True # Mostra a barra lateral explicando o gradiente de cor
    )

    # 3. Exibir o gráfico correto no Streamlit
    st.plotly_chart(fig2, use_container_width=True)

    #Pergunta 02

    # Filtrando os dados de homicídios de mulheres em 2022
    homicidio_mulheres = (
        (novo_df['Year'] == 2022) &
        (novo_df['Sex'] == 'Female') &
        (novo_df['Age'] == 'Total') &
        (novo_df['Unit of measurement'] == 'Rate per 100,000 population') &
        (novo_df['Indicator'] == 'Victims of intentional homicide') &
        (novo_df['Dimension'] == 'Total')
    )

    df_homicidio_mulheres = novo_df[homicidio_mulheres].copy()
    df_homicidio_mulheres = df_homicidio_mulheres.drop_duplicates(subset=['Country'])
    
    # Ordenando do maior para o menor e pegando o Top 10
    df_ordenado = df_homicidio_mulheres.sort_values(by='VALUE', ascending=False)
    df_homicidio_mulheres_2022 = df_ordenado.head(10)

    # Invertemos a ordem aqui para que a maior barra fique no topo do gráfico horizontal do Plotly
    df_grafico_mulheres = df_homicidio_mulheres_2022.sort_values(by='VALUE', ascending=True)

    # 1. Criar o gráfico de barras interativo com gradiente vermelho (Reds)
    fig3 = px.bar(
        df_grafico_mulheres,
        x='VALUE',
        y='Country',
        orientation='h',
        title='Top 10 Países com Maiores Índices de Homicídios de Mulheres (2022)',
        text='VALUE',
        color='VALUE',  # Define o gradiente baseado no valor numérico
        color_continuous_scale=px.colors.sequential.Reds,  # Novo gradiente de intensidade (tons de vermelho)
        labels={'VALUE': 'Taxa de Homicídios (por 100 mil mulheres)', 'Country': 'Países', 'color': 'Taxa'}
    )

    # 2. Configurações visuais e rótulos
    fig3.update_traces(
        texttemplate='%{textTarget:.1f}',  # Mostra apenas uma casa decimal nas barras
        textposition='outside'            # Coloca o texto para fora da barra
    )

    fig3.update_layout(
        title_font=dict(size=16, family="Arial"),
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.2)',  # Grade sutil para modo escuro/claro
            nticks=6                               # Equivalente ao nbins=6 do seu código original
        ),
        yaxis=dict(showgrid=False),
        
        # Remove fundos fixos para total integração com o tema do Streamlit
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=150),
        
        coloraxis_showscale=True  # Mostra a barra lateral com a legenda do gradiente
    )

    # 3. Renderizar o gráfico no seu DataApp
    st.plotly_chart(fig3, use_container_width=True)

    # Pergunta 03

    df_regioes = novo_df[
        (novo_df['Indicator'] == 'Victims of intentional homicide') &
        (novo_df['Unit of measurement'] == 'Counts')
    ].copy()

    # Agrupar por 'Region' e somar os valores de 'VALUE'
    df_homicidios_por_regiao = df_regioes.groupby('Region')['VALUE'].sum().reset_index()

    # GARANTIA: Arredondar os valores somados para evitar dízimas decimais longas
    df_homicidios_por_regiao['VALUE'] = df_homicidios_por_regiao['VALUE'].round(0)

    # Ordenar os dados em ordem decrescente e pegar as top 10 regiões
    df_top_10_regioes = df_homicidios_por_regiao.sort_values(by='VALUE', ascending=False).head(10)

    # Ordenar em ordem crescente para que a maior barra fique no topo do gráfico do Plotly
    df_top_10_regioes_plot = df_top_10_regioes.sort_values(by='VALUE', ascending=True)

    # 1. Criar o gráfico de barras interativo com gradiente Viridis
    fig4 = px.bar(
        df_top_10_regioes_plot,
        x='VALUE',
        y='Region',
        orientation='h',
        title="Top 10 Regiões com maior Número Total de Homicídios",
        text='VALUE',
        color='VALUE',
        color_continuous_scale=px.colors.sequential.Viridis,
        labels={'VALUE': 'Número Total de Homicídios', 'Region': 'Região', 'color': 'Total'}
    )

    # 2. Configurações visuais, formatação de números grandes e rótulos
    fig4.update_traces(
        texttemplate='%{text:,.0f}',  # Formata números grandes com separador de milhar
        textposition='outside'        # Coloca o texto para fora da barra
    )

    fig4.update_layout(
        title_font=dict(size=16, family="Arial"),
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.2)',
            # Ativa a barra de deslize (scroll/slider) no eixo X
            rangeslider=dict(visible=True), 
            autorange=True
        ),
        yaxis=dict(showgrid=False),
        
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=150, r=50, t=50, b=50),
        coloraxis_showscale=True
    )

    # 3. Exibir o gráfico no Streamlit
    st.plotly_chart(fig4, use_container_width=True)

    # Pergunta 04

# Ordenar os dados em ordem crescente para que a maior barra fique no topo do gráfico horizontal
    df_homicidios_subregiao = novo_df[
        (novo_df['Indicator'] == 'Victims of intentional homicide') &
        (novo_df['Unit of measurement'] == 'Counts')
    ].copy()
    df_homicidios_por_pais_subregiao = df_homicidios_subregiao.groupby(['Subregion', 'Country'])['VALUE'].sum().reset_index()
    id_menor_homicidio = df_homicidios_por_pais_subregiao.groupby('Subregion')['VALUE'].idxmin()
    df_menor_homicidio_por_subregiao = df_homicidios_por_pais_subregiao.loc[id_menor_homicidio].sort_values(by='VALUE', ascending=True)
    df_grafico_menor = df_menor_homicidio_por_subregiao.sort_values(by='VALUE', ascending=True)

    # 1. Criar o gráfico de barras interativo com legenda por Sub-região
    fig5 = px.bar(
        df_grafico_menor,
        x='VALUE',
        y='Country',
        orientation='h',
        color='Subregion',  # <--- Mapeia a cor pela Sub-região (equivalente ao hue do Seaborn)
        title='Países com o Menor Número de Homicídios por Sub-Região',
        text='VALUE',       # Adiciona o valor numérico bruto
        color_continuous_scale=px.colors.sequential.Cividis, # Nova paleta elegante para diferenciar
        labels={'VALUE': 'Número de Homicídios', 'Country': 'País', 'Subregion': 'Sub-Região'}
    )

    # 2. Configurações visuais e formatação dos rótulos
    fig5.update_traces(
        texttemplate='%{text:,.0f}',  # Formata números com separador de milhar se necessário
        textposition='outside'        # Coloca o texto para fora da barra
    )

    fig5.update_layout(
        title_font=dict(size=16, family="Arial"),
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.2)',
            # ATIVAÇÃO DO SLIDER: Adiciona a barra de deslize horizontal que você gostou!
            rangeslider=dict(visible=True),
            autorange=True
        ),
        yaxis=dict(showgrid=False),
        
        # Remove os fundos fixos brancos para respeitar o Dark/Light mode do Streamlit
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=150, r=50, t=50, b=50)
    )

    # 3. Exibir o gráfico no Streamlit
    st.plotly_chart(fig5, use_container_width=True)

    # Pergunta 05

    mortes_mulheres = (
        (novo_df['Sex'] == 'Female') &
        (novo_df['Age'] == 'Total') &
        (novo_df['Unit of measurement'] == 'Counts') &
        (novo_df['Indicator'] == 'Victims of intentional homicide') &
        (novo_df['Dimension'] == 'Total')
    )

    df_mortes_mulheres = novo_df[mortes_mulheres].copy()

    # Agrupando por país e calculando a soma dos 'VALUE' ao longo dos anos.
    df_mortes_mulheres_por_pais = df_mortes_mulheres.groupby('Country')['VALUE'].sum().reset_index()

    # GARANTIA: Arredondar os valores para evitar casas decimais na contagem absoluta
    df_mortes_mulheres_por_pais['VALUE'] = df_mortes_mulheres_por_pais['VALUE'].round(0)

    # Ordenando o DataFrame pelo VALUE do menor para o maior
    df_menores_mortes_mulheres = df_mortes_mulheres_por_pais.sort_values(by='VALUE', ascending=True)

    # Pegando as 10 primeiras linhas (menores valores acumulados)
    df_top10_menores_mortes_mulheres = df_menores_mortes_mulheres.head(10)

    # Mantendo a ordenação crescente para o Plotly renderizar do menor para o maior corretamente na tela
    df_grafico_mulheres_menor = df_top10_menores_mortes_mulheres.sort_values(by='VALUE', ascending=True)

    # 1. Criar o gráfico de barras interativo com gradiente Mint
    fig6 = px.bar(
        df_grafico_mulheres_menor,
        x='VALUE',
        y='Country',
        orientation='h',
        title='Top 10 Países com Menores Números de Homicídios de Mulheres (Total Acumulado)',
        text='VALUE',
        color='VALUE',  # Modifica a intensidade da cor conforme o valor aumenta
        color_continuous_scale=px.colors.sequential.Mint,  # Tons de verde limpos para representar taxas menores
        labels={'VALUE': 'Número Total de Homicídios', 'Country': 'Países', 'color': 'Total'}
    )

    # 2. Configurações visuais e formatação dos rótulos
    fig6.update_traces(
        texttemplate='%{text:,.0f}',  # Remove decimais e formata milhares
        textposition='outside'        # Posiciona os valores após o fim das barras
    )

    fig6.update_layout(
        title_font=dict(size=16, family="Arial"),
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.2)',
            nticks=6,  # Mantém o limite de divisões visuais do seu nbins=6 original
            # ATIVAÇÃO DO SLIDER: Garante a barra de rolagem horizontal se necessário
            rangeslider=dict(visible=True),
            autorange=True
        ),
        yaxis=dict(showgrid=False),
        
        # Transparência total para integrar com o Modo Claro / Modo Escuro do Streamlit
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=150, r=50, t=50, b=50),
        
        coloraxis_showscale=True  # Barra de escala de intensidade na lateral direita
    )

    # 3. Exibir o gráfico dinâmico no Streamlit
    st.plotly_chart(fig6, use_container_width=True)

    # Pergunta 06

    df_sub = (
        (novo_df['Indicator'] == 'Victims of intentional homicide') &
        (novo_df['Unit of measurement'] == 'Counts')
    )

    df_sub = novo_df[df_sub].copy()

    # Agrupar 'Subregion' pela soma de 'VALUE'
    df_subregiao = df_sub.groupby(['Subregion'])['VALUE'].sum().reset_index()

    # GARANTIA: Arredondar os valores somados para evitar dízimas no total absoluto
    df_subregiao['VALUE'] = df_subregiao['VALUE'].round(0)

    # Colocar dados em Ordem Decrescente e pegar o Top 10
    df_top_10_subregiao = df_subregiao.sort_values(by='VALUE', ascending=False).head(10)

    # Ordenar em ordem crescente para o Plotly renderizar a maior barra no topo
    df_top_10_subregiao_plot = df_top_10_subregiao.sort_values(by='VALUE', ascending=True)

    # 1. Criar o gráfico de barras interativo com gradiente Inferno
    fig7 = px.bar(
        df_top_10_subregiao_plot,
        x='VALUE',
        y='Subregion',
        orientation='h',
        title="Top 10 Sub-Regiões com maior Número Total de Homicídios",
        text='VALUE',
        color='VALUE',  # Define a intensidade do gradiente com base no volume
        color_continuous_scale=px.colors.sequential.Inferno,  # Transição marcante do roxo ao amarelo/fogo
        labels={'VALUE': 'Número Total de Homicídios', 'Subregion': 'Sub-região', 'color': 'Total'}
    )

    # 2. Configurações visuais e formatação dos rótulos
    fig7.update_traces(
        texttemplate='%{text:,.0f}',  # Formata com separador de milhar e sem decimais
        textposition='outside'        # Posiciona os números para fora da barra
    )

    fig7.update_layout(
        title_font=dict(size=16, family="Arial"),
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.2)',
            # ATIVAÇÃO DO SLIDER: Barra de deslize horizontal que você escolheu
            rangeslider=dict(visible=True),
            autorange=True
        ),
        yaxis=dict(showgrid=False),
        
        # Transparência total para se fundir perfeitamente ao tema do Streamlit
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=150, r=50, t=50, b=50),
        
        coloraxis_showscale=True  # Mostra a régua lateral com o gradiente de intensidade
    )

    # 3. Exibir o gráfico no Streamlit
    st.plotly_chart(fig7, use_container_width=True)

    # Pergunta 07

# Filtrando os dados para mortes em cada continente em 2020.
    homicidio_continentes = (
        (novo_df['Year'] == 2020) &
        (novo_df['Sex'] == 'Total') &
        (novo_df['Age'] == 'Total') &
        (novo_df['Unit of measurement'] == 'Counts') &
        (novo_df['Indicator'] == 'Victims of intentional homicide') &
        (novo_df['Dimension'] == 'Total')
    )

    df_homicidio_continentes = novo_df[homicidio_continentes].copy()

    # GARANTIA: Arredondar os valores para evitar casas decimais na contagem absoluta
    df_homicidio_continentes['VALUE'] = df_homicidio_continentes['VALUE'].round(0)

    # Encontrando o país com o maior número de homicídios por continente em 2020
    paises_homicidio_continente = df_homicidio_continentes.loc[df_homicidio_continentes.groupby('Region')['VALUE'].idxmax()]

    df_homicidio_continente_2020 = paises_homicidio_continente.sort_values(by='VALUE', ascending=False)

    # Ordenar em ordem crescente para o Plotly renderizar a maior barra no topo do gráfico horizontal
    df_grafico_continente = df_homicidio_continente_2020.sort_values(by='VALUE', ascending=True)

    # 1. Criar o gráfico de barras interativo mapeando a cor pelo Continente (Region)
    fig8 = px.bar(
        df_grafico_continente,
        x='VALUE',
        y='Country',
        orientation='h',
        color='Region',  # <--- Mantém o comportamento do hue='Region' do seu código original
        title='País com o Maior Número de Homicídios por Continente (2020)',
        text='VALUE',
        color_discrete_sequence=px.colors.qualitative.Safe, # Paleta elegante e de alto contraste para categorias
        labels={'VALUE': 'Número de Homicídios', 'Country': 'País', 'Region': 'Continente'}
    )

    # 2. Configurações visuais e formatação dos rótulos
    fig8.update_traces(
        texttemplate='%{text:,.0f}',  # Formata com separador de milhar e remove decimais dos valores das barras
        textposition='outside'        # Posiciona os textos após o fim das barras
    )

    fig8.update_layout(
        title_font=dict(size=16, family="Arial"),
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.2)',
            nticks=6,  # Mantém o controle de divisões visuais do seu nbins=6 original
            # ATIVAÇÃO DO SLIDER: Permite deslizar horizontalmente sem quebras
            rangeslider=dict(visible=True),
            autorange=True
        ),
        yaxis=dict(showgrid=False),
        
        # Transparência total para se adaptar de forma perfeita tanto ao Light quanto ao Dark Mode do Streamlit
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=150, r=50, t=50, b=50)
    )

    # 3. Exibir o gráfico no Streamlit
    st.plotly_chart(fig8, use_container_width=True)

    # Pergunta 08

# Filtrando os dados para homicídios de mulheres em 2021
    homicidios_mulheres_2021 = (
        (novo_df['Year'] == 2021) &
        (novo_df['Sex'] == 'Female') &
        (novo_df['Age'] == 'Total') &
        (novo_df['Unit of measurement'] == 'Rate per 100,000 population') &
        (novo_df['Indicator'] == 'Victims of intentional homicide') &
        (novo_df['Dimension'] == 'Total')
    )

    df_mulheres_2021 = novo_df[homicidios_mulheres_2021].copy()

    # Encontrando o país com o maior 'VALUE' (taxa de homicídios)
    pais_mais_violento = df_mulheres_2021.loc[df_mulheres_2021['VALUE'].idxmax()]

    # Criamos um DataFrame temporário com uma linha só para o Plotly Express renderizar corretamente
    df_single_bar = pd.DataFrame([pais_mais_violento])

    # 1. Criar o gráfico de barra única interativo
    fig9 = px.bar(
        df_single_bar,
        x='VALUE',
        y='Country',
        orientation='h',
        title='País Mais Violento para Mulheres (2021)',
        text='VALUE',
        color='VALUE',  # Mantém o gradiente de intensidade baseado no valor
        color_continuous_scale=px.colors.sequential.Reds,  # Paleta de alerta em tons de vermelho
        labels={'VALUE': 'Taxa de Homicídios (por 100 mil mulheres)', 'Country': 'País', 'color': 'Taxa'}
    )

    # 2. Configurações visuais e formatação com precisão de 2 casas decimais
    fig9.update_traces(
        texttemplate='%{textTarget:.2f}',  # Mantém a formatação original de duas casas decimais (ex: 12.34)
        textposition='outside'            # Garante que o número fique após o fim da barra
    )

    fig9.update_layout(
        title_font=dict(size=16, family="Arial"),
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.2)',
            # Mantém a consistência com a barra de rolagem horizontal que você escolheu
            rangeslider=dict(visible=True),
            autorange=True
        ),
        yaxis=dict(showgrid=False),
        
        # Remove os fundos fixos para se fundir perfeitamente ao Tema Escuro e Claro do Streamlit
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=150, r=50, t=50, b=50),
        
        coloraxis_showscale=True  # Mostra a régua lateral de intensidade
    )

    # 3. Exibir o gráfico no seu DataApp
    st.plotly_chart(fig9, use_container_width=True)

    # Pergunta 09
    # Filtrando os dados para as taxas de homicídio geral
    homicidio_intencional = (
        (novo_df['Sex'] == 'Total') &
        (novo_df['Age'] == 'Total') &
        (novo_df['Unit of measurement'] == 'Rate per 100,000 population') &
        (novo_df['Indicator'] == 'Victims of intentional homicide') &
        (novo_df['Dimension'] == 'Total')
    )

    df_homicidio_intencional = novo_df[homicidio_intencional].copy()
    df_homicidio_intencional = df_homicidio_intencional.drop_duplicates(subset=['Country'])
    
    # Ordenando e pegando o Top 10
    df_homicidio_intencional_ordenado = df_homicidio_intencional.sort_values(by='VALUE', ascending=False)
    df_homicidio_intencional_top10 = df_homicidio_intencional_ordenado.head(10)

    # Invertemos a ordem para que a maior barra fique no topo do gráfico horizontal do Plotly
    df_grafico_intencional = df_homicidio_intencional_top10.sort_values(by='VALUE', ascending=True)

    # 1. Criar o gráfico de barras interativo com gradiente Laranja-Vermelho (YlOrRd)
    fig10 = px.bar(
        df_grafico_intencional,
        x='VALUE',
        y='Country',
        orientation='h',
        title="Top 10 Países com Maiores Taxas de Homicídio Intencional",
        text='VALUE',
        color='VALUE',  # Intensidade da cor varia conforme a taxa aumenta
        color_continuous_scale=px.colors.sequential.YlOrRd,  # Degradê de alerta do amarelo ao vermelho escuro
        labels={'VALUE': 'Taxa de Homicídios (por 100 mil habitantes)', 'Country': 'Países', 'color': 'Taxa'}
    )

    # 2. Configurações visuais e formatação dos rótulos
    fig10.update_traces(
        texttemplate='%{textTarget:.1f}',  # Exibe os valores nas barras com 1 casa decimal
        textposition='outside'            # Garante que os números fiquem para fora das barras
    )

    fig10.update_layout(
        title_font=dict(size=16, family="Arial"),
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.2)',
            nticks=6,  # Mantém o limite de 6 divisões visuais do seu nbins=6 original
            # ATIVAÇÃO DO SLIDER: Mantém o padrão de rolagem horizontal seguro contra cortes
            rangeslider=dict(visible=True),
            autorange=True
        ),
        yaxis=dict(showgrid=False),
        
        # Remove os fundos fixos para se moldar dinamicamente ao Modo Escuro / Claro do Streamlit
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=150, r=50, t=50, b=50),
        
        coloraxis_showscale=True  # Barra de escala de intensidade na lateral direita
    )

    # 3. Renderizar o gráfico no Streamlit
    st.plotly_chart(fig10, use_container_width=True)

    # Pergunta 10
# Filtrando os dados para a evolução do Brasil nos últimos 10 anos
    filtro_brasil = (
        (novo_df['Country'] == 'Brazil') &
        (novo_df['Year'] >= 2013) &
        (novo_df['Year'] <= 2022) &
        (novo_df['Indicator'] == 'Victims of intentional homicide') &
        (novo_df['Sex'] == 'Total') &
        (novo_df['Age'] == 'Total') &
        (novo_df['Dimension'] == 'Total') &
        (novo_df['Unit of measurement'] == 'Rate per 100,000 population')
    )

    df_brasil_10anos = novo_df[filtro_brasil].copy()
    
    # Calcular a média
    media_br = df_brasil_10anos['VALUE'].mean()
    
    # Ordenando por Ano de forma crescente
    df_brasil_ordenado = df_brasil_10anos.sort_values(by='Year', ascending=True)
    
    # Mantemos o ano como string para o Plotly tratar cada ano como uma barra separada (categoria)
    df_brasil_ordenado['Year'] = df_brasil_ordenado['Year'].astype(str)  

    # 1. Criar o gráfico de barras horizontais interativo com gradiente vermelho
    fig11 = px.bar(
        df_brasil_ordenado,
        x='VALUE',
        y='Year',
        orientation='h',
        title="Evolução da Taxa de Homicídios no Brasil (Últimos 10 Anos Disponíveis)",
        text='VALUE',
        color='VALUE',  # A cor muda conforme a intensidade da taxa do ano
        color_continuous_scale=px.colors.sequential.Reds,  # Degradê vermelho de alerta igual ao original
        labels={'VALUE': 'Taxa de Homicídios (por 100 mil habitantes)', 'Year': 'Anos', 'color': 'Taxa'}
    )

    # 2. Configurações visuais e formatação dos rótulos
    fig11.update_traces(
        texttemplate='%{textTarget:.2f}',  # Mostra a taxa nas barras com precisão de 2 casas decimais
        textposition='outside'            # Coloca os números para fora das barras
    )

    fig11.update_layout(
        title_font=dict(size=16, family="Arial"),
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.2)',
            nticks=6,  # Mantém o limite de 6 divisões visuais do seu nbins=6 original
            # ATIVAÇÃO DO SLIDER: Garante que os números das taxas fiquem livres de cortes
            rangeslider=dict(visible=True),
            autorange=True
        ),
        yaxis=dict(showgrid=False),
        
        # Remove fundos fixos para total integração com o Dark Mode e Light Mode
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=150, r=50, t=50, b=50),
        
        coloraxis_showscale=True  # Barra lateral indicando a intensidade das cores
    )

    # 3. Renderizar o gráfico final no Streamlit
    st.plotly_chart(fig11, use_container_width=True)
    st.info(f"A média da taxa de homicídios no Brasil (2013-2022) é de **{media_br:.2f}** por 100 mil habitantes.")

if __name__ == "__main__":
    main()    
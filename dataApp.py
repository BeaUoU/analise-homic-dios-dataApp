import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression

st.set_page_config(
    page_title="DataApp - Homicídios no Mundo",
    page_icon="📊",
    layout="wide"
)

@st.cache_data
def carregar_dados():
    url = 'https://docs.google.com/spreadsheets/d/14GjwAZoqXbdEv_l3WWpCqSQv1A9imODE/export?format=csv'
    df = pd.read_csv(url)
    
    colunas_inuteis = ['Iso3_code', 'Source']
    df_clean = df.drop(columns=colunas_inuteis, errors='ignore')
    df_clean = df_clean[df_clean['Indicator'] == 'Victims of intentional homicide']
    
    df_clean['VALUE'] = df_clean['VALUE'].astype(str).str.replace(',', '.')
    df_clean['VALUE'] = pd.to_numeric(df_clean['VALUE'], errors='coerce')
    
    df_clean['Region'] = df_clean['Region'].fillna('Não Especificado')
    df_clean['Subregion'] = df_clean['Subregion'].fillna('Não Especificado')
    df_clean['Country'] = df_clean['Country'].fillna('Não Especificado')
    
    return df_clean

df_base = carregar_dados()

# O processo de filtragem é hierárquico: Região > Sub-região > Países > Anos, garantindo que cada etapa refine a base para a próxima
todas_regioes = sorted(df_base['Region'].unique().tolist())
regioes_selecionadas = st.sidebar.multiselect("Selecione as Regiões", todas_regioes, default=todas_regioes)

if not regioes_selecionadas:
    regioes_selecionadas = todas_regioes

df_filtrado_regiao = df_base[df_base['Region'].isin(regioes_selecionadas)]

todas_subregioes = sorted(df_filtrado_regiao['Subregion'].unique().tolist())
subregioes_selecionadas = st.sidebar.multiselect("Selecione as Sub-regiões", todas_subregioes)

if not subregioes_selecionadas:
    df_filtrado_sub = df_filtrado_regiao
else:
    df_filtrado_sub = df_filtrado_regiao[df_filtrado_regiao['Subregion'].isin(subregioes_selecionadas)]

todos_paises = sorted(df_filtrado_sub['Country'].unique().tolist())
paises_selecionados = st.sidebar.multiselect("Selecione os Países (Opcional)", todos_paises)

ano_min, ano_max = int(df_base['Year'].min()), int(df_base['Year'].max())
anos_selecionados = st.sidebar.slider("Intervalo de Anos", ano_min, ano_max, (ano_min, ano_max))

if not paises_selecionados:
    df_filtrado = df_filtrado_sub[
        (df_filtrado_sub['Year'] >= anos_selecionados[0]) & 
        (df_filtrado_sub['Year'] <= anos_selecionados[1])
    ]
else:
    df_filtrado = df_filtrado_sub[
        (df_filtrado_sub['Country'].isin(paises_selecionados)) & 
        (df_filtrado_sub['Year'] >= anos_selecionados[0]) & 
        (df_filtrado_sub['Year'] <= anos_selecionados[1])
    ]

#interface Principal
st.title("Painel Interativo: Análise Global de Homicídios")
st.caption("Explore distribuições geográficas, proporções e tendências estatísticas de forma dinâmica.")

#definindo as abas principais para organizar o conteúdo de forma clara e segmentada
aba1, aba2, aba3, aba4, aba5 = st.tabs([
    "Tendências e Predições", 
    "Recorte de Gênero", 
    "Análise por Regiões", 
    "Panoramas",
    "Análises Específicas"
])


#aba1: modelagem preditiva e distribuição geográfica
with aba1:
    st.subheader("Modelagem Preditiva e Distribuição Geográfica")
    
    df_taxa = df_base[df_base['Unit of measurement'] == 'Rate per 100,000 population']
    df_tendencia = df_taxa.groupby('Year')['VALUE'].mean().reset_index()

    #no streamlit, é importante verificar se o dataframe não está vazio antes de tentar ajustar um modelo de regressão linear, para evitar erros de execução
    if not df_tendencia.empty:
        X_reg = df_tendencia[['Year']]
        y_reg = df_tendencia['VALUE']
        regressor_linear = LinearRegression()
        regressor_linear.fit(X_reg, y_reg)
        futuro = pd.DataFrame({'Year': [2023, 2024, 2025, 2026]})
        futuro['Taxa Prevista'] = regressor_linear.predict(futuro)

        df_historico_plot = df_tendencia.rename(columns={'VALUE': 'Taxa de Homicídio', 'Year': 'Ano'})
        df_historico_plot['Tipo'] = 'Dados Históricos Reais'
        df_futuro_plot = futuro.rename(columns={'Taxa Prevista': 'Taxa de Homicídio', 'Year': 'Ano'})
        df_futuro_plot['Tipo'] = 'Previsão (Regressão Linear)'
        df_completo = pd.concat([df_historico_plot, df_futuro_plot], ignore_index=True)

        fig1 = px.line(
            df_completo, x='Ano', y='Taxa de Homicídio', color='Tipo',
            title='Tendência Histórica e Previsão Estatística do Índice Global',
            markers=True,
            color_discrete_map={'Dados Históricos Reais': '#1f77b4', 'Previsão (Regressão Linear)': '#9467bd'}
        )

        #substituto do matplotlib, o plotly express é uma biblioteca de visualização de dados que permite criar gráficos interativos e personalizáveis. 
        st.plotly_chart(fig1, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("Visão de Calor Global")
    df_mapa = df_filtrado[
        (df_filtrado['Unit of measurement'] == 'Rate per 100,000 population') & 
        (df_filtrado['Sex'] == 'Total') & (df_filtrado['Age'] == 'Total')
    ]
    if not df_mapa.empty:
        ultimo_ano_filtro = df_mapa['Year'].max()
        df_mapa_recente = df_mapa[df_mapa['Year'] == ultimo_ano_filtro]
        
        fig_mapa = px.choropleth(
            df_mapa_recente,
            locations="Country",
            locationmode="country names",
            color="VALUE",
            hover_name="Country",
            title=f"Taxa de Homicídios por 100 mil Habitantes no Mundo ({ultimo_ano_filtro})",
            color_continuous_scale=px.colors.sequential.YlOrRd
        )
        
        fig_mapa.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',  
            plot_bgcolor='rgba(0,0,0,0)',  
            geo=dict(
                showframe=False, 
                showcoastlines=True, 
                projection_type='mollweide', 
                bgcolor='rgba(0,0,0,0)',     
                showocean=False
            )
        )
        st.plotly_chart(fig_mapa, use_container_width=True)

# aba2: análise comparativa de gênero, focando nas taxas por 100 mil habitantes
with aba2:
    st.subheader("Análise Comparativa de Gênero e Impacto nos Homicídios (Por Taxas)")
    
    df_genero_taxas = df_filtrado[
        (df_filtrado['Unit of measurement'] == 'Rate per 100,000 population') & 
        (df_filtrado['Sex'].isin(['Female', 'Male'])) &
        (df_filtrado['Age'] == 'Total') &
        (df_filtrado['Dimension'] == 'Total')
    ]
    
    if not df_genero_taxas.empty:
        tab_distribuicao, tab_rankings = st.tabs(["Comparação Direta de Gênero", "Rankings de Países"])
        
        #uitlizaremos sub-abas para organizar a análise comparativa de gênero, permitindo que o usuário explore tanto a evolução temporal quanto os rankings de países críticos de forma clara e segmentada.
       
        #sub-aba1: comparação direta de gênero
        with tab_distribuicao:
            st.markdown("### Distribuição e Disparidade de Gênero")
            
            #gráfico de Linhas para mostrar a evolução temporal das taxas médias de homicídios por gênero
            df_linha_genero = df_genero_taxas.groupby(['Year', 'Sex'])['VALUE'].mean().reset_index()
            
            fig_linha = px.line(
                df_linha_genero, x='Year', y='VALUE', color='Sex',
                title="Evolução Temporal da Taxa Média de Homicídios por Gênero",
                labels={'VALUE': 'Taxa Média (por 100 mil hab.)', 'Year': 'Ano', 'Sex': 'Gênero'},
                color_discrete_map={'Female': '#EF553B', 'Male': '#636EFA'} # Vermelho para mulheres, Azul para homens
            )
            fig_linha.update_traces(mode='lines+markers')
            fig_linha.update_layout(xaxis=dict(tickmode='linear'))
            st.plotly_chart(fig_linha, use_container_width=True)
            
            st.markdown("---")
            
            #utilizaremos um gráfico de boxplot para mostrar a dispersão estatística das taxas por gênero, destacando a mediana, os quartis e os outliers
            fig_box = px.box(
                df_genero_taxas, x='Sex', y='VALUE', color='Sex',
                title="Dispersão Estatística e Severidade das Taxas por Gênero",
                labels={'VALUE': 'Taxa (por 100 mil hab.)', 'Sex': 'Gênero'},
                color_discrete_map={'Female': '#EF553B', 'Male': '#636EFA'}
            )
            st.plotly_chart(fig_box, use_container_width=True)

        #sub-aba2: rankings de países
        with tab_rankings:
            st.markdown("### Top Países com Maiores Taxas de Homicídio por Gênero")
            
            #top 10 Feminino e Masculino lado a lado
            col_graf_f, col_graf_m = st.columns(2)
            
            #agrupando os dados por país pegando a taxa média do período selecionado nos filtros
            df_pais_genero = df_genero_taxas.groupby(['Country', 'Sex', 'Region'])['VALUE'].mean().reset_index()
            
            #separando os dataframes para descobrir os tops locais
            df_f_pure = df_pais_genero[df_pais_genero['Sex'] == 'Female'].sort_values(by='VALUE', ascending=False).head(10)
            df_m_pure = df_pais_genero[df_pais_genero['Sex'] == 'Male'].sort_values(by='VALUE', ascending=False).head(10)
            
            #maior valor absoluto entre ambos os gêneros para travar o eixo X
            max_f = df_f_pure['VALUE'].max() if not df_f_pure.empty else 0
            max_m = df_m_pure['VALUE'].max() if not df_m_pure.empty else 0
            limite_max_x = max(max_f, max_m) * 1.15 
            
            #tratamento de seguranca
            if limite_max_x == 0:
                limite_max_x = 10
            
            #top 10 Feminino
            with col_graf_f:
                if not df_f_pure.empty:
                    df_f_top = df_f_pure.sort_values(by='VALUE', ascending=True) # Inverte para exibição em barra horizontal
                    
                    fig_top_f = px.bar(
                        df_f_top, x='VALUE', y='Country', orientation='h',
                        title='Top 10 Países: Maiores Taxas Médias (Mulheres)',
                        labels={'VALUE': 'Taxa por 100k', 'Country': 'País'},
                        text='VALUE', color='VALUE', color_continuous_scale='Reds',
                        range_x=[0, limite_max_x]
                    )
                    fig_top_f.update_traces(texttemplate='%{x:.1f}', textposition='outside')
                    fig_top_f.update_layout(coloraxis_showscale=False, height=450)
                    st.plotly_chart(fig_top_f, use_container_width=True)
                else:
                    st.info("Sem dados suficientes para gerar o ranking feminino.")
                    
            #top 10 Masculino
            with col_graf_m:
                if not df_m_pure.empty:
                    df_m_top = df_m_pure.sort_values(by='VALUE', ascending=True)
                    
                    fig_top_m = px.bar(
                        df_m_top, x='VALUE', y='Country', orientation='h',
                        title='Top 10 Países: Maiores Taxas Médias (Homens)',
                        labels={'VALUE': 'Taxa por 100k', 'Country': 'País'},
                        text='VALUE', color='VALUE', color_continuous_scale='Blues',
                        range_x=[0, limite_max_x]
                    )
                    fig_top_m.update_traces(texttemplate='%{x:.1f}', textposition='outside')
                    fig_top_m.update_layout(coloraxis_showscale=False, height=450)
                    st.plotly_chart(fig_top_m, use_container_width=True)
                else:
                    st.info("Sem dados suficientes para gerar o ranking masculino.")
                    
#aba 3: análise comparativa por regiões, focando nas taxas por 100 mil habitantes
with aba3: 
    st.subheader("Comparações e Distribuição Geográfica Macroporcional (Por Taxas)")
    
    df_reg_taxas = df_filtrado[df_filtrado['Unit of measurement'] == 'Rate per 100,000 population']
    
    if not df_reg_taxas.empty:
        
        #sub-abas
        sub_tab_crimes, sub_tab_seguranca = st.tabs(["Foco em Alta Criminalidade", "Foco em Segurança"])
        
        #sub_aba1: zonas críticas
        with sub_tab_crimes:
            st.markdown("Análise de Zonas Críticas (Maiores Taxas Médias)")
            
            #media das taxas por região para o gráfico de pizza
            df_h_regiao = df_reg_taxas.groupby('Region')['VALUE'].mean().reset_index()
            fig_donut = px.pie(
                df_h_regiao, values='VALUE', names='Region', hole=0.4,
                title="Distribuição Proporcional das Taxas Médias de Homicídios por Região",
                color_discrete_sequence=px.colors.qualitative.Safe
            )
            fig_donut.update_traces(textinfo='percent+label', textposition='inside')
            st.plotly_chart(fig_donut, use_container_width=True)
            
            st.markdown("---")
            
            #8 sub-regiões com maiores taxas medias
            df_subregiao = df_reg_taxas.groupby('Subregion')['VALUE'].mean().reset_index()
            df_subregiao_top = df_subregiao.sort_values(by='VALUE', ascending=False).head(8)
            fig_pizza = px.pie(
                df_subregiao_top, values='VALUE', names='Subregion',
                title="As 8 Sub-Regiões com Maiores Taxas Médias de Homicídios",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_pizza.update_traces(textinfo='value+percent', texttemplate='%{value:.1f} por 100k<br>(%{percent})')
            st.plotly_chart(fig_pizza, use_container_width=True)

            st.markdown("---")
            
            #paises líderes em crime por sub-região (maiores taxas)
            df_sub_country = df_reg_taxas.groupby(['Subregion', 'Country'])['VALUE'].max().reset_index()
            id_maior_sub = df_sub_country.groupby('Subregion')['VALUE'].idxmax()
            df_maior_sub = df_sub_country.loc[id_maior_sub].sort_values(by='VALUE', ascending=True)
            
            fig_crime_sub = px.bar(
                df_maior_sub, x='VALUE', y='Country', orientation='h', color='Subregion',
                title='Países com as Maiores Taxas de Homicídios por Sub-Região', text='VALUE',
                labels={'VALUE': 'Taxa (por 100 mil hab.)', 'Country': 'País'},
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            fig_crime_sub.update_traces(texttemplate='%{x:.1f}', textposition='outside')
            st.plotly_chart(fig_crime_sub, use_container_width=True)
            
            st.markdown("---")

            #paises líderes em crime por região (maiores taxas)
            df_reg_country = df_reg_taxas.groupby(['Region', 'Country'])['VALUE'].max().reset_index()
            id_maior_reg = df_reg_country.groupby('Region')['VALUE'].idxmax()
            df_maior_reg = df_reg_country.loc[id_maior_reg].sort_values(by='VALUE', ascending=True)
            
            fig_crime_reg = px.bar(
                df_maior_reg, x='VALUE', y='Country', orientation='h', color='Region',
                title='Países com as Maiores Taxas de Homicídios por Região', text='VALUE',
                labels={'VALUE': 'Taxa (por 100 mil hab.)', 'Country': 'País'},
                color_discrete_sequence=px.colors.qualitative.Set1
            )
            fig_crime_reg.update_traces(texttemplate='%{x:.1f}', textposition='outside')
            st.plotly_chart(fig_crime_reg, use_container_width=True)

        #sub_aba2: zonas seguras
        with sub_tab_seguranca:
            st.markdown("### Análise de Zonas Seguras (Menores Taxas Médias)")
            
            #distribuição Proporcional de Segurança por Região
            df_h_regiao_seg = df_reg_taxas.groupby('Region')['VALUE'].mean().reset_index()
            max_taxa = df_h_regiao_seg['VALUE'].max() if df_h_regiao_seg['VALUE'].max() > 0 else 1
            df_h_regiao_seg['Indice_Seguranca'] = (max_taxa + 1) - df_h_regiao_seg['VALUE']
            
            fig_donut_seg = px.pie(
                df_h_regiao_seg, values='Indice_Seguranca', names='Region', hole=0.4,
                title="Distribuição Proporcional de Segurança por Região (Inverso da Taxa de Homicídios)",
                color_discrete_sequence=px.colors.qualitative.Prism
            )
            fig_donut_seg.update_traces(textinfo='label+percent', textposition='inside')
            st.plotly_chart(fig_donut_seg, use_container_width=True)
            
            st.markdown("---")
            
            #8 sub--reigões com menores taxas médias
            df_subregiao_seg = df_reg_taxas.groupby('Subregion')['VALUE'].mean().reset_index()
            df_subregiao_seg_top = df_subregiao_seg.sort_values(by='VALUE', ascending=True).head(8)
            
            fig_pizza_seg = px.pie(
                df_subregiao_seg_top, values='VALUE', names='Subregion',
                title="As 8 Sub-Regiões com as Menores Taxas Médias de Homicídios",
                color_discrete_sequence=px.colors.qualitative.Pastel2
            )
            fig_pizza_seg.update_traces(textinfo='value+percent', texttemplate='%{value:.2f} por 100k')
            st.plotly_chart(fig_pizza_seg, use_container_width=True)

            st.markdown("---")
            
            #paises líderes em segurança por sub-região (menores taxas)
            #aqui evitamos usar o valor máximo para não correr o risco de pegar países com taxas negativas ou zero, que poderiam distorcer a análise de segurança. Em vez disso, pegamos a média das taxas por país e sub-região para identificar os líderes em segurança de forma mais representativa.
            df_sub_country_seg = df_reg_taxas.groupby(['Subregion', 'Country'])['VALUE'].mean().reset_index()
            
            #filtra países com taxas reais muito baixas, mas representativas (> 0)
            #se não houver nenhum > 0, ele mantém os pontos normais para não quebrar.
            df_sub_country_filtrado = df_sub_country_seg[df_sub_country_seg['VALUE'] > 0]
            if df_sub_country_filtrado.empty:
                df_sub_country_filtrado = df_sub_country_seg

            id_menor_sub = df_sub_country_filtrado.groupby('Subregion')['VALUE'].idxmin()
            df_menor_sub = df_sub_country_filtrado.loc[id_menor_sub].sort_values(by='VALUE', ascending=False)
            
            fig5 = px.bar(
                df_menor_sub, x='VALUE', y='Country', orientation='h', color='Subregion',
                title='Países com Menores Taxas Controladas de Homicídios por Sub-Região', text='VALUE',
                labels={'VALUE': 'Taxa Média (por 100 mil hab.)', 'Country': 'País'},
                color_discrete_sequence=px.colors.qualitative.Safe
            )
            fig5.update_traces(texttemplate='%{x:.3f}', textposition='outside')
            st.plotly_chart(fig5, use_container_width=True)
            
            st.markdown("---")

            #paises líderes em segurança por região
            df_reg_country_seg = df_reg_taxas.groupby(['Region', 'Country'])['VALUE'].mean().reset_index()
            df_reg_country_filtrado = df_reg_country_seg[df_reg_country_seg['VALUE'] > 0]
            if df_reg_country_filtrado.empty:
                df_reg_country_filtrado = df_reg_country_seg

            id_menor_reg = df_reg_country_filtrado.groupby('Region')['VALUE'].idxmin()
            df_menor_reg = df_reg_country_filtrado.loc[id_menor_reg].sort_values(by='VALUE', ascending=False)
            
            fig_seg_reg = px.bar(
                df_menor_reg, x='VALUE', y='Country', orientation='h', color='Region',
                title='Países de Destaque Seguro por Região (Taxas Médias Mínimas)', text='VALUE',
                labels={'VALUE': 'Taxa Média (por 100 mil hab.)', 'Country': 'País'},
                color_discrete_sequence=px.colors.qualitative.Vivid
            )
            fig_seg_reg.update_traces(texttemplate='%{x:.3f}', textposition='outside')
            st.plotly_chart(fig_seg_reg, use_container_width=True)

#aba4: análise focada no cenário brasileiro, destacando a evolução temporal e o panorama comparativo global, utilizando as taxas por 100 mil habitantes para manter a consistência com as análises anteriores
with aba4:
    st.subheader("Evolução Estatística e Histórica (Escopo Filtrado)")
    
    # 1. Garante que estamos pegando dados puramente por TAXA e Totais (sem duplicar por gênero/idade)
    df_evolucao_taxas = df_filtrado[
        (df_filtrado['Unit of measurement'] == 'Rate per 100,000 population') &
        (df_filtrado['Sex'] == 'Total') & 
        (df_filtrado['Age'] == 'Total') &
        (df_filtrado['Dimension'] == 'Total')
    ]
    
    if not df_evolucao_taxas.empty:
        
        #grafico 1: evolução temporal dinâmica (linha) para os países selecionados no filtro
        df_linha_tempo = df_evolucao_taxas.groupby(['Year', 'Country'])['VALUE'].mean().reset_index()
        df_linha_tempo = df_linha_tempo.sort_values(by='Year', ascending=True)
        qtd_paises = df_linha_tempo['Country'].nunique()
        titulo_linha = (
            f"Evolução Temporal da Taxa de Homicídios — {df_linha_tempo['Country'].iloc[0]}" 
            if qtd_paises == 1 
            else "Comparativo Histórico Temporal das Taxas de Homicídios"
        )
        
        fig_linha_dinamica = px.line(
            df_linha_tempo, x='Year', y='VALUE', color='Country' if qtd_paises > 1 else None,
            title=titulo_linha,
            labels={'VALUE': 'Taxa por 100 mil hab.', 'Year': 'Ano', 'Country': 'País'},
            markers=True,
            color_discrete_sequence=px.colors.qualitative.Dark24
        )
        
        if qtd_paises == 1:
            fig_linha_dinamica.update_traces(line_color="#870000")
            
        fig_linha_dinamica.update_layout(xaxis=dict(tickmode='linear', dtick=1))
        st.plotly_chart(fig_linha_dinamica, use_container_width=True)
        
        st.markdown("---")
        
        #grafico 2: ranking global dinâmico (top 10 países com maiores taxas médias no período selecionado)
        # pegamos a média das taxas do período selecionado para os países que estão no filtro
        df_ranking_dinamico = df_evolucao_taxas.groupby('Country')['VALUE'].mean().reset_index()
        limite_top = min(10, len(df_ranking_dinamico))
        df_top_mundo_dinamico = df_ranking_dinamico.sort_values(by='VALUE', ascending=False).head(limite_top)
        df_top_mundo_dinamico = df_top_mundo_dinamico.sort_values(by='VALUE', ascending=True) # Ordem para o gráfico horizontal
        max_valor_ranking = df_top_mundo_dinamico['VALUE'].max() if not df_top_mundo_dinamico.empty else 10
        
        fig_bars_global_top = px.bar(
            df_top_mundo_dinamico, x='VALUE', y='Country', orientation='h',
            title=f"Top {limite_top} Zonas Críticas no Escopo Filtrado (Taxa Média do Período)",
            labels={'VALUE': 'Taxa Média Registrada (por 100 mil hab.)', 'Country': 'País'},
            text='VALUE', color='VALUE', color_continuous_scale=px.colors.sequential.YlOrRd,
            range_x=[0, max_valor_ranking * 1.15]
        )
        fig_bars_global_top.update_traces(texttemplate='%{x:.1f}', textposition='outside')
        fig_bars_global_top.update_layout(coloraxis_showscale=True)
        st.plotly_chart(fig_bars_global_top, use_container_width=True)
        
    else:
        st.warning("Nenhum dado estatístico de taxa por 100 mil habitantes foi encontrado para a combinação de filtros atual.")

with aba5:
    st.subheader("Análises Específicas")

    # Pergunta01: Quais são os 10 países com as maiores taxas de homicídios por 100 mil habitantes nos últimos 5 anos?
    ano_inicio, ano_fim = 2018, 2022
    ultimos_5_anos = df_taxa[(df_taxa['Year'] >= ano_inicio) & (df_taxa['Year'] <= ano_fim)]
    df_media_paises = ultimos_5_anos.groupby('Country')['VALUE'].mean().reset_index()
    df_top_10 = df_media_paises.sort_values(by='VALUE', ascending=False).head(10)
    df_grafico = df_top_10.sort_values(by='VALUE', ascending=True)

    if not df_grafico.empty:
        fig_top10 = px.bar(
            df_grafico, 
            x='VALUE', 
            y='Country', 
            orientation='h',
            title=f'Pergunta 01: Top 10 Países com Maiores Taxas de Homicídios<br><sup>(Média de {ano_inicio} a {ano_fim})</sup>',
            text='VALUE',
            labels={'VALUE': 'Taxa Média (por 100 mil habitantes)', 'Country': 'Países'},
            color='VALUE',
            color_continuous_scale='plasma'
        )

        fig_top10.update_traces(
            texttemplate='%{x:.1f}', 
            textposition='outside'
        )
        
        fig_top10.update_layout(
            showlegend=False,
            coloraxis_showscale=False,
            height=500
        )
        st.plotly_chart(fig_top10, use_container_width=True)
    else:
        st.warning("Não há dados disponíveis para o período de 2018 a 2022.")

    st.markdown("---")

    # Pergunta02: Quais países apresentam os 10 maiores índices de homicídios de mulheres em 2022?
    homicidio_mulheres = (
        (df_base['Year'] == 2022) &
        (df_base['Sex'] == 'Female') &
        (df_base['Age'] == 'Total') &
        (df_base['Unit of measurement'] == 'Rate per 100,000 population') &
        (df_base['Indicator'] == 'Victims of intentional homicide') &
        (df_base['Dimension'] == 'Total')
    )

    df_homicidio_mulheres = df_base[homicidio_mulheres].copy()
    df_homicidio_mulheres = df_homicidio_mulheres.drop_duplicates(subset=['Country'])
    df_ordenado = df_homicidio_mulheres.sort_values(by='VALUE', ascending=False)
    df_homicidio_mulheres_2022 = df_ordenado.head(10)
    df_grafico_mulheres = df_homicidio_mulheres_2022.sort_values(by='VALUE', ascending=True)

    if not df_grafico_mulheres.empty:
        fig_mulheres = px.bar(
            df_grafico_mulheres,
            x='VALUE',
            y='Country',
            orientation='h',
            title='Pergunta 02:Top 10 Países com Maiores Índices de Homicídios de Mulheres (2022)',
            text='VALUE',
            labels={'VALUE': 'Taxa de Homicídios (por 100 mil mulheres)', 'Country': 'Países'},
            color='VALUE',
            color_continuous_scale='magma'
        )
        fig_mulheres.update_traces(
            texttemplate='%{x:.1f}', 
            textposition='outside'
        )

        fig_mulheres.update_layout(
            showlegend=False,
            coloraxis_showscale=False, 
            height=500,
            xaxis=dict(nticks=6)
        )
        st.plotly_chart(fig_mulheres, use_container_width=True)
    else:
        st.warning("Não há dados disponíveis para os homicídios de mulheres em 2022.")

    st.markdown("---")

    # Pergunta03: Quais as regiões com mais homicídios?
    df_regioes = df_base[
        (df_base['Indicator'] == 'Victims of intentional homicide') &
        (df_base['Unit of measurement'] == 'Counts')
    ].copy()

    df_homicidios_por_regiao = df_regioes.groupby('Region')['VALUE'].sum().reset_index()
    df_top_10_regioes = df_homicidios_por_regiao.sort_values(by='VALUE', ascending=False).head(10)
    df_grafico_regioes = df_top_10_regioes.sort_values(by='VALUE', ascending=True)

    if not df_grafico_regioes.empty:
        fig_regioes = px.bar(
            df_grafico_regioes,
            x='VALUE',
            y='Region',
            orientation='h',
            title='Top 10 Regiões com maior Número de Homicídios (Volume Total Acumulado)',
            text='VALUE',
            labels={'VALUE': 'Número Absoluto de Homicídios', 'Region': 'Região'},
            color='VALUE',
            color_continuous_scale='sunset'
        )

        fig_regioes.update_traces(
            texttemplate='%{x:,.0f}', 
            textposition='outside'
        )

        fig_regioes.update_layout(
            showlegend=False,
            coloraxis_showscale=False,
            height=500
        )
        st.plotly_chart(fig_regioes, use_container_width=True)
    else:
        st.warning("Não há dados de contagem absoluta disponíveis para gerar o gráfico de regiões.") 

    st.markdown("---")

    # Pergunta04: Países com o Menor Número de Homicídios por Sub-Região
    df_homicidios_subregiao = df_base[
        (df_base['Indicator'] == 'Victims of intentional homicide') &
        (df_base['Unit of measurement'] == 'Counts')
    ].copy()

    df_homicidios_por_pais_subregiao = df_homicidios_subregiao.groupby(['Subregion', 'Country'])['VALUE'].sum().reset_index()
    id_menor_homicidio = df_homicidios_por_pais_subregiao.groupby('Subregion')['VALUE'].idxmin()
    df_menor_homicidio_por_subregiao = df_homicidios_por_pais_subregiao.loc[id_menor_homicidio].sort_values(by='VALUE', ascending=True)
    df_grafico_menor_homicidio = df_menor_homicidio_por_subregiao.sort_values(by='VALUE', ascending=True)

    if not df_grafico_menor_homicidio.empty:
        fig_menor_homicidio = px.bar(
            df_grafico_menor_homicidio,
            x='VALUE',
            y='Country',
            color='Subregion',
            orientation='h',
            title='Países Líderes em Segurança: Menor Número de Homicídios por Sub-Região',
            text='VALUE',
            labels={'VALUE': 'Número bruto de Homicídios', 'Country': 'País', 'Subregion': 'Sub-região'},
            color_discrete_sequence=px.colors.sequential.Viridis
        )

        fig_menor_homicidio.update_traces(
            texttemplate='%{x:,.0f}', 
            textposition='outside'
        )

        fig_menor_homicidio.update_layout(
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.3,
                xanchor="left",
                x=0.0
            ),
            height=600
        )
        st.plotly_chart(fig_menor_homicidio, use_container_width=True)
    else:
        st.warning("Não há dados disponíveis para gerar o gráfico de menor número de homicídios por sub-região.")
    
    st.markdown("---")

    # Pergunta05: Top 10 Países com Menores Números de Homicídios de Mulheres (Todos os Anos)
    mortes_mulheres = (
        (df_base['Sex'] == 'Female') &
        (df_base['Age'] == 'Total') &
        (df_base['Unit of measurement'] == 'Counts') &
        (df_base['Indicator'] == 'Victims of intentional homicide') &
        (df_base['Dimension'] == 'Total')
    )

    df_mortes_mulheres = df_base[mortes_mulheres].copy()
    df_mortes_mulheres_por_pais = df_mortes_mulheres.groupby('Country')['VALUE'].sum().reset_index()
    df_menores_mortes_mulheres = df_mortes_mulheres_por_pais.sort_values(by='VALUE', ascending=True)
    df_top10_menores_mortes_mulheres = df_menores_mortes_mulheres.head(10)
    df_grafico_menores_mulheres = df_top10_menores_mortes_mulheres.sort_values(by='VALUE', ascending=True)

    if not df_grafico_menores_mulheres.empty:
        fig_menores_mulheres = px.bar(
            df_grafico_menores_mulheres,
            x='VALUE',
            y='Country',
            orientation='h',
            title='Top 10 Países com Menores Números de Homicídios de Mulheres<br><sup>(Volume Total Acumulado - Todos os Anos)</sup>',
            text='VALUE',
            labels={'VALUE': 'Número bruto de Homicídios', 'Country': 'Países'},
            color='VALUE',
            color_continuous_scale='viridis'
        )

        fig_menores_mulheres.update_traces(
            texttemplate='%{x:.0f}', 
            textposition='outside'
        )

        fig_menores_mulheres.update_layout(
            showlegend=False,
            coloraxis_showscale=False, 
            height=500,
            xaxis=dict(
                nticks=6,
                tickformat=',.0f' 
            )
        )
        st.plotly_chart(fig_menores_mulheres, use_container_width=True)
    else:
        st.warning("Não há dados disponíveis para gerar o gráfico de menores índices de homicídios de mulheres.")

    st.markdown("---")

    # Pergunta06: Top 10 Sub-Regiões com maior Número de Homicídios (Volume Total Acumulado)
    df_sub_base = df_base[
        (df_base['Indicator'] == 'Victims of intentional homicide') &
        (df_base['Unit of measurement'] == 'Counts')
    ].copy()

    df_subregiao = df_sub_base.groupby(['Subregion'])['VALUE'].sum().reset_index()
    df_top_10_subregiao = df_subregiao.sort_values(by='VALUE', ascending=False).head(10)
    df_grafico_subregiao = df_top_10_subregiao.sort_values(by='VALUE', ascending=True)

    if not df_grafico_subregiao.empty:
        fig_subregioes = px.bar(
            df_grafico_subregiao,
            x='VALUE',
            y='Subregion',
            orientation='h',
            title='Top 10 Sub-Regiões com maior Número de Homicídios<br><sup>(Volume Total Acumulado)</sup>',
            text='VALUE',
            labels={'VALUE': 'Número bruto de Homicídios', 'Subregion': 'Sub-região'},
            color='VALUE',
            color_continuous_scale='sunset'
        )

        fig_subregioes.update_traces(
            texttemplate='%{x:,.0f}', 
            textposition='outside'
        )

        fig_subregioes.update_layout(
            showlegend=False,
            coloraxis_showscale=False,
            height=550
        )

        st.plotly_chart(fig_subregioes, use_container_width=True)
    else:
        st.warning("Não há dados de contagem absoluta disponíveis para gerar o gráfico de sub-regiões.")

    st.markdown("---")

    # Pergunta07: País com o Maior Número de Homicídios por Continente (2020)
    homicidio_continentes = (
        (df_base['Year'] == 2020) &
        (df_base['Sex'] == 'Total') &
        (df_base['Age'] == 'Total') &
        (df_base['Unit of measurement'] == 'Counts') &
        (df_base['Indicator'] == 'Victims of intentional homicide') &
        (df_base['Dimension'] == 'Total')
    )

    df_homicidio_continentes = df_base[homicidio_continentes].copy()

    if not df_homicidio_continentes.empty:
        paises_homicidio_continente = df_homicidio_continentes.loc[df_homicidio_continentes.groupby('Region')['VALUE'].idxmax()]
        df_homicidio_continente_2020 = paises_homicidio_continente.sort_values(by='VALUE', ascending=False)
        df_grafico_continente = df_homicidio_continente_2020.sort_values(by='VALUE', ascending=True)

        fig_continentes = px.bar(
            df_grafico_continente,
            x='VALUE',
            y='Country',
            color='Region',
            orientation='h',
            title='País com o Maior Número de Homicídios por Continente (Ano de Referência: 2020)',
            text='VALUE',
            labels={'VALUE': 'Número bruto de Homicídios', 'Country': 'País', 'Region': 'Continente'},
            color_discrete_sequence=px.colors.sequential.Viridis 
        )

        fig_continentes.update_traces(
            texttemplate='%{x:,.0f}', 
            textposition='outside'
        )

        fig_continentes.update_layout(
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.3,
                xanchor="left",
                x=0.0
            ),
            height=500,
            xaxis=dict(nticks=6)
        )
        st.plotly_chart(fig_continentes, use_container_width=True)
    else:
        st.warning("Não há dados de contagem disponíveis para o ano de 2020 nos continentes.")

    st.markdown("---")

    # Pergunta08: Quais os 10 países mais violentos para as mulheres em 2021?
    homicidios_mulheres_2021 = (
        (df_base['Year'] == 2021) &
        (df_base['Sex'] == 'Female') &
        (df_base['Age'] == 'Total') &
        (df_base['Unit of measurement'] == 'Rate per 100,000 population') &
        (df_base['Indicator'] == 'Victims of intentional homicide') &
        (df_base['Dimension'] == 'Total')
    )

    df_mulheres_2021 = df_base[homicidios_mulheres_2021].copy()

    if not df_mulheres_2021.empty:
        df_mulheres_2021_limpo = df_mulheres_2021.drop_duplicates(subset=['Country'])
        df_ordenado_mulheres_2021 = df_mulheres_2021_limpo.sort_values(by='VALUE', ascending=False)
        df_top10_mulheres_2021 = df_ordenado_mulheres_2021.head(10)
        df_grafico_mulheres_2021 = df_top10_mulheres_2021.sort_values(by='VALUE', ascending=True)

        fig_mulheres_2021 = px.bar(
            df_grafico_mulheres_2021,
            x='VALUE',
            y='Country',
            orientation='h',
            title='Top 10 Países Mais Violentos para Mulheres (Ano de Referência: 2021)',
            text='VALUE',
            labels={'VALUE': 'Taxa de Homicídios (por 100 mil mulheres)', 'Country': 'País'},
            color='VALUE',
            color_continuous_scale='Reds' 
        )

        fig_mulheres_2021.update_traces(
            texttemplate='%{x:.2f}', 
            textposition='outside'
        )

        fig_mulheres_2021.update_layout(
            showlegend=False,
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_mulheres_2021, use_container_width=True)
    else:
        st.warning("Não há dados de taxas disponíveis para os homicídios de mulheres em 2021.")

    st.markdown("---")

    # Pergunta09: Top 10 Maiores Valores Absolutos de Vítimas de Homicídio Intencional
    homicidio_intencional = (
        (df_base['Sex'] == 'Total') &
        (df_base['Age'] == 'Total') &
        (df_base['Unit of measurement'] == 'Counts') &
        (df_base['Indicator'] == 'Victims of intentional homicide') &
        (df_base['Dimension'] == 'Total')
    )

    df_homicidio_intencional = df_base[homicidio_intencional].copy()
    df_homicidio_intencional = df_homicidio_intencional.drop_duplicates(subset=['Country'])
    df_homicidio_intencional_ordenado = df_homicidio_intencional.sort_values(by='VALUE', ascending=False)
    df_homicidio_intencional_top10 = df_homicidio_intencional_ordenado.head(10)
    df_grafico_absoluto = df_homicidio_intencional_top10.sort_values(by='VALUE', ascending=True)

    if not df_grafico_absoluto.empty:
        fig_absoluto = px.bar(
            df_grafico_absoluto,
            x='VALUE',
            y='Country',
            orientation='h',
            title='Top 10 Maiores Valores Absolutos de Vítimas de Homicídio Intencional',
            text='VALUE',
            labels={'VALUE': 'Número Absoluto de Vítimas', 'Country': 'Países'},
            color='VALUE',
            color_continuous_scale='Reds'
        )

        fig_absoluto.update_traces(
            texttemplate='%{x:,.0f}', 
            textposition='outside'
        )

        fig_absoluto.update_layout(
            showlegend=False,
            coloraxis_showscale=False, 
            height=500,
            xaxis=dict(
                nticks=6,
                tickformat=',.0f'
            )
        )
        st.plotly_chart(fig_absoluto, use_container_width=True)
    else:
        st.warning("Não há dados de contagem absoluta disponíveis para gerar o gráfico de maiores vítimas.")

    st.markdown("---")

    # Pergunta10: Média de homicídios no Brasil nos últimos 10 anos (2013-2022)
    filtro_brasil = (
        (df_base['Country'] == 'Brazil') &
        (df_base['Year'] >= 2013) &
        (df_base['Year'] <= 2022) &
        (df_base['Indicator'] == 'Victims of intentional homicide') &
        (df_base['Sex'] == 'Total') &
        (df_base['Age'] == 'Total') &
        (df_base['Dimension'] == 'Total') &
        (df_base['Unit of measurement'] == 'Rate per 100,000 population')
    )

    df_brasil_10anos = df_base[filtro_brasil].copy()

    if not df_brasil_10anos.empty:
        media_br = df_brasil_10anos['VALUE'].mean()
        df_brasil_ordenado = df_brasil_10anos.sort_values(by='Year', ascending=True)
        df_brasil_ordenado['Year'] = df_brasil_ordenado['Year'].astype(str)

        fig_brasil_10anos = px.bar(
            df_brasil_ordenado,
            x='VALUE',
            y='Year',
            orientation='h',
            title='Evolução da Taxa de Homicídios no Brasil (2013-2022)',
            text='VALUE',
            labels={'VALUE': 'Taxa de Homicídios (por 100 mil habitantes)', 'Year': 'Ano'},
            color='VALUE',
            color_continuous_scale='Reds' 
        )

        fig_brasil_10anos.update_traces(
            texttemplate='%{x:.2f}', 
            textposition='outside'
        )

        fig_brasil_10anos.update_layout(
            showlegend=False,
            coloraxis_showscale=False,
            height=500,
            xaxis=dict(nticks=6)
        )
        st.plotly_chart(fig_brasil_10anos, use_container_width=True)
        st.metric(
            label="Taxa Média de Homicídios no Brasil (Período 2013-2022)", 
            value=f"{media_br:.2f} por 100k hab."
        )
    else:
        st.warning("Não há dados históricos disponíveis para o Brasil no período selecionado.")

if __name__ == "__main__":
    pass
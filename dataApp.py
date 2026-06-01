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

# BARRA LATERAL - FILTROS GLOBAIS (Otimizados para evitar mapas vazios)
st.sidebar.header("Filtros Globais")

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

# INTERFACE DO USUÁRIO - ABAS E LAYOUT
st.title("Painel Interativo: Análise Global de Homicídios")
st.caption("Explore distribuições geográficas, proporções e tendências estatísticas de forma dinâmica.")

aba1, aba2, aba3, aba4 = st.tabs([
    "Tendências e Predições", 
    "Recorte de Gênero", 
    "Análise por Regiões", 
    "🇧🇷 Panorama Brasil"
])


# ABA 1: TENDÊNCIAS E PREDIÇÕES
with aba1:
    st.subheader("Modelagem Preditiva e Distribuição Geográfica")
    
    df_taxa = df_base[df_base['Unit of measurement'] == 'Rate per 100,000 population']
    df_tendencia = df_taxa.groupby('Year')['VALUE'].mean().reset_index()

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

# ABA 2: RECORTE DE GÊNERO (MULHERES)
with aba2:
    st.subheader("Análise Qualitativa: Homicídios Contra Mulheres")
    
    df_m_total = df_filtrado[(df_filtrado['Sex'] == 'Female') & (df_filtrado['Unit of measurement'] == 'Counts')]
    total_casos_m = df_m_total['VALUE'].sum() if not df_m_total.empty else 0
    
    col_kpi1, col_kpi2 = st.columns(2)
    col_kpi1.metric(label="Total Acumulado de Casos (Mulheres - Escopo Filtrado)", value=f"{total_casos_m:,.0f}".replace(",", "."))
    
    df_m_2021 = df_base[
        (df_base['Year'] == 2021) & (df_base['Sex'] == 'Female') & 
        (df_base['Unit of measurement'] == 'Rate per 100,000 population')
    ]
    if not df_m_2021.empty:
        critico_2021 = df_m_2021.loc[df_m_2021['VALUE'].idxmax()]
        col_kpi2.metric(label=f"Maior Taxa Feminina em 2021 ({critico_2021['Country']})", value=f"{critico_2021['VALUE']:.2f} por 100k")

    st.markdown("---")
    
    # Gráfico 1: Top 10 Países (Ocupando largura total)
    df_m_2022 = df_filtrado[
        (df_filtrado['Year'] == 2022) & (df_filtrado['Sex'] == 'Female') & 
        (df_filtrado['Unit of measurement'] == 'Rate per 100,000 population')
    ].drop_duplicates(subset=['Country'])
    
    if not df_m_2022.empty:
        df_m_2022_top = df_m_2022.sort_values(by='VALUE', ascending=False).head(10).sort_values(by='VALUE', ascending=True)
        fig3 = px.bar(
            df_m_2022_top, x='VALUE', y='Country', orientation='h',
            title='Top 10 Países com Maiores Índices de Homicídios de Mulheres (2022)',
            text='VALUE', color='VALUE', color_continuous_scale=px.colors.sequential.Reds
        )
        fig3.update_traces(texttemplate='%{x:.1f}', textposition='outside')
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.warning("Sem dados de taxa feminina para 2022 nos filtros atuais.")

    st.markdown("---")

    # Gráfico 2: SUBSTITUÍDO (Dispersão/Bolinhas -> Barras Verticais Limpas)
    df_m_counts = df_filtrado[
        (df_filtrado['Sex'] == 'Female') & (df_filtrado['Unit of measurement'] == 'Counts') &
        (df_filtrado['Age'] == 'Total') & (df_filtrado['Dimension'] == 'Total')
    ]
    if not df_m_counts.empty:
        df_m_acumulado = df_m_counts.groupby(['Country', 'Region'])['VALUE'].sum().reset_index()
        df_m_acumulado_top = df_m_acumulado.sort_values(by='VALUE', ascending=False).head(20)
        
        fig_bar_female_total = px.bar(
            df_m_acumulado_top, x='Country', y='VALUE', color='Region',
            title='Volume Total de Casos Acumulados de Homicídios de Mulheres (Top 20 Países)',
            labels={'VALUE': 'Casos Absolutos', 'Country': 'País'},
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        st.plotly_chart(fig_bar_female_total, use_container_width=True)

# ABA 3: ANÁLISE POR REGIÕES E SUB-REGIÕES
with aba3:
    st.subheader("Comparações e Distribuição Geográfica Macroporcional")
    
    df_reg_counts = df_filtrado[df_filtrado['Unit of measurement'] == 'Counts']
    
    if not df_reg_counts.empty:
        df_h_regiao = df_reg_counts.groupby('Region')['VALUE'].sum().reset_index()
        fig_donut = px.pie(
            df_h_regiao, values='VALUE', names='Region', hole=0.4,
            title="Distribuição Proporcional Absoluta de Crimes por Continente",
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        fig_donut.update_traces(textinfo='percent+label')
        st.plotly_chart(fig_donut, use_container_width=True)
        
        st.markdown("---")
        
        df_subregiao = df_reg_counts.groupby('Subregion')['VALUE'].sum().reset_index()
        df_subregiao_top = df_subregiao.sort_values(by='VALUE', ascending=False).head(8)
        fig_pizza = px.pie(
            df_subregiao_top, values='VALUE', names='Subregion',
            title="Representatividade das 8 Sub-Regiões Mais Críticas",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_pizza.update_traces(textinfo='percent')
        st.plotly_chart(fig_pizza, use_container_width=True)

        st.markdown("---")
        
        df_sub_country = df_reg_counts.groupby(['Subregion', 'Country'])['VALUE'].sum().reset_index()
        id_menor = df_sub_country.groupby('Subregion')['VALUE'].idxmin()
        df_menor_sub = df_sub_country.loc[id_menor].sort_values(by='VALUE', ascending=True)
        
        fig5 = px.bar(
            df_menor_sub, x='VALUE', y='Country', orientation='h', color='Subregion',
            title='Países Líderes em Segurança Absoluta por Sub-Região', text='VALUE'
        )
        fig5.update_traces(texttemplate='%{x:,.0f}', textposition='outside')
        st.plotly_chart(fig5, use_container_width=True)
        
        st.markdown("---")
        
    df_2020_counts = df_base[
        (df_base['Year'] == 2020) & (df_base['Sex'] == 'Total') & 
        (df_base['Unit of measurement'] == 'Counts') & (df_base['Age'] == 'Total')
    ]
    if not df_2020_counts.empty:
        idx_max_reg = df_2020_counts.groupby('Region')['VALUE'].idxmax()
        df_max_reg_2020 = df_2020_counts.loc[idx_max_reg].sort_values(by='VALUE', ascending=True)
        
        fig8 = px.bar(
            df_max_reg_2020, x='VALUE', y='Country', orientation='h', color='Region',
            title='Epicentros de Crimes por Continente (Ano de Referência: 2020)', text='VALUE'
        )
        fig8.update_traces(texttemplate='%{x:,.0f}', textposition='outside')
        st.plotly_chart(fig8, use_container_width=True)

# ABA 4: HISTÓRICO E FOCO NO BRASIL
with aba4:
    st.subheader("Evolução Estatística do Cenário Brasileiro")
    
    filtro_br_base = (
        (df_base['Country'] == 'Brazil') & (df_base['Year'] >= 2013) & (df_base['Year'] <= 2022) &
        (df_base['Sex'] == 'Total') & (df_base['Age'] == 'Total') &
        (df_base['Unit of measurement'] == 'Rate per 100,000 population')
    )
    df_br = df_base[filtro_br_base].copy()
    
    if not df_br.empty:
        df_br_ordenado = df_br.sort_values(by='Year', ascending=True)
        
        fig_area_br = px.area(
            df_br_ordenado, x='Year', y='VALUE',
            title="Linha Histórica Temporal da Taxa de Homicídios no Brasil (2013-2022)",
            labels={'VALUE': 'Taxa por 100 mil hab.', 'Year': 'Ano'},
            markers=True
        )
        fig_area_br.update_traces(line_color='#d62728', fillcolor='rgba(214, 39, 40, 0.2)')
        fig_area_br.update_layout(xaxis=dict(tickmode='linear', dtick=1))
        st.plotly_chart(fig_area_br, use_container_width=True)
    else:
        st.warning("Dados históricos do Brasil indisponíveis para o escopo selecionado.")

    st.markdown("---")
    
    df_geral_taxas = df_base[
        (df_base['Sex'] == 'Total') & (df_base['Age'] == 'Total') & 
        (df_base['Unit of measurement'] == 'Rate per 100,000 population')
    ].drop_duplicates(subset=['Country'])
    
    if not df_geral_taxas.empty:
        df_top_10_mundo = df_geral_taxas.sort_values(by='VALUE', ascending=False).head(10).sort_values(by='VALUE', ascending=True)
        
        fig_bars_global_top = px.bar(
            df_top_10_mundo, x='VALUE', y='Country', orientation='h',
            title="Principais Zonas Críticas Globais (Taxa Máxima Registrada)",
            labels={'VALUE': 'Taxa Máxima Registrada (por 100 mil hab.)', 'Country': 'País'},
            text='VALUE', color='VALUE', color_continuous_scale=px.colors.sequential.YlOrRd
        )
        fig_bars_global_top.update_traces(texttemplate='%{x:.1f}', textposition='outside')
        st.plotly_chart(fig_bars_global_top, use_container_width=True)

if __name__ == "__main__":
    pass
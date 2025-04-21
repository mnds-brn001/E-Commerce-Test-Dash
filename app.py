import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from utils.KPIs import load_data, calculate_kpis, calculate_acquisition_retention_kpis, filter_by_date_range, kpi_card, render_kpi_block, render_plotly_glass_card, render_kpi_block_title
from utils.insights import (
    generate_overview_insights, render_overview_insights,
    calculate_customer_behavior_insights, render_customer_behavior_insights,
    render_revenue_insights, render_satisfaction_insights,
    render_delivery_insights, render_improvement_opportunities,
    analyze_category_performance, render_category_recommendations
)
from utils.descriptions import render_page_title
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
# Importar funções de análise NLP
from utils.nlp_analysis import analyze_reviews



# Configuração da página
st.set_page_config(
    page_title="Dashboard Olist",
    page_icon="📊",
    layout="wide"
)

# Carregar dados para obter o período disponível
df = load_data()
min_date = pd.to_datetime(df['order_purchase_timestamp']).min()
max_date = pd.to_datetime(df['order_purchase_timestamp']).max()

# Sidebar
st.sidebar.title("Configurações")

# Filtro de período
st.sidebar.subheader("Período de Análise")
periodo = st.sidebar.selectbox(
    "Selecione o período:",
    [
        "Todo o período",
        "Último mês",
        "Últimos 2 meses",
        "Último trimestre",
        "Último semestre",
        "Último ano",
        "Últimos 2 anos"
    ]
)

# Calcular o período selecionado
def get_date_range(periodo):
    hoje = max_date
    if periodo == "Todo o período":
        return None
    elif periodo == "Último mês":
        return [hoje - timedelta(days=30), hoje]
    elif periodo == "Últimos 2 meses":
        return [hoje - timedelta(days=60), hoje]
    elif periodo == "Último trimestre":
        return [hoje - timedelta(days=90), hoje]
    elif periodo == "Último semestre":
        return [hoje - timedelta(days=180), hoje]
    elif periodo == "Último ano":
        return [hoje - timedelta(days=365), hoje]
    elif periodo == "Últimos 2 anos":
        return [hoje - timedelta(days=730), hoje]

# Aplicar filtro de data
date_range = get_date_range(periodo)
filtered_df = filter_by_date_range(df, date_range)

# Filtro de gasto com marketing
st.sidebar.subheader("Total Gasto com Marketing")
marketing_spend = st.sidebar.number_input(
    "Valor (R$):",
    min_value=0,
    max_value=5000000,
    value=50000,
    step=1000,
    help="Digite o valor total gasto com marketing no período selecionado"
)

# Navegação
st.sidebar.markdown("---")
st.sidebar.title("Navegação")
pagina = st.sidebar.radio(
    "Selecione a página:",
    ["Visão Geral", "Aquisição e Retenção", "Comportamento do Cliente",
    "Produtos e Categorias","Análise de Churn","Análise Estratégica"]
)

# Funções auxiliares
def format_value(value, is_integer=False):
    """Formata um valor numérico com separador de milhares e duas casas decimais."""
    if is_integer:
        return f"{int(value):,}"
    return f"{value:,.2f}"

def format_percentage(value):
    """Formata um valor como porcentagem com duas casas decimais."""
    return f"{value*100:.2f}%"

# Exibir a página selecionada
if pagina == "Visão Geral":
    kpis = calculate_kpis(filtered_df, marketing_spend, date_range)
    insights = generate_overview_insights(filtered_df)
    render_page_title("Resumo Executivo", "📊")
    
    kpi_values = {
        "💰 Receita Total": f"R$ {format_value(kpis['total_revenue'])}",
        "📦 Total de Pedidos": format_value(kpis['total_orders'], is_integer=True),
        "👥 Total de Clientes": format_value(kpis['total_customers'], is_integer=True)
    }
    render_kpi_block(kpi_values=kpi_values, cols_per_row=3)
    st.markdown("---")
    # Insights de Receita
    render_revenue_insights(insights)
    
    # Gráfico de Evolução da Receita
    monthly_revenue = insights['revenue']['monthly_revenue']
    fig_revenue = px.line(
        monthly_revenue,
        x='order_purchase_timestamp',
        y='price',
        title=" ",
        labels={'price': 'Receita (R$)', 'order_purchase_timestamp': 'Mês'}
    )
    fig_revenue.update_layout(showlegend=False)
    render_plotly_glass_card("Evolução da Receita Mensal", fig_revenue)
    
    # Experiência do Cliente
    render_page_title("Experiência do Cliente", "😊")
    
    # Métricas de Experiência
    exp_kpis = {
        "😊 Satisfação Média": format_value(kpis['csat']),
        "📦 Tempo de Entrega": f"{int(kpis['avg_delivery_time'])} dias",
        "❌ Taxa de Cancelamento": format_percentage(kpis['cancellation_rate'])
    }
    render_kpi_block(kpi_values=exp_kpis, cols_per_row=3)
    st.markdown("---")
    # Insights de Satisfação e Entrega
    col1, col2 = st.columns(2)
    with col1:
        render_satisfaction_insights(insights)
    with col2:
        render_delivery_insights(insights)
    
    st.markdown("---")
    # Gráficos de Satisfação e Cancelamento
    col1, col2 = st.columns(2)
    
    with col1:
        # Gráfico de Satisfação
        monthly_satisfaction = insights['satisfaction']['monthly_satisfaction']
        fig_satisfaction = px.line(
            monthly_satisfaction,
            x='order_purchase_timestamp',
            y='review_score',
            title=" ",
            labels={'review_score': 'Nota Média', 'order_purchase_timestamp': 'Mês'}
        )
        fig_satisfaction.update_layout(
            yaxis=dict(range=[0, 5]),
            showlegend=False
        )
        render_plotly_glass_card("Evolução da Satisfação do Cliente", fig_satisfaction)
    
    with col2:
        # Gráfico de Cancelamentos
        monthly_cancellation = insights['cancellation']['monthly_cancellation']
        fig_cancellation = px.line(
            monthly_cancellation,
            x='order_purchase_timestamp',
            y='pedido_cancelado',
            title=" ",
            labels={'pedido_cancelado': 'Taxa de Cancelamento', 'order_purchase_timestamp': 'Mês'}
        )
        fig_cancellation.update_layout(
            yaxis=dict(tickformat=".1%"),
            showlegend=False
        )
        render_plotly_glass_card("Evolução da Taxa de Cancelamento", fig_cancellation)
    
    # ===== SEÇÃO 4: OPORTUNIDADES DE MELHORIA =====
    st.markdown("## 🎯 Oportunidades de Melhoria")
    
    # Métricas de Atenção
    attention_kpis = {
        "🎯 Taxa de Abandono": format_percentage(kpis['abandonment_rate']),
        "💸 Receita Perdida": f"R$ {format_value(kpis['lost_revenue'])}",
        "💰 Ticket Médio": f"R$ {format_value(kpis['average_ticket'])}"
    }
    render_kpi_block(kpi_values=attention_kpis, cols_per_row=3)
    st.markdown("---")
    # Oportunidades de Melhoria
    render_improvement_opportunities(insights)
    

    
    
    

elif pagina == "Análise Estratégica":
    render_page_title("Análise Estratégica", "📈")
    kpis = calculate_kpis(filtered_df, marketing_spend, date_range)
    
    # ===== SEÇÃO 1: VISÃO GERAL E KPIs PRINCIPAIS =====
    # Preparar dicionário de KPIs principais
    main_kpis = {
        "💰 Receita Total": f"R$ {format_value(kpis['total_revenue'])}",
        "📈 Ticket Médio": f"R$ {format_value(kpis['average_ticket'])}",
        "👥 Total de Clientes": format_value(kpis['total_customers'], is_integer=True)
    }
    
    # Renderizar bloco de KPIs principais com efeito glass
    render_kpi_block(main_kpis, cols_per_row=3)
    
    # ===== SEÇÃO 2: PREVISÃO DE RECEITA =====
   
    # Preparar dados para análise
    filtered_df['month'] = pd.to_datetime(filtered_df['order_purchase_timestamp']).dt.to_period('M')
    monthly_category_sales = filtered_df.groupby(['month', 'product_category_name']).agg({
        'price': 'sum',
        'order_id': 'count',
        'pedido_cancelado': 'mean'
    }).reset_index()
    monthly_category_sales['month'] = monthly_category_sales['month'].astype(str)
    
    # Identificar as 5 categorias com maior volume de vendas
    top_categories = filtered_df.groupby('product_category_name')['order_id'].count().sort_values(ascending=False).head(5).index.tolist()
    
    # Filtrar apenas as categorias principais
    top_category_sales = monthly_category_sales[monthly_category_sales['product_category_name'].isin(top_categories)]
    
    # Calcular média diária de receita
    filtered_df['date'] = pd.to_datetime(filtered_df['order_purchase_timestamp']).dt.date
    daily_revenue = filtered_df.groupby('date')['price'].sum().reset_index()
    
    # Adicionar dia da semana para análise de sazonalidade
    daily_revenue['day_of_week'] = pd.to_datetime(daily_revenue['date']).dt.day_name()
    
    # Calcular média móvel de 7 dias
    daily_revenue['ma7'] = daily_revenue['price'].rolling(window=7).mean()
    
    # Calcular fatores de sazonalidade semanal
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekly_seasonality = daily_revenue.groupby('day_of_week')['price'].mean().reindex(day_order)
    weekly_seasonality = weekly_seasonality / weekly_seasonality.mean()  # Normalizar
    
    # Calcular tendência de crescimento (últimos 30 dias)
    recent_data = daily_revenue.tail(30)
    if len(recent_data) >= 2:
        x = np.arange(len(recent_data))
        y = recent_data['price'].values
        z = np.polyfit(x, y, 1)
        growth_rate = z[0]  # Coeficiente de crescimento diário
    else:
        growth_rate = 0
    
    # Calcular previsão para os próximos 30 dias
    last_date = daily_revenue['date'].iloc[-1]
    forecast_dates = pd.date_range(start=last_date, periods=31, freq='D')[1:]
    
    # Criar DataFrame para previsão
    forecast_df = pd.DataFrame({'date': forecast_dates})
    forecast_df['day_of_week'] = forecast_df['date'].dt.day_name()
    
    # Aplicar fatores de sazonalidade
    forecast_df['seasonality_factor'] = forecast_df['day_of_week'].map(weekly_seasonality)
    
    # Calcular previsão base
    base_forecast = daily_revenue['ma7'].iloc[-1]
    
    # Aplicar tendência de crescimento e sazonalidade
    for i in range(len(forecast_df)):
        days_ahead = i + 1
        forecast_df.loc[i, 'forecast'] = base_forecast * forecast_df.loc[i, 'seasonality_factor'] + (growth_rate * days_ahead)
    
    # Calcular intervalo de confiança (simplificado)
    std_dev = daily_revenue['price'].std()
    forecast_df['lower_bound'] = forecast_df['forecast'] - (1.96 * std_dev)
    forecast_df['upper_bound'] = forecast_df['forecast'] + (1.96 * std_dev)
    
    # Calcular métricas de previsão
    total_forecast = forecast_df['forecast'].sum()
    previous_30_days = daily_revenue.tail(30)['price'].sum()
    growth_percentage = (total_forecast - previous_30_days) / previous_30_days * 100 if previous_30_days > 0 else 0
    max_day = forecast_df.loc[forecast_df['forecast'].idxmax()]
    
    # Preparar dicionário de KPIs de previsão
    forecast_kpis = {
        "💰 Receita Total Prevista (30 dias)": f"R$ {format_value(total_forecast)}",
        "📈 Crescimento Previsto": f"{format_value(growth_percentage)}%",
        "📅 Dia com Maior Receita Prevista": f"{max_day['date'].strftime('%d/%m/%Y')} ({max_day['day_of_week']})"
    }
    
    # Renderizar bloco de KPIs de previsão com efeito glass
    render_kpi_block(forecast_kpis, cols_per_row=3)
    
    # Criar gráfico de previsão
    fig_forecast = go.Figure()
    
    # Adicionar dados históricos
    fig_forecast.add_trace(go.Scatter(
        x=daily_revenue['date'],
        y=daily_revenue['price'],
        name='Receita Histórica',
        line=dict(
            width=2,
            color='#1f77b4'
        ),
        mode='lines+markers',
        marker=dict(
            size=6,
            symbol='circle'
        )
    ))
    
    # Adicionar previsão
    fig_forecast.add_trace(go.Scatter(
        x=forecast_df['date'],
        y=forecast_df['forecast'],
        name='Previsão',
        line=dict(
            width=2,
            color='#ff7f0e',
            dash='dash'
        ),
        mode='lines+markers',
        marker=dict(
            size=6,
            symbol='diamond'
        )
    ))
    
    # Adicionar intervalo de confiança
    fig_forecast.add_trace(go.Scatter(
        x=forecast_df['date'].tolist() + forecast_df['date'].tolist()[::-1],
        y=forecast_df['upper_bound'].tolist() + forecast_df['lower_bound'].tolist()[::-1],
        fill='toself',
        fillcolor='rgba(255, 127, 14, 0.2)',
        line=dict(color='rgba(255, 127, 14, 0)'),
        name='Intervalo de Confiança'
    ))
    
    fig_forecast.update_layout(
        title="",
        xaxis_title="Data",
        yaxis_title="Receita (R$)",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=12)
        ),
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Rockwell"
        )
    )
    fig_forecast.update_layout(dragmode=False, hovermode=False)
    
    # Renderizar gráfico com efeito glass
    render_plotly_glass_card("🔮 Evolução e Previsão de Receita", fig_forecast)
    
    # ===== SEÇÃO 3: SAZONALIDADE E PADRÕES DE VENDA =====
    st.header("📅 Sazonalidade e Padrões de Venda")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Sazonalidade de Vendas
        
        # Calcular vendas por dia da semana
        filtered_df['day_of_week'] = pd.to_datetime(filtered_df['order_purchase_timestamp']).dt.day_name()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_revenue = filtered_df.groupby('day_of_week')['price'].sum().reindex(day_order)
        
        # Calcular vendas por mês
        filtered_df['month'] = pd.to_datetime(filtered_df['order_purchase_timestamp']).dt.month_name()
        month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                      'July', 'August', 'September', 'October', 'November', 'December']
        month_revenue = filtered_df.groupby('month')['price'].sum().reindex(month_order)
        
        # Criar gráfico de sazonalidade
        fig_seasonality = go.Figure()
        
        # Adicionar barras para dia da semana
        fig_seasonality.add_trace(go.Bar(
            x=day_revenue.index,
            y=day_revenue.values,
            name='Por Dia da Semana',
            marker_color='#1f77b4'
        ))
        
        # Adicionar barras para mês
        fig_seasonality.add_trace(go.Bar(
            x=month_revenue.index,
            y=month_revenue.values,
            name='Por Mês',
            marker_color='#ff7f0e',
            visible=False
        ))
        
        # Adicionar botões para alternar entre visualizações
        fig_seasonality.update_layout(
            title=" ",
            xaxis_title="Período",
            yaxis_title="Receita (R$)",
            updatemenus=[
                dict(
                    type="buttons",
                    direction="down",
                    buttons=[
                        dict(
                            args=[{"visible": [True, False]}],
                            label="Por Dia da Semana",
                            method="update"
                        ),
                        dict(
                            args=[{"visible": [False, True]}],
                            label="Por Mês",
                            method="update"
                        )
                    ],
                    x=0.1,
                    y=1.1
                )
            ],
            showlegend=False
        )
        fig_seasonality.update_layout(dragmode=False, hovermode=False)
        
        # Renderizar gráfico com efeito glass
        render_plotly_glass_card("📅 Sazonalidade de Vendas", fig_seasonality)
        
        # Identificar o dia da semana com maior receita
        best_day = day_revenue.idxmax()
        best_day_revenue = day_revenue.max()
        
        # Identificar o mês com maior receita
        best_month = month_revenue.idxmax()
        best_month_revenue = month_revenue.max()
        
        st.markdown(f"""
        **Insights de Sazonalidade:**
        - **Melhor dia para vendas**: {best_day} (R$ {format_value(best_day_revenue)})
        - **Melhor mês para vendas**: {best_month} (R$ {format_value(best_month_revenue)})
        """)
    
    with col2:
        # Ticket Médio por Perfil
        
        # Calcular ticket médio por estado
        state_ticket = filtered_df.groupby('customer_state')['price'].mean().sort_values(ascending=False)
        
        # Criar gráfico de ticket médio
        fig_ticket = go.Figure()
        
        fig_ticket.add_trace(go.Bar(
            x=state_ticket.index,
            y=state_ticket.values,
            name='Ticket Médio',
            marker_color='#1f77b4'
        ))
        
        fig_ticket.update_layout(
            title=" ",
            xaxis_title="Estado",
            yaxis_title="Ticket Médio (R$)",
            showlegend=False
        )
        fig_ticket.update_layout(dragmode=False, hovermode=False)
        
        # Renderizar gráfico com efeito glass
        render_plotly_glass_card("💵 Ticket Médio por Estado", fig_ticket)
        
        # Identificar o estado com maior ticket médio
        best_state = state_ticket.idxmax()
        best_state_ticket = state_ticket.max()
        
        st.markdown(f"""
        **Insights de Ticket Médio:**
        - **Estado com maior ticket médio**: {best_state} (R$ {format_value(best_state_ticket)})
        """)
    
    # ===== SEÇÃO 4: RENTABILIDADE E ANÁLISE DE CATEGORIAS =====
    st.header("💰 Rentabilidade e Análise de Categorias")
    
    # Preparar dados para análise
    filtered_df['month'] = pd.to_datetime(filtered_df['order_purchase_timestamp']).dt.to_period('M')
    monthly_category_sales = filtered_df.groupby(['month', 'product_category_name']).agg({
        'price': 'sum',
        'order_id': 'count',
        'pedido_cancelado': 'mean'
    }).reset_index()
    monthly_category_sales['month'] = monthly_category_sales['month'].astype(str)
    
    # Identificar as 5 categorias com maior volume de vendas
    top_categories = filtered_df.groupby('product_category_name')['order_id'].count().sort_values(ascending=False).head(5).index.tolist()
    
    # Filtrar apenas as categorias principais
    top_category_sales = monthly_category_sales[monthly_category_sales['product_category_name'].isin(top_categories)]
    
    # Layout em duas colunas para os gráficos
    col1, col2 = st.columns(2)
    
    with col1:
        # Top 10 Categorias por Rentabilidade
        st.subheader("📈 Top 10 Categorias por Rentabilidade")
        
        # Calcular rentabilidade por categoria
        category_profit = filtered_df.groupby('product_category_name').agg({
            'price': 'sum',
            'order_id': 'count'
        }).reset_index()
        
        category_profit['avg_price'] = category_profit['price'] / category_profit['order_id']
        category_profit['profit_margin'] = 0.3  # Simulando margem de 30%
        category_profit['profit'] = category_profit['price'] * category_profit['profit_margin']
        
        # Ordenar por lucro
        category_profit = category_profit.sort_values('profit', ascending=False).head(10)
        
        # Identificar a categoria mais rentável
        best_category = category_profit.iloc[0]['product_category_name']
        best_category_profit = category_profit.iloc[0]['profit']
        
        # Criar gráfico de rentabilidade
        fig_profit = go.Figure()
        
        fig_profit.add_trace(go.Bar(
            x=category_profit['product_category_name'],
            y=category_profit['profit'],
            name='Lucro',
            marker_color='#2ca02c'
        ))
        
        fig_profit.update_layout(
            xaxis_title="Categoria",
            yaxis_title="Lucro (R$)",
            showlegend=False,
            xaxis_tickangle=45
        )
        fig_profit.update_layout(dragmode=False, hovermode=False)
        
        # Renderizar gráfico com efeito glass
        render_plotly_glass_card("📈 Top 10 Categorias por Rentabilidade", fig_profit)
    
    with col2:
        # Taxa de Crescimento por Categoria
        st.subheader("📊 Taxa de Crescimento por Categoria")
        
        # Calcular taxa de crescimento para cada categoria
        category_growth = {}
        for category in top_categories:
            category_data = top_category_sales[top_category_sales['product_category_name'] == category]
            if len(category_data) >= 2:
                first_month = category_data.iloc[0]['order_id']
                last_month = category_data.iloc[-1]['order_id']
                growth_rate = (last_month - first_month) / first_month * 100 if first_month > 0 else 0
                category_growth[category] = growth_rate
        
        # Ordenar categorias por taxa de crescimento
        sorted_categories = sorted(category_growth.items(), key=lambda x: x[1], reverse=True)
        
        # Criar gráfico de barras para taxa de crescimento
        fig_growth = go.Figure()
        
        fig_growth.add_trace(go.Bar(
            x=[cat[0] for cat in sorted_categories],
            y=[cat[1] for cat in sorted_categories],
            name='Taxa de Crescimento',
            marker_color='#2ca02c'
        ))
        
        fig_growth.update_layout(
            xaxis_title="Categoria",
            yaxis_title="Taxa de Crescimento (%)",
            showlegend=False
        )
        fig_growth.update_layout(dragmode=False, hovermode=False)
        
        # Renderizar gráfico com efeito glass
        render_plotly_glass_card("📊 Taxa de Crescimento por Categoria", fig_growth)
    
    # ===== SEÇÃO 5: PREVISÃO DE DEMANDA POR CATEGORIA =====
    st.header("📈 Previsão de Demanda por Categoria")
    
    # Identificar as 5 categorias com maior volume de vendas
    top_categories = filtered_df.groupby('product_category_name')['order_id'].count().sort_values(ascending=False).head(5).index.tolist()
    
    # Criar DataFrame para previsão
    last_month = pd.to_datetime(monthly_category_sales['month'].iloc[-1])
    forecast_months = pd.date_range(start=last_month, periods=4, freq='M')[1:]
    
    # Calcular previsão para cada categoria
    forecast_data = []
    
    for category in top_categories:
        category_data = top_category_sales[top_category_sales['product_category_name'] == category]
        
        if len(category_data) >= 3:
            # Calcular média móvel de 3 meses
            ma3 = category_data['order_id'].rolling(window=3).mean().iloc[-1]
            
            # Calcular tendência (últimos 3 meses)
            recent_data = category_data.tail(3)
            x = np.arange(len(recent_data))
            y = recent_data['order_id'].values
            z = np.polyfit(x, y, 1)
            trend = z[0]  # Coeficiente de crescimento mensal
            
            # Calcular previsão para os próximos 3 meses
            for i, month in enumerate(forecast_months):
                forecast = ma3 + (trend * (i + 1))
                forecast_data.append({
                    'month': month,
                    'product_category_name': category,
                    'forecast': max(0, forecast)  # Garantir que a previsão não seja negativa
                })
    
    forecast_df = pd.DataFrame(forecast_data)
    
    # Criar gráfico de previsão
    fig_forecast = go.Figure()
    
    # Definir paleta de cores para as categorias
    category_colors = {
        'historical': [
            '#1f77b4',  # Azul
            '#ff7f0e',  # Laranja
            '#2ca02c',  # Verde
            '#d62728',  # Vermelho
            '#9467bd',  # Roxo
            '#8c564b',  # Marrom
            '#e377c2',  # Rosa
            '#7f7f7f',  # Cinza
            '#bcbd22',  # Verde-amarelo
            '#17becf'   # Ciano
        ],
        'forecast': [
            '#1f77b4',  # Azul (mais claro)
            '#ff7f0e',  # Laranja (mais claro)
            '#2ca02c',  # Verde (mais claro)
            '#d62728',  # Vermelho (mais claro)
            '#9467bd',  # Roxo (mais claro)
            '#8c564b',  # Marrom (mais claro)
            '#e377c2',  # Rosa (mais claro)
            '#7f7f7f',  # Cinza (mais claro)
            '#bcbd22',  # Verde-amarelo (mais claro)
            '#17becf'   # Ciano (mais claro)
        ]
    }
    
    # Adicionar dados históricos
    for i, category in enumerate(top_categories):
        category_data = top_category_sales[top_category_sales['product_category_name'] == category]
        fig_forecast.add_trace(go.Scatter(
            x=category_data['month'],
            y=category_data['order_id'],
            name=f'{category} (Histórico)',
            line=dict(
                width=2,
                color=category_colors['historical'][i % len(category_colors['historical'])]
            ),
            mode='lines+markers',
            marker=dict(
                size=6,
                symbol='circle'
            )
        ))
    
    # Adicionar previsão
    for i, category in enumerate(top_categories):
        category_forecast = forecast_df[forecast_df['product_category_name'] == category]
        if not category_forecast.empty:
            fig_forecast.add_trace(go.Scatter(
                x=category_forecast['month'],
                y=category_forecast['forecast'],
                name=f'{category} (Previsão)',
                line=dict(
                    width=2,
                    color=category_colors['forecast'][i % len(category_colors['forecast'])],
                    dash='dash'
                ),
                mode='lines+markers',
                marker=dict(
                    size=6,
                    symbol='diamond'
                )
            ))
    
    fig_forecast.update_layout(
        title="",
        xaxis_title="Mês",
        yaxis_title="Quantidade de Pedidos",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=12)
        ),
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Rockwell"
        )
    )
    fig_forecast.update_layout(dragmode=False, hovermode=False)
    
    # Renderizar gráfico com efeito glass
    render_plotly_glass_card("📈 Previsão de Demanda por Categoria", fig_forecast)
    
    # ===== SEÇÃO 6: RECOMENDAÇÕES E INSIGHTS =====
    st.header("💡 Recomendações e Insights")
    
    # Calcular métricas avançadas para recomendações
    recommendations = []
    
    # Definir limites mínimos
    MIN_MONTHLY_ORDERS = 10  # Mínimo de pedidos mensais para análise
    MIN_TOTAL_REVENUE = 5000  # Mínimo de receita total para análise
    
    for category in top_categories:
        category_data = top_category_sales[top_category_sales['product_category_name'] == category]
        category_revenue = filtered_df[filtered_df['product_category_name'] == category]['price'].sum()
        avg_monthly_orders = category_data['order_id'].mean()
        
        # Verificar volumes mínimos
        if avg_monthly_orders >= MIN_MONTHLY_ORDERS and category_revenue >= MIN_TOTAL_REVENUE:
            if not category_data.empty and not category_forecast.empty:
                # Calcular métricas de tendência
                last_month_sales = category_data.iloc[-1]['order_id']
                next_month_forecast = category_forecast.iloc[0]['forecast']
                
                # Calcular variação percentual
                variation = (next_month_forecast - last_month_sales) / last_month_sales * 100 if last_month_sales > 0 else 0
                
                # Calcular giro de estoque (simulado)
                inventory_turnover = avg_monthly_orders / 30  # Média diária de vendas
                
                # Calcular estoque ideal baseado na previsão e lead time
                lead_time_days = 15  # Tempo médio de reposição em dias
                safety_stock_days = 7  # Estoque de segurança em dias
                ideal_stock = (next_month_forecast / 30) * (lead_time_days + safety_stock_days)
                
                # Determinar ação baseada em múltiplos fatores
                if variation > 20 and inventory_turnover > 1:
                    action = "Aumentar significativamente"
                    reason = "Alto crescimento previsto com bom giro de estoque"
                elif variation > 10 and inventory_turnover > 0.5:
                    action = "Aumentar moderadamente"
                    reason = "Crescimento moderado com giro adequado"
                elif variation < -20 and inventory_turnover < 0.3:
                    action = "Reduzir significativamente"
                    reason = "Queda significativa nas vendas e baixo giro"
                elif variation < -10 and inventory_turnover < 0.5:
                    action = "Reduzir moderadamente"
                    reason = "Queda moderada nas vendas"
                else:
                    action = "Manter"
                    reason = "Demanda estável"
                
                recommendations.append({
                    'category': category,
                    'variation': variation,
                    'action': action,
                    'reason': reason,
                    'ideal_stock': ideal_stock,
                    'inventory_turnover': inventory_turnover
                })
    
    # Ordenar recomendações por variação absoluta
    recommendations.sort(key=lambda x: abs(x['variation']), reverse=True)
    
    # Exibir recomendações em um formato mais visual
    st.subheader("📦 Recomendações de Estoque")
    
    # Criar colunas para as recomendações
    rec_cols = st.columns(3)
    
    for i, rec in enumerate(recommendations):
        col_idx = i % 3
        with rec_cols[col_idx]:
            # Definir cor de fundo com base na variação
            bg_color = "rgba(46, 204, 113, 0.2)" if rec['variation'] > 0 else "rgba(231, 76, 60, 0.2)" if rec['variation'] < 0 else "rgba(52, 152, 219, 0.2)"
            
            # Definir cor do texto com base na variação
            text_color = "#2ecc71" if rec['variation'] > 0 else "#e74c3c" if rec['variation'] < 0 else "#3498db"
            
            st.markdown(f"""
            <div style="
                backdrop-filter: blur(10px);
                background: {bg_color};
                border-radius: 20px;
                padding: 30px;
                margin: 10px 0;
                border: 1px solid {text_color};
                box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
                color: #333;
                text-align: center;
            ">
                <h3 style="margin: 0; color: {text_color};">{rec['category']}</h3>
                <h1 style="margin: 10px 0; color: {text_color};">{rec['action']}</h1>
                <p style="opacity: 0.8; margin: 0;color: {text_color};">Variação prevista: {format_value(rec['variation'])}%</p>
                <p style="opacity: 0.8; margin: 5px 0;color: {text_color};">Giro de estoque: {format_value(rec['inventory_turnover'])} un/dia</p>
                <p style="opacity: 0.8; margin: 5px 0;color: {text_color};">Estoque ideal: {format_value(rec['ideal_stock'], is_integer=True)} un</p>
                <p style="font-size: 0.9em; margin-top: 10px; font-style: italic;color: {text_color};">{rec['reason']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Resumo dos insights principais
    st.subheader("📊 Resumo dos Insights Principais")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        **Insights de Receita:**
        - **Receita Total**: R$ {format_value(kpis['total_revenue'])}
        - **Crescimento Previsto**: {format_value(growth_percentage)}%
        - **Melhor dia para vendas**: {best_day}
        - **Melhor mês para vendas**: {best_month}
        """)
    
    with col2:
        st.markdown(f"""
        **Insights de Produtos:**
        - **Categoria mais rentável**: {best_category} (Lucro: R$ {format_value(best_category_profit)})
        - **Estado com maior ticket médio**: {best_state} (R$ {format_value(state_ticket.max() if 'state_ticket' in locals() else 0)})
        - **Categoria com maior crescimento**: {sorted_categories[0][0] if sorted_categories else "N/A"} ({format_value(sorted_categories[0][1] if sorted_categories else 0)}%)
        """)

elif pagina == "Aquisição e Retenção":
    render_page_title("Aquisição e Retenção", "🔄")
    kpis = calculate_kpis(filtered_df, marketing_spend, date_range)
    acquisition_kpis = calculate_acquisition_retention_kpis(filtered_df, marketing_spend, date_range)
    
    # 📊 Métricas
    
    # Preparar dicionário de KPIs de Clientes
    customer_kpis = {
        "👥 Novos Clientes (Período)": format_value(acquisition_kpis['total_new_customers'], is_integer=True),
        "🔄 Taxa de Recompra": format_percentage(acquisition_kpis['repurchase_rate']),
        "⏳ Tempo até 2ª Compra": f"{int(acquisition_kpis['avg_time_to_second'])} dias"
    }
    
    # Renderizar título e bloco de KPIs de Clientes com efeito glass
    st.markdown("---")
    render_kpi_block(kpi_values=customer_kpis, cols_per_row=3)
    
    # Preparar dicionário de KPIs Financeiros
    financial_kpis = {
        "💰 CAC": f"R$ {format_value(acquisition_kpis['cac'])}",
        "📈 LTV": f"R$ {format_value(acquisition_kpis['ltv'])}",
        "⚖️ LTV/CAC": format_value(acquisition_kpis['ltv'] / acquisition_kpis['cac'] if acquisition_kpis['cac'] > 0 else 0)
    }
    
    # Renderizar título e bloco de KPIs Financeiros com efeito glass
    st.markdown("---")
    render_kpi_block(kpi_values=financial_kpis, cols_per_row=3)
    
    st.markdown("---")
    
    # 📈 Análise LTV/CAC
    #st.header("📈 Análise LTV/CAC")
    
    # Calcular LTV e CAC por mês
    monthly_metrics = filtered_df.groupby(filtered_df['order_purchase_timestamp'].dt.to_period('M')).agg({
        'price': 'sum',
        'customer_unique_id': 'nunique',
        'pedido_cancelado': 'sum'
    }).reset_index()
    
    monthly_metrics['order_purchase_timestamp'] = monthly_metrics['order_purchase_timestamp'].astype(str)
    monthly_metrics['monthly_revenue'] = monthly_metrics['price'] - (monthly_metrics['price'] * monthly_metrics['pedido_cancelado'])
    
    # Separar cálculo do LTV da visualização
    monthly_metrics['monthly_ltv_raw'] = monthly_metrics['monthly_revenue'] / monthly_metrics['customer_unique_id']
    monthly_metrics['monthly_ltv'] = -monthly_metrics['monthly_ltv_raw']  # só para visualização
    monthly_metrics['monthly_cac'] = marketing_spend / 12
    
    # Calcular razão LTV/CAC usando o valor real (positivo)
    monthly_metrics['ltv_cac_ratio'] = monthly_metrics['monthly_ltv_raw'] / monthly_metrics['monthly_cac']
    
    # Status atual usando valores reais
    current_ltv = acquisition_kpis['ltv']  # já vem positivo
    current_cac = acquisition_kpis['cac']
    current_ratio = current_ltv / current_cac if current_cac > 0 else 0
    
    # Determinar status e cor
    if current_ratio < 1:
        status = "Alerta Crítico 🚨"
        status_color = "#dc3545"  # Vermelho para situação crítica
    elif current_ratio < 2:
        status = "Observar Insights ⚠️"
        status_color = "#ffc107"  # Amarelo para situação de atenção
    elif current_ratio < 2.85:
        status = "Operando na Linha"
        status_color = "#17a2b8"  # Azul claro para situação razoável
    elif 2.85 <= current_ratio <= 3.15:
        status = "Desempenho Ideal ✅"
        status_color = "#28a745"  # Verde para situação ideal
    else:
        status = "Desempenho Excelente 💰"
        status_color = "#007bff"  # Azul para situação excelente
    
    # Determinar cor do texto baseado no tema
    is_dark_theme = st.get_option("theme.base") == "dark"
    text_color = "rgba(255,255,255,0.9)" if is_dark_theme else "rgba(0,0,0,0.9)"
    
    # Gráfico de Evolução LTV vs CAC
    fig_comparison = go.Figure()
    
    # Adicionar dados históricos com anotação explicativa
    fig_comparison.add_trace(go.Scatter(
        x=monthly_metrics['order_purchase_timestamp'],
        y=monthly_metrics['monthly_ltv'],
        name='LTV (sinal invertido para visualização)',
        fill='tozeroy',
        line=dict(color='rgba(46, 204, 113, 0.3)'),
        fillcolor='rgba(46, 204, 113, 0.3)'
    ))
    
    fig_comparison.add_trace(go.Scatter(
        x=monthly_metrics['order_purchase_timestamp'],
        y=monthly_metrics['monthly_cac'],
        name='CAC',
        fill='tozeroy',
        line=dict(color='rgba(231, 76, 60, 0.3)'),
        fillcolor='rgba(231, 76, 60, 0.3)'
    ))
    
    fig_comparison.add_trace(go.Scatter(
        x=monthly_metrics['order_purchase_timestamp'],
        y=monthly_metrics['ltv_cac_ratio'],
        name='Razão LTV/CAC',
        line=dict(color='#2c3e50', width=2),
        yaxis='y2'
    ))
    
    # Adicionar anotação explicativa
    fig_comparison.add_annotation(
        x=0.5,
        y=1.1,
        xref="paper",
        yref="paper",
        text="Nota: O LTV está representado com sinal invertido apenas para facilitar a visualização no gráfico",
        showarrow=False,
        font=dict(size=12, color="#666")
    )
    
    fig_comparison.update_layout(
        showlegend=True,
        yaxis2=dict(
            title="Razão LTV/CAC",
            overlaying="y",
            side="right",
            showgrid=False
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Renderizar gráfico com efeito glass
    render_plotly_glass_card("📈 Evolução LTV vs CAC ao Longo do Tempo", fig_comparison)
    
    # Análise de tendência dinâmica
    if len(monthly_metrics) >= 2:
        # Calcular período analisado
        start_date = pd.to_datetime(monthly_metrics['order_purchase_timestamp'].iloc[0])
        end_date = pd.to_datetime(monthly_metrics['order_purchase_timestamp'].iloc[-1])
        meses_filtrados = ((end_date.year - start_date.year) * 12 + end_date.month - start_date.month) + 1
        
        # Calcular médias para diferentes períodos
        n_months = min(3, len(monthly_metrics))
        recent_ratio = monthly_metrics['ltv_cac_ratio'].tail(n_months).mean()
        older_ratio = monthly_metrics['ltv_cac_ratio'].head(n_months).mean()
        
        # Calcular variação percentual
        delta_percent = ((recent_ratio - older_ratio) / abs(older_ratio)) * 100 if older_ratio != 0 else 0
        
        # Determinar direção da tendência e ícone
        if abs(delta_percent) < 1:
            trend_icon = "➡️"
            trend_color = "#808080"
            trend_text = "estável"
        elif delta_percent > 0:
            trend_icon = "⬆️"
            trend_color = "#28a745"
            trend_text = "crescimento"
        else:
            trend_icon = "⬇️"
            trend_color = "#dc3545"
            trend_text = "queda"
        
        # Criar texto de período baseado no filtro
        if periodo == "Todo o período":
            periodo_texto = "no período total"
        elif periodo == "Último mês":
            periodo_texto = "no último mês"
        elif periodo == "Últimos 2 meses":
            periodo_texto = "nos últimos 2 meses"
        elif periodo == "Último trimestre":
            periodo_texto = "no último trimestre"
        elif periodo == "Último semestre":
            periodo_texto = "no último semestre"
        elif periodo == "Último ano":
            periodo_texto = "no último ano"
        elif periodo == "Últimos 2 anos":
            periodo_texto = "nos últimos 2 anos"
        
        # Layout para Status e Análise de Tendência
        col1, col2 = st.columns(2)
        
        with col1:
            # Status Card
            st.markdown(f"""
            <div style="
                backdrop-filter: blur(10px);
                background: rgba(255, 255, 255, 0.08);
                border-radius: 20px;
                padding: 25px;
                margin: 20px 0;
                border: 1px solid {status_color};
                box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
                text-align: center;
            ">
                <h3 style="margin-top: 0;">Status Atual</h3>
                <p style="font-size: 24px; font-weight: bold; color: {status_color};">{status}</p>
                <p style="font-size: 18px; color: {status_color};">Razão LTV/CAC: {format_value(current_ratio)}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Ações Recomendadas com efeito glass
            if current_ratio < 1:
                recommendations = [
                    ("📉 Reduzir o CAC", "Otimize suas campanhas de marketing para reduzir o custo de aquisição"),
                    ("📈 Aumentar o LTV", "Implemente estratégias de upselling e cross-selling"),
                    ("💰 Revisar modelo", "Avalie se o preço dos produtos/serviços está adequado")
                ]
                rec_color = "#e74c3c"  # Vermelho para situação crítica
                rec_icon = "🚨"
                rec_status = "Situação Crítica"
            elif current_ratio < 2:
                recommendations = [
                    ("📊 Otimizar CAC", "Foque em canais de aquisição mais eficientes"),
                    ("🔄 Melhorar retenção", "Implemente programas de fidelidade básicos"),
                    ("📈 Aumentar ticket", "Promova produtos de maior valor")
                ]
                rec_color = "#f39c12"  # Laranja para situação de atenção
                rec_icon = "⚠️"
                rec_status = "Necessita Melhorias"
            elif current_ratio < 2.85:
                recommendations = [
                    ("📈 Refinar CAC", "Identifique e elimine desperdícios nos canais atuais"),
                    ("💎 Premiumização", "Desenvolva ofertas de maior valor agregado"),
                    ("🔄 Programas de fidelidade", "Implemente estratégias avançadas de retenção")
                ]
                rec_color = "#f1c40f"  # Amarelo para situação razoável
                rec_icon = "😬"
                rec_status = "Próximo do Ideal"
            elif 2.85 <= current_ratio <= 3.15:
                recommendations = [
                    ("⚖️ Manter equilíbrio", "Continue com as estratégias atuais"),
                    ("📊 Monitorar métricas", "Acompanhe de perto as mudanças no CAC e LTV"),
                    ("🔄 Otimizar processos", "Busque melhorias incrementais")
                ]
                rec_color = "#2ecc71"  # Verde para situação saudável
                rec_icon = "✅"
                rec_status = "Situação Ideal"
            else:
                recommendations = [
                    ("📈 Expandir marketing", "Aumente o investimento em canais eficientes"),
                    ("🌍 Novos mercados", "Explore oportunidades de expansão"),
                    ("💎 Premiumização", "Desenvolva ofertas exclusivas de alto valor")
                ]
                rec_color = "#3498db"  # Azul para situação excelente
                rec_icon = "💰"
                rec_status = "Excelente Desempenho"
            
            # Generate recommendations HTML as a separate string
            recs_html = ""
            for title, desc in recommendations:
                recs_html += (
                    f"<li style='margin-bottom: 15px;'>"
                    f"<strong style='color: {rec_color};'>{title}:</strong> "
                    f"<span>{desc}</span>"
                    f"</li>"
                )
            
            # Build the recommendations block with minimal f-string interpolation
            recommendations_block = (
                "<div style='"
                "backdrop-filter: blur(10px);"
                "background: rgba(255,255,255,0.08);"
                "border-radius: 20px;"
                "padding: 25px;"
                "margin: 30px 0;"
                f"border: 1px solid {rec_color};"
                f"box-shadow: 0 4px 20px rgba(0,0,0,0.1);"
                "'>"
                "<div style='"
                "display: flex;"
                "align-items: center;"
                "margin-bottom: 20px;"
                "padding-bottom: 15px;"
                f"border-bottom: 1px solid {rec_color};"
                "'>"
                f"<h3 style='margin: 0;'>🎯 Ações Recomendadas</h3>"
                f"<div style='margin-left: auto; padding: 5px 12px; background: rgba({rec_color.replace('#', '')}, 0.1); border-radius: 15px; border: 1px solid {rec_color};'>"
                f"<span style='color: {rec_color};'>{rec_icon} {rec_status}</span>"
                "</div></div>"
                "<ul style='font-size: 1.1em; padding-left: 20px; line-height: 1.7; margin: 0;'>"
                f"{recs_html}"
                "</ul></div>"
            )
            
            st.markdown(recommendations_block, unsafe_allow_html=True)
        
        with col2:
            # Passo 1: Criar a tabela do guia como string separada
            guide_table = f"""
            <div style="
                backdrop-filter: blur(10px);
                background: rgba(255, 255, 255, 0.08);
                border-radius: 20px;
                padding: 25px;
                margin-bottom: 20px;
                border: 1px solid rgba(255,255,255,0.3);
                box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
            ">
                <h3 style="margin-top: 0;">📋 Guia de Interpretação: LTV/CAC</h3>
                <table style="width: 100%; border-collapse: collapse; font-size: 1.05em;">
                    <thead>
                        <tr style="border-bottom: 1px solid rgba(255,255,255,0.3);">
                            <th align="left">Faixa</th>
                            <th align="left">Interpretação</th>
                            <th align="left">Situação</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr><td>&lt; 1</td><td>Você perde dinheiro por cliente</td><td style="color: #e74c3c;">🚨 Ruim</td></tr>
                        <tr><td>1 a 2</td><td>Lucro muito baixo</td><td style="color: #f39c12;">⚠️ Limite</td></tr>
                        <tr><td>2 a 2.85</td><td>Lucro baixo</td><td style="color: #f1c40f;">😬 Razoável</td></tr>
                        <tr><td>2.85 a 3.15</td><td>Ponto ideal (clássico)</td><td style="color: #2ecc71;">✅ Saudável</td></tr>
                        <tr><td>&gt; 3.15</td><td>Lucro alto</td><td style="color: #3498db;">💰 Excelente</td></tr>
                    </tbody>
                </table>
            </div>
            """
            
            # Passo 2: Montar os blocos de análise como string segura
            analysis_blocks = f"""
            <div style="
                display: flex;
                gap: 20px;
                margin-bottom: 20px;
            ">
                <div style="
                    flex: 1;
                    backdrop-filter: blur(10px);
                    background: rgba(255, 255, 255, 0.08);
                    border-radius: 20px;
                    padding: 25px;
                    border: 1px solid rgba(255,255,255,0.3);
                    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
                ">
                    <h3 style="margin-top: 0;">📈 Análise de Tendência</h3>
                    <p style="font-size: 1.1em;">{trend_icon} A razão LTV/CAC está em <strong style='color:{trend_color};'>{trend_text}</strong></p>
                    <p style="font-size: 1.1em;">Variação de <strong>{delta_percent:+.1f}%</strong> {periodo_texto}</p>
                </div>
                <div style="
                    flex: 1;
                    backdrop-filter: blur(10px);
                    background: rgba(255, 255, 255, 0.08);
                    border-radius: 20px;
                    padding: 25px;
                    border: 1px solid rgba(255,255,255,0.3);
                    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
                ">
                    <h3 style="margin-top: 0;">📊 Detalhamento da Análise</h3>
                    <ul style="list-style-type: none; padding-left: 0; margin: 0;">
                        <li style="margin: 8px 0;">📅 Período analisado: <strong>{start_date.strftime('%b/%Y')} a {end_date.strftime('%b/%Y')}</strong></li>
                        <li style="margin: 8px 0;">📉 LTV/CAC médio período inicial: <strong>{format_value(older_ratio)}</strong></li>
                        <li style="margin: 8px 0;">📈 LTV/CAC médio período recente: <strong>{format_value(recent_ratio)}</strong></li>
                        <li style="margin: 8px 0;">📊 Meses considerados por período: <strong>{n_months}</strong></li>
                    </ul>
                </div>
            </div>
            """
            
            # Passo 3: Combinar os elementos e renderizar
            final_card = guide_table + analysis_blocks
            
            # Renderizar no Streamlit
            st.markdown(final_card, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Período insuficiente para análise de tendência (mínimo 2 meses)")
    
    st.markdown("---")
    
    # 📈 Análise de Aquisição
    st.markdown("<h2 style='text-align: center;'>📈 Análise de Aquisição</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Gráfico de Novos vs Retornando
        fig_customers = go.Figure()
        
        fig_customers.add_trace(go.Bar(
            x=acquisition_kpis['new_customers']['month'],
            y=acquisition_kpis['new_customers']['customer_unique_id'],
            name='Novos Clientes',
            marker_color='#1f77b4'
        ))
        
        fig_customers.add_trace(go.Bar(
            x=acquisition_kpis['returning_customers']['month'],
            y=acquisition_kpis['returning_customers']['customer_unique_id'],
            name='Clientes Retornando',
            marker_color='#2ca02c'
        ))
        
        fig_customers.update_layout(
            title=" ",
            barmode='stack',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Renderizar gráfico com efeito glass
        render_plotly_glass_card("👥 Evolução de Clientes", fig_customers)
    
    with col2:
        # Funil de Status dos Pedidos
        funnel_df = filtered_df.sort_values('order_purchase_timestamp')
        
        # Calcular quantidade de pedidos em cada etapa
        funnel_counts = {
            'created': len(funnel_df),
            'approved': len(funnel_df[funnel_df['order_status'].isin(['approved', 'shipped', 'delivered'])]),
            'shipped': len(funnel_df[funnel_df['order_status'].isin(['shipped', 'delivered'])]),
            'delivered': len(funnel_df[funnel_df['order_status'] == 'delivered'])
        }
        
        # Criar DataFrame para o funil
        funnel_data = pd.DataFrame({
            'status': list(funnel_counts.keys()),
            'count': list(funnel_counts.values())
        })
        
        # Definir labels em português
        status_labels = {
            'created': 'Pedidos Criados',
            'approved': 'Pedidos Aprovados',
            'shipped': 'Pedidos Enviados',
            'delivered': 'Pedidos Entregues'
        }
        
        funnel_data['status_label'] = funnel_data['status'].map(status_labels)
        
        # Criar gráfico de funil
        fig_funnel = go.Figure(go.Funnel(
            y=funnel_data['status_label'],
            x=funnel_data['count'],
            textinfo="value+percent initial",
            textposition="inside",
            marker=dict(color=["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"])
        ))
        
        fig_funnel.update_layout(
            title=" ",
            showlegend=False
        )
        
        # Renderizar gráfico com efeito glass
        render_plotly_glass_card("🔄 Funil de Pedidos", fig_funnel)
        
        # Calcular taxas de conversão entre etapas
        conversion_rates = {
            'created_to_approved': (funnel_counts['approved'] / funnel_counts['created']) * 100,
            'approved_to_shipped': (funnel_counts['shipped'] / funnel_counts['approved']) * 100,
            'shipped_to_delivered': (funnel_counts['delivered'] / funnel_counts['shipped']) * 100
        }
        
        # Determinar status e ícones baseados nas taxas
        def get_status_icon(rate):
            if rate >= 95:
                return "🟢"  # Verde para alta conversão
            elif rate >= 85:
                return "🟡"  # Amarelo para conversão média
            else:
                return "🔴"  # Vermelho para baixa conversão
        
        # Criar seção de conversão com efeito glass
        conversion_section = f"""
        <div style="
            backdrop-filter: blur(10px);
            background: rgba(255, 255, 255, 0.08);
            border-radius: 20px;
            padding: 25px;
            margin: 30px 0;
            border: 1px solid rgba(255, 255, 255, 0.3);
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        ">
            <h3 style="margin-top: 0;">🔄 Taxa de Conversão entre Etapas</h3>
            <ul style="font-size: 1.1em; padding-left: 20px;">
                <li>{get_status_icon(conversion_rates['created_to_approved'])} <strong>Pedidos Criados → Aprovados:</strong> {conversion_rates['created_to_approved']:.1f}%</li>
                <li>{get_status_icon(conversion_rates['approved_to_shipped'])} <strong>Pedidos Aprovados → Enviados:</strong> {conversion_rates['approved_to_shipped']:.1f}%</li>
                <li>{get_status_icon(conversion_rates['shipped_to_delivered'])} <strong>Pedidos Enviados → Entregues:</strong> {conversion_rates['shipped_to_delivered']:.1f}%</li>
            </ul>
            <div style="
                margin-top: 20px;
                background: rgba(0, 255, 100, 0.1);
                padding: 15px;
                border-left: 4px solid #00ff66;
                border-radius: 10px;
                font-size: 1.05em;
            ">
                💡 <strong>Insight:</strong> {
                    'Funil de pedidos operando com <strong>taxas saudáveis</strong> de conversão.'
                    if all(rate >= 95 for rate in conversion_rates.values())
                    else 'Oportunidades de melhoria identificadas nas taxas de conversão.'
                }
            </div>
        </div>
        """
        
        st.markdown(conversion_section, unsafe_allow_html=True)

elif pagina == "Comportamento do Cliente":
    render_page_title("Comportamento do Cliente", "👥")
    kpis = calculate_kpis(filtered_df, marketing_spend, date_range)
    acquisition_kpis = calculate_acquisition_retention_kpis(filtered_df, marketing_spend, date_range)
    
    # ===== SEÇÃO 1: VISÃO GERAL =====
    # Preparar dicionário de KPIs de Cliente
    st.markdown("---")
    customer_kpis = {
        "🎯 Taxa de Abandono": format_percentage(kpis['abandonment_rate']),
        "😊 Satisfação do Cliente": format_value(kpis['csat']),
        "🔄 Taxa de Recompra": format_percentage(acquisition_kpis['repurchase_rate'])
    }
    render_kpi_block(kpi_values=customer_kpis, cols_per_row=3)
    
    # Preparar dicionário de KPIs de Tempo
    st.markdown("<h2 style='text-align: center;'>⏱️ Métricas de Tempo</h2>", unsafe_allow_html=True)
    time_kpis = {
        "📦 Tempo Médio de Entrega": f"{int(kpis['avg_delivery_time'])} dias",
        "⏳ Tempo até 2ª Compra": f"{int(acquisition_kpis['avg_time_to_second'])} dias",
        "💰 Ticket Médio": f"R$ {format_value(kpis['average_ticket'])}"
    }
    render_kpi_block(kpi_values=time_kpis, cols_per_row=3)
    
    st.markdown("---")
    
    # ===== SEÇÃO 2: ANÁLISE DETALHADA =====
    render_customer_behavior_insights(filtered_df)
    
    # ===== SEÇÃO 3: ANÁLISE DE TEXTOS DAS AVALIAÇÕES =====
    st.header("📝 Análise de Textos das Avaliações")
    
    # Realizar análise NLP
    nlp_results = analyze_reviews(filtered_df)
    
    # Exibir wordclouds em três colunas
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("☀️ Avaliações Positivas")
        st.pyplot(nlp_results['positive_wordcloud'])
        
        st.markdown("**Palavras mais frequentes:**")
        for word, freq in nlp_results['positive_freq'].items():
            st.markdown(f"- {word}: {freq} ocorrências")
        
        st.markdown("**Principais tópicos (LDA):**")
        for topic in nlp_results['positive_topics_lda']:
            st.markdown(f"- {topic}")
        
        st.markdown("**Padrões encontrados:**")
        for category, count in nlp_results['sentiment_patterns']['positive'].items():
            st.markdown(f"- {category.title()}: {count} menções")

    st.markdown("---")
    with col2:
        st.subheader("⚖️ Avaliações Neutras")
        st.pyplot(nlp_results['neutral_wordcloud'])
        
        st.markdown("**Palavras mais frequentes:**")
        for word, freq in nlp_results['neutral_freq'].items():
            st.markdown(f"- {word}: {freq} ocorrências")
        
        st.markdown("**Principais tópicos (LDA):**")
        for topic in nlp_results['neutral_topics_lda']:
            st.markdown(f"- {topic}")
        
        st.markdown("**Padrões encontrados:**")
        for category, count in nlp_results['sentiment_patterns']['neutral'].items():
            st.markdown(f"- {category.title()}: {count} menções")
    
    with col3:
        st.subheader("🌧️ Avaliações Negativas")
        st.pyplot(nlp_results['negative_wordcloud'])
        
        st.markdown("**Palavras mais frequentes:**")
        for word, freq in nlp_results['negative_freq'].items():
            st.markdown(f"- {word}: {freq} ocorrências")
        
        st.markdown("**Principais tópicos (LDA):**")
        for topic in nlp_results['negative_topics_lda']:
            st.markdown(f"- {topic}")
    
        st.markdown("**Padrões encontrados:**")
        for category, count in nlp_results['sentiment_patterns']['negative'].items():
            st.markdown(f"- {category.title()}: {count} menções")
    
    # Métricas gerais
    st.markdown("---")
    
    # Preparar dicionário de KPIs de Análise de Texto
    render_kpi_block_title("📊 Métricas Gerais")
    text_analysis_kpis = {
        "📊 Total de Avaliações Positivas": nlp_results['metrics']['positive_count'],
        "📊 Total de Avaliações Neutras": nlp_results['metrics']['neutral_count'],
        "📊 Total de Avaliações Negativas": nlp_results['metrics']['negative_count']
    }
    render_kpi_block(kpi_values=text_analysis_kpis, cols_per_row=3)
    
    # Preparar dicionário de KPIs de Tamanho Médio
    render_kpi_block_title("📏 Tamanho Médio das Avaliações")
    length_kpis = {
        "📏 Tamanho Médio (Positivas)": f"{int(nlp_results['metrics']['avg_positive_length'])} caracteres",
        "📏 Tamanho Médio (Neutras)": f"{int(nlp_results['metrics']['avg_neutral_length'])} caracteres",
        "📏 Tamanho Médio (Negativas)": f"{int(nlp_results['metrics']['avg_negative_length'])} caracteres"
    }
    render_kpi_block(kpi_values=length_kpis, cols_per_row=3)
    
    # Proporções
    st.markdown("---")
    
    total_reviews = (nlp_results['metrics']['positive_count'] + 
                    nlp_results['metrics']['neutral_count'] + 
                    nlp_results['metrics']['negative_count'])
    
    # Preparar dicionário de KPIs de Proporções
    render_kpi_block_title("📈 Distribuição das Avaliações")
    proportion_kpis = {
        "📈 Proporção Positivas": f"{(nlp_results['metrics']['positive_count'] / total_reviews):.1%}",
        "📈 Proporção Neutras": f"{(nlp_results['metrics']['neutral_count'] / total_reviews):.1%}",
        "📈 Proporção Negativas": f"{(nlp_results['metrics']['negative_count'] / total_reviews):.1%}"
    }
    render_kpi_block(kpi_values=proportion_kpis, cols_per_row=3)

elif pagina == "Produtos e Categorias":
    render_page_title("Produtos e Categorias", "📦")
    st.title("Produtos e Categorias")
    kpis = calculate_kpis(filtered_df, marketing_spend, date_range)
    
    # Adicionar filtro de categorias
    st.sidebar.markdown("---")
    st.sidebar.header("🏷️ Filtros de Categoria")
    
    # Obter top categorias por volume e receita
    top_by_volume = filtered_df['product_category_name'].value_counts().head(10).index.tolist()
    top_by_revenue = filtered_df.groupby('product_category_name')['price'].sum().sort_values(ascending=False).head(10).index.tolist()
    
    # Combinar e remover duplicatas mantendo a ordem
    categorias_populares = list(dict.fromkeys(top_by_volume + top_by_revenue))
    
    # Adicionar opção "Todas as categorias" no início
    todas_categorias = ["Todas as categorias"] + categorias_populares
    
    selected_categorias = st.sidebar.multiselect(
        "Selecione as categorias",
        todas_categorias,
        default=["Todas as categorias"],
        help="Selecione 'Todas as categorias' ou escolha categorias específicas para análise"
    )
    
    # Filtrar DataFrame baseado na seleção
    if "Todas as categorias" not in selected_categorias:
        filtered_df = filtered_df[filtered_df['product_category_name'].isin(selected_categorias)]
    
    # Adicionar métricas de contexto
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Métricas das Categorias Selecionadas")
    
    # Calcular métricas para as categorias selecionadas
    total_revenue = filtered_df['price'].sum()
    total_orders = filtered_df['order_id'].nunique()
    avg_ticket = total_revenue / total_orders if total_orders > 0 else 0
    
    st.sidebar.metric("Receita Total", f"R$ {format_value(total_revenue)}")
    st.sidebar.metric("Pedidos", format_value(total_orders, is_integer=True))
    st.sidebar.metric("Ticket Médio", f"R$ {format_value(avg_ticket)}")
    
    # 📊 Visão Geral
    # Preparar dicionário de KPIs principais
    render_kpi_block_title("📊 Visão Geral")
    main_kpis = {
        "💰 Receita Total": f"R$ {format_value(kpis['total_revenue'])}",
        "📦 Total de Pedidos": format_value(kpis['total_orders']),
        "👥 Total de Clientes": format_value(kpis['total_customers']),
        "📊 Ticket Médio": f"R$ {format_value(kpis['average_ticket'])}"
    }
    render_kpi_block(kpi_values=main_kpis, cols_per_row=4)
    
    # Adicionar informação sobre o filtro ativo
    if "Todas as categorias" not in selected_categorias:
        st.info(f"📌 Mostrando dados para {len(selected_categorias)} categorias selecionadas")
    
    st.markdown("---")
    
    # 📈 Análise de Desempenho
    st.header("📈 Análise de Desempenho")
    
    # Primeira linha de gráficos
    col1, col2 = st.columns(2)
    
    with col1:
        # Top 10 Categorias por Receita
        category_revenue = filtered_df.groupby('product_category_name')['price'].sum().sort_values(ascending=False).head(10)
        fig_category = px.bar(
            x=category_revenue.index,
            y=category_revenue.values,
            title=" ",
            labels={'x': 'Categoria', 'y': 'Receita (R$)'},
            color=category_revenue.values,
            color_continuous_scale='Viridis'
        )
        fig_category.update_layout(showlegend=False)
        
        # Renderizar gráfico com efeito glass
        render_plotly_glass_card("💰 Top 10 Categorias por Receita", fig_category)
        
        # Distribuição de Preços por Categoria
        fig_price_dist = px.box(
            filtered_df,
            x='product_category_name',
            y='price',
            title=" ",
            labels={'price': 'Preço (R$)', 'product_category_name': 'Categoria'}
        )
        fig_price_dist.update_layout(showlegend=False)
        
        # Renderizar gráfico com efeito glass
        render_plotly_glass_card("💵 Distribuição de Preços por Categoria", fig_price_dist)
    
    with col2:
        # Top 10 Categorias por Quantidade
        category_quantity = filtered_df.groupby('product_category_name')['order_id'].count().sort_values(ascending=False).head(10)
        fig_quantity = px.bar(
            x=category_quantity.index,
            y=category_quantity.values,
            title=" ",
            labels={'x': 'Categoria', 'y': 'Quantidade de Pedidos'},
            color=category_quantity.values,
            color_continuous_scale='Viridis'
        )
        fig_quantity.update_layout(showlegend=False)
        
        # Renderizar gráfico com efeito glass
        render_plotly_glass_card("📦 Top 10 Categorias por Quantidade", fig_quantity)
        
        # Taxa de Cancelamento por Categoria
        category_cancellation = filtered_df.groupby('product_category_name')['pedido_cancelado'].mean().sort_values(ascending=False)
        fig_cancellation = px.bar(
            x=category_cancellation.index,
            y=category_cancellation.values,
            title=" ",
            labels={'x': 'Categoria', 'y': 'Taxa de Cancelamento'},
            color=category_cancellation.values,
            color_continuous_scale='Reds'
        )
        fig_cancellation.update_layout(
            yaxis=dict(tickformat=".1%"),
            showlegend=False
        )
        
        # Renderizar gráfico com efeito glass
        render_plotly_glass_card("❌ Taxa de Cancelamento por Categoria", fig_cancellation)
    
    # Análise de Categorias
    render_kpi_block_title("📦 Análise de Categorias")
    
    # Calcular análise de categorias
    category_analysis = analyze_category_performance(df)
    
    # Renderizar recomendações
    render_category_recommendations(category_analysis)

    st.markdown("---")
    
    # Calcular métricas por produto
    product_metrics = filtered_df.groupby(['product_id', 'product_category_name']).agg({
        'price': ['sum', 'mean', 'count'],
        'review_score': 'mean',
        'pedido_cancelado': 'mean'
    }).reset_index()
    
    # Renomear colunas
    product_metrics.columns = ['product_id', 'category', 'total_revenue', 'avg_price', 'total_sales', 'avg_rating', 'cancel_rate']
    
    # Calcular score composto para ranking
    product_metrics['composite_score'] = (
        0.4 * (product_metrics['total_revenue'] / product_metrics['total_revenue'].max()) +
        0.3 * (product_metrics['total_sales'] / product_metrics['total_sales'].max()) +
        0.2 * (product_metrics['avg_rating'] / 5) +
        0.1 * (1 - product_metrics['cancel_rate'])
    )
    
    # Identificar top produtos
    top_products = product_metrics.nlargest(5, 'composite_score')
    
    # Preparar KPIs para produtos em destaque
    top_product_kpis = {}
    for _, product in top_products.iterrows():
        top_product_kpis[f"🏆 {product['category']}"] = f"""
        <ul style="list-style-type: none; padding-left: 0; margin: 0; line-height: 1.5;">
            <li>Receita: R$ {format_value(product['total_revenue'])}</li>
            <li>Vendas: {format_value(product['total_sales'], is_integer=True)} unidades</li>
            <li>Avaliação: {format_value(product['avg_rating'])}/5.0</li>
            <li>Preço Médio: R$ {format_value(product['avg_price'])}</li>
        </ul>
        """
    
    
    
    # Análise de Preço vs. Volume
    st.subheader("📊 Análise de Preço vs. Volume de Vendas")
    
    fig_price_volume = go.Figure()
    
    # Adicionar scatter plot
    fig_price_volume.add_trace(go.Scatter(
        x=product_metrics['avg_price'],
        y=product_metrics['total_sales'],
        mode='markers',
        marker=dict(
            size=10,
            color=product_metrics['avg_rating'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='Avaliação Média')
        ),
        text=product_metrics['category'],
        hovertemplate=
        '<b>%{text}</b><br>' +
        'Preço Médio: R$%{x:.2f}<br>' +
        'Vendas: %{y}<br>' +
        '<extra></extra>'
    ))
    
    fig_price_volume.update_layout(
        xaxis_title="Preço Médio (R$)",
        yaxis_title="Volume de Vendas",
        showlegend=False
        )
        
        # Renderizar gráfico com efeito glass
    render_plotly_glass_card("📊 Relação Preço vs. Volume de Vendas", fig_price_volume)


    # Renderizar KPIs dos produtos em destaque
    render_kpi_block_title("🏆 Top 5 Produtos")
    render_kpi_block(kpi_values=top_product_kpis, cols_per_row=5)


    # Insights sobre os produtos
    st.title("📊 Insights e Recomendações")
    
    

    
    # Calcular métricas gerais
    avg_price_market = product_metrics['avg_price'].mean()
    avg_rating_market = product_metrics['avg_rating'].mean()
    
    # Identificar produtos premium (alto preço, alta avaliação)
    premium_products = product_metrics[
        (product_metrics['avg_price'] > avg_price_market) &
        (product_metrics['avg_rating'] > avg_rating_market)
    ]
    
    # Identificar produtos populares (alto volume, preço acessível)
    popular_products = product_metrics[
        (product_metrics['total_sales'] > product_metrics['total_sales'].mean()) &
        (product_metrics['avg_price'] <= avg_price_market)
    ]
    
    st.markdown(f"""
    #### 📈 Análise de Mercado
    
    1. **Produtos Premium** (Alto preço, Alta avaliação)
       - {len(premium_products)} produtos identificados
       - Preço médio: R$ {format_value(premium_products['avg_price'].mean())}
       - Avaliação média: {format_value(premium_products['avg_rating'].mean())}/5.0
    
    2. **Produtos Populares** (Alto volume, Preço acessível)
       - {len(popular_products)} produtos identificados
       - Volume médio: {format_value(popular_products['total_sales'].mean(), is_integer=True)} unidades
       - Preço médio: R$ {format_value(popular_products['avg_price'].mean())}
    
    3. **Métricas de Mercado**
       - Preço médio do mercado: R$ {format_value(avg_price_market)}
       - Avaliação média do mercado: {format_value(avg_rating_market)}/5.0
    """)

elif pagina == "Análise de Churn":
    render_page_title("Análise de Churn", "📉")
    import paginas.analise_churn
    paginas.analise_churn.app()
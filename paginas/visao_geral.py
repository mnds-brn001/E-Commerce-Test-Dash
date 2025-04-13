import streamlit as st
import pandas as pd
from utils.KPIs import calculate_kpis, load_data
import matplotlib.pyplot as plt
import plotly.express as px

def format_value(value, is_integer=False):
    """Formata valores numéricos ou Series para exibição."""
    if isinstance(value, pd.Series):
        value = value.sum()  # ou .mean() dependendo do que você quer mostrar
    if isinstance(value, (int, float)):
        if is_integer:
            return f"{int(value):,}".replace(",", ".")
        return f"{value:,.2f}".replace(",", ".")
    return str(value)

def format_percentage(value):
    """Formata valores como porcentagem."""
    if isinstance(value, pd.Series):
        value = value.sum()
    if isinstance(value, (int, float)):
        return f"{value:.2%}"
    return str(value)

def show(marketing_spend=50000):
    df = load_data()
    kpis = calculate_kpis(df, marketing_spend)

    st.title("Visão Geral")

    # Layout dos KPIs
    col1, col2, col3 = st.columns(3)

    # Primeira linha de KPIs
    col1.metric("💵 Receita Total", f"R$ {format_value(kpis['Receita Total'])}")
    col2.metric("📑 Total de Pedidos", format_value(kpis['Número Total de Pedidos'], is_integer=True))
    col3.metric("👥 Total de Clientes", format_value(kpis['Total de Clientes'], is_integer=True))

    # Segunda linha de KPIs
    col1.metric("📦 Ticket Médio (AOV)", f"R$ {format_value(kpis['Valor Medio do Pedido (AOV)'])}")
    col2.metric("🔁 LTV", f"R$ {format_value(kpis['Lifetime Value (LTV)'])}")
    col3.metric("💰 CAC", f"R$ {format_value(kpis['Custo de Aquisição de Cliente (CAC)'])}")

    # Terceira linha de KPIs
    col1.metric("😊 CSAT", f"{format_value(kpis['Índice de Satisfação do Cliente (CSAT)'])}/5")
    col2.metric("⏳ Tempo Médio de Entrega", f"{format_value(kpis['Tempo Médio de Entrega (dias)'])} dias")
    col3.metric("❌ Pedidos Cancelados", format_percentage(kpis['Taxa de Cancelamento']))

    # Quarta linha de KPIs
    col1.metric("💸 Receita Perdida", f"R$ {format_value(kpis['Receita Perdida'])}")
    col2.metric("💸 Receita Perdida (%)", format_percentage(kpis['Receita Perdida (%)']))
    
    st.markdown("---")

    # Gráficos
    col1, col2 = st.columns(2)

    with col1:
        # Gráfico de Receita Mensal
        st.subheader("📅 Receita Mensal")
        fig_revenue = px.line(kpis['monthly_revenue'], 
                            x='month_str', 
                            y='price',
                            title="Evolução da Receita Mensal",
                            labels={'price': 'Receita', 'month_str': 'Mês'})
        fig_revenue.update_layout(dragmode=False, hovermode=False)
        st.plotly_chart(fig_revenue, use_container_width=True)

        # Gráfico de Status dos Pedidos
        st.subheader("📊 Status dos Pedidos")
        fig_status = px.pie(values=kpis['order_status'].values, 
                          names=kpis['order_status'].index,
                          title="Distribuição de Status dos Pedidos")
        fig_status.update_layout(dragmode=False, hovermode=False)
        st.plotly_chart(fig_status, use_container_width=True)

    with col2:
        # Gráfico de Evolução do Ticket Médio
        st.subheader("📈 Evolução do Ticket Médio")
        fig_aov = px.line(kpis['monthly_aov'],
                         x='month_str',
                         y='aov',
                         title="Evolução do Ticket Médio por Mês",
                         labels={'aov': 'Ticket Médio', 'month_str': 'Mês'})
        fig_aov.update_layout(dragmode=False, hovermode=False)
        st.plotly_chart(fig_aov, use_container_width=True)

        # Gráfico de Pedidos por Estado
        st.subheader("🗺️ Pedidos por Estado")
        fig_state = px.bar(kpis['orders_by_state'],
                          title="Distribuição de Pedidos por Estado",
                          labels={'value': 'Número de Pedidos', 'customer_state': 'Estado'})
        fig_state.update_layout(dragmode=False, hovermode=False)
        st.plotly_chart(fig_state, use_container_width=True)

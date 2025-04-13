import streamlit as st
import pandas as pd
from utils.KPIs import calculate_kpis, load_data
import plotly.express as px
import plotly.graph_objects as go

def format_value(value, is_integer=False):
    """Formata valores numéricos ou Series para exibição."""
    if isinstance(value, pd.Series):
        value = value.sum()
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

def show(marketing_spend=50000, date_range=None):
    df = load_data()
    kpis = calculate_kpis(df, marketing_spend, date_range)

    st.title("Aquisição e Retenção")

    # Layout dos KPIs
    col1, col2, col3 = st.columns(3)

    # Primeira linha de KPIs
    col1.metric("👥 Novos Clientes (Mês)", format_value(kpis['new_customers']['customer_unique_id'].iloc[-1], is_integer=True))
    col2.metric("🔄 Taxa de Recompra", format_percentage(kpis['repurchase_rate']))
    col3.metric("⏳ Tempo até 2ª Compra", f"{format_value(kpis['avg_time_to_second'])} dias")

    # Segunda linha de KPIs
    col1.metric("💰 CAC", f"R$ {format_value(kpis['cac'])}")
    col2.metric("🔁 LTV", f"R$ {format_value(kpis['ltv'])}")
    col3.metric("📈 LTV/CAC", format_value(kpis['ltv'] / kpis['cac'] if kpis['cac'] > 0 else 0))

    st.markdown("---")

    # Gráficos
    col1, col2 = st.columns(2)

    with col1:
        # Gráfico de Novos vs Retornando
        st.subheader("👥 Novos vs Clientes Retornando")
        fig_customers = go.Figure()
        
        # Adicionar novos clientes
        fig_customers.add_trace(go.Bar(
            x=kpis['new_customers']['month'],
            y=kpis['new_customers']['customer_unique_id'],
            name='Novos Clientes',
            marker_color='#1f77b4'
        ))
        
        # Adicionar clientes retornando
        fig_customers.add_trace(go.Bar(
            x=kpis['returning_customers']['month'],
            y=kpis['returning_customers']['customer_unique_id'],
            name='Clientes Retornando',
            marker_color='#2ca02c'
        ))
        
        fig_customers.update_layout(
            title="Evolução de Novos e Clientes Retornando",
            barmode='stack',
            xaxis_title="Mês",
            yaxis_title="Número de Clientes",
            yaxis=dict(tickformat=",d")  # Formato inteiro para o eixo Y
        )
        fig_customers.update_layout(dragmode=False, hovermode=False)
        st.plotly_chart(fig_customers, use_container_width=True)

        # Gráfico de Evolução dos Novos Clientes
        st.subheader("📈 Evolução dos Novos Clientes")
        fig_new = px.line(kpis['new_customers'],
                         x='month',
                         y='customer_unique_id',
                         title="Evolução Mensal de Novos Clientes",
                         labels={'customer_unique_id': 'Novos Clientes', 'month': 'Mês'})
        fig_new.update_layout(
            yaxis=dict(tickformat=",d"),  # Formato inteiro para o eixo Y
            showlegend=False
        )
        fig_new.update_layout(dragmode=False, hovermode=False)
        st.plotly_chart(fig_new, use_container_width=True)

    with col2:
        # Funil de Conversão
        st.subheader("🔄 Funil de Conversão")
        fig_funnel = go.Figure(go.Funnel(
            y=kpis['funnel_data']['Etapa'],
            x=kpis['funnel_data']['Quantidade'],
            textinfo="value+percent initial",
            textposition="inside",
            marker=dict(color=["#1f77b4", "#ff7f0e", "#2ca02c"])
        ))
        fig_funnel.update_layout(
            title="Funil de Conversão",
            showlegend=False
        )
        fig_funnel.update_layout(dragmode=False, hovermode=False)
        st.plotly_chart(fig_funnel, use_container_width=True)

        # Comparativo LTV vs CAC
        st.subheader("💰 LTV vs CAC")
        fig_comparison = go.Figure()
        fig_comparison.add_trace(go.Bar(
            name='LTV',
            x=['LTV'],
            y=[kpis['ltv']],
            marker_color='#2ca02c'
        ))
        fig_comparison.add_trace(go.Bar(
            name='CAC',
            x=['CAC'],
            y=[kpis['cac']],
            marker_color='#ff7f0e'
        ))
        fig_comparison.update_layout(
            title="Comparativo LTV vs CAC",
            yaxis_title="Valor (R$)",
            showlegend=True
        )
        fig_comparison.update_layout(dragmode=False, hovermode=False)
        st.plotly_chart(fig_comparison, use_container_width=True)

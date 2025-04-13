import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from utils.KPIs import load_data, calculate_kpis, calculate_acquisition_retention_kpis
import pandas as pd

def format_value(value, is_integer=False):
    """Formata um valor numérico com separador de milhares e duas casas decimais."""
    if is_integer:
        return f"{int(value):,}"
    return f"{value:,.2f}"

def format_percentage(value):
    """Formata um valor como porcentagem com duas casas decimais."""
    return f"{value*100:.2f}%"

def show(marketing_spend=50000, date_range=None):
    df = load_data()
    kpis = calculate_kpis(df, marketing_spend, date_range)
    acquisition_kpis = calculate_acquisition_retention_kpis(df, marketing_spend, date_range)

    st.title("Comportamento do Cliente")

    # Layout dos KPIs
    col1, col2, col3 = st.columns(3)

    # Primeira linha de KPIs
    col1.metric("🎯 Taxa de Abandono", format_percentage(kpis['abandonment_rate']))
    col2.metric("😊 Satisfação do Cliente", format_value(kpis['csat']))
    col3.metric("💰 Ticket Médio", f"R$ {format_value(kpis['average_ticket'])}")

    # Segunda linha de KPIs
    col1.metric("📦 Tempo Médio de Entrega", f"{format_value(kpis['avg_delivery_time'])} dias")
    col2.metric("🔄 Taxa de Recompra", format_percentage(acquisition_kpis['repurchase_rate']))
    col3.metric("⏳ Tempo até 2ª Compra", f"{format_value(acquisition_kpis['avg_time_to_second'])} dias")

    st.markdown("---")

    # Gráficos
    col1, col2 = st.columns(2)

    with col1:
        # Gráfico de Satisfação do Cliente ao Longo do Tempo
        st.subheader("😊 Satisfação do Cliente ao Longo do Tempo")
        fig_satisfaction = px.line(
            df.groupby(df['order_purchase_timestamp'].dt.to_period('M'))['review_score'].mean().reset_index(),
            x='order_purchase_timestamp',
            y='review_score',
            title="Evolução da Satisfação do Cliente",
            labels={'review_score': 'Nota Média', 'order_purchase_timestamp': 'Mês'}
        )
        fig_satisfaction.update_layout(
            yaxis=dict(range=[0, 5]),
            showlegend=False
        )
        fig_satisfaction.update_layout(dragmode=False, hovermode=False)
        st.plotly_chart(fig_satisfaction, use_container_width=True)

        # Gráfico de Distribuição de Satisfação
        st.subheader("📊 Distribuição de Satisfação")
        fig_dist = px.histogram(
            df,
            x='review_score',
            title="Distribuição das Avaliações",
            labels={'review_score': 'Nota', 'count': 'Quantidade de Avaliações'}
        )
        fig_dist.update_layout(
            xaxis=dict(range=[0, 5]),
            showlegend=False
        )
        fig_dist.update_layout(dragmode=False, hovermode=False)
        st.plotly_chart(fig_dist, use_container_width=True)

    with col2:
        # Gráfico de Tempo de Entrega ao Longo do Tempo
        st.subheader("📦 Tempo de Entrega ao Longo do Tempo")
        df['delivery_time'] = (pd.to_datetime(df['order_delivered_customer_date']) - 
                             pd.to_datetime(df['order_purchase_timestamp'])).dt.days
        fig_delivery = px.line(
            df.groupby(df['order_purchase_timestamp'].dt.to_period('M'))['delivery_time'].mean().reset_index(),
            x='order_purchase_timestamp',
            y='delivery_time',
            title="Evolução do Tempo de Entrega",
            labels={'delivery_time': 'Dias', 'order_purchase_timestamp': 'Mês'}
        )
        fig_delivery.update_layout(showlegend=False)
        fig_delivery.update_layout(dragmode=False, hovermode=False)
        st.plotly_chart(fig_delivery, use_container_width=True)

        # Gráfico de Ticket Médio ao Longo do Tempo
        st.subheader("💰 Ticket Médio ao Longo do Tempo")
        fig_ticket = px.line(
            df.groupby(df['order_purchase_timestamp'].dt.to_period('M'))['price'].mean().reset_index(),
            x='order_purchase_timestamp',
            y='price',
            title="Evolução do Ticket Médio",
            labels={'price': 'Valor Médio (R$)', 'order_purchase_timestamp': 'Mês'}
        )
        fig_ticket.update_layout(showlegend=False)
        fig_ticket.update_layout(dragmode=False, hovermode=False)
        st.plotly_chart(fig_ticket, use_container_width=True)

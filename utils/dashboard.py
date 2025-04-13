import streamlit as st
import pandas as pd
from utils.KPIs import calculate_kpis
import matplotlib.pyplot as plt
import plotly.express as px


def show():
    st.title("📊 E-commerce Dashboard")
    
    # Carregar os dados
    df = pd.read_parquet("olist_merged_data.parquet")

    # Calcular os KPIs
    kpis = calculate_kpis(df)

    # Exibir os KPIs em cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🛍️ Taxa de Conversão", f"{kpis['Taxa de Conversão']:.2%}")
    col2.metric("💰 CAC", f"R${kpis['Custo de Aquisição de Cliente (CAC)']:.2f}")
    col3.metric("📈 LTV", f"R${kpis['Lifetime Value (LTV)']:.2f}")
    col4.metric("🛒 AOV", f"R${kpis['Valor Medio do Pedido (AOV)']:.2f}")

    st.markdown("---")

    # Gráfico de Receita ao longo do tempo
    st.subheader("📅 Receita ao Longo do Tempo")
    revenue_by_date = df.groupby("order_purchase_timestamp")["price"].sum().reset_index()
    fig = px.line(revenue_by_date, x="order_purchase_timestamp", y="price", title="Receita ao Longo do Tempo")

    # Desativar interações pesadas
    fig.update_layout(dragmode=False, hovermode=False)

    st.plotly_chart(fig, use_container_width=True)
    # Gráfico de CSAT
    st.subheader("🔎 Índice de Satisfação do Cliente (CSAT)")
    review_scores = df["review_score"].value_counts().sort_index()
    st.bar_chart(review_scores)

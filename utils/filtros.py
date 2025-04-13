import streamlit as st
import pandas as pd

def show():
    st.title("🔍 Filtros para Análise")

    # Carregar os dados
    df = pd.read_parquet("olist_merged_data.parquet")

    # Filtro por categoria de produto
    categorias = df["product_category_name"].unique().tolist()
    categoria_selecionada = st.selectbox("Selecione a Categoria:", ["Todas"] + categorias)
    
    if categoria_selecionada != "Todas":
        df = df[df["product_category_name"] == categoria_selecionada]

    # Filtro por período
    st.sidebar.subheader("📅 Selecione o Período")
    data_inicio = st.sidebar.date_input("Data Inicial", df["order_purchase_timestamp"].min())
    data_fim = st.sidebar.date_input("Data Final", df["order_purchase_timestamp"].max())

    df = df[(df["order_purchase_timestamp"] >= str(data_inicio)) & (df["order_purchase_timestamp"] <= str(data_fim))]

    # Exibir os dados filtrados
    st.dataframe(df.head())
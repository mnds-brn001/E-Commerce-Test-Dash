import pandas as pd
import numpy as np
import streamlit as st

@st.cache_data  
def load_and_merge_olist_data():
    # Carregar datasets
    orders = pd.read_csv("olist_orders_dataset.csv")
    customers = pd.read_csv("olist_customers_dataset.csv")
    order_items = pd.read_csv("olist_order_items_dataset.csv")
    payments = pd.read_csv("olist_order_payments_dataset.csv")
    reviews = pd.read_csv("olist_order_reviews_dataset.csv")
    products = pd.read_csv("olist_products_dataset.csv")
    sellers = pd.read_csv("olist_sellers_dataset.csv")
    geolocation = pd.read_csv("olist_geolocation_dataset.csv")
    category_translation = pd.read_csv("product_category_name_translation.csv")
    
    # Converter coluna de data para datetime
    orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
    
    # Filtrar dados até julho de 2018
    cutoff_date = pd.to_datetime('2018-08-01')
    orders = orders[orders['order_purchase_timestamp'] < cutoff_date]
    
    # Merge principal: orders + customers
    df = orders.merge(customers, on='customer_id', how='left')
    
    # Adicionar detalhes dos itens do pedido
    df = df.merge(order_items, on='order_id', how='left')
    
    # Adicionar pagamentos
    df = df.merge(payments, on='order_id', how='left')
    
    # Adicionar avaliações
    df = df.merge(reviews, on='order_id', how='left')
    
    # Adicionar detalhes do produto
    df = df.merge(products, on='product_id', how='left')
    
    # Adicionar nome da categoria traduzido
    df = df.merge(category_translation, on='product_category_name', how='left')
    
    # Adicionar informações dos vendedores
    df = df.merge(sellers, on='seller_id', how='left')

    # Criar uma flag para identificar pedidos cancelados (aleatório)
    np.random.seed(42)  # Garantir reprodutibilidade
    df["pedido_cancelado"] = np.random.choice([0, 1], size=len(df), p=[0.9, 0.1])  # 10% cancelados

    # Simular uma coluna de carrinhos abandonados (aleatório, baseado nos clientes)
    df["carrinho_abandonado"] = np.random.choice([0, 1], size=len(df), p=[0.85, 0.15])  # 15% abandonados

    # Receita perdida com pedidos cancelados
    df["receita_perdida"] = df["price"] * df["pedido_cancelado"]

    # Simular valores de CSAT (Customer Satisfaction Score) entre 1 e 5
    df["csat_score"] = np.random.randint(1, 6, size=len(df))

    
    # Salvar como CSV e Parquet
    df.to_csv("olist_merged_data.csv", index=False)
    df.to_parquet("olist_merged_data.parquet", index=False)
    
    print("Dataset consolidado salvo com sucesso!")

if __name__ == "__main__":
    load_and_merge_olist_data()
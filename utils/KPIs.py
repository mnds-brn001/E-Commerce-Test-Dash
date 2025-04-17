import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

@st.cache_data
def load_data():
    """Carrega os dados consolidados do Olist."""
    return pd.read_parquet("olist_merged_data.parquet")

def filter_by_date_range(df, date_range):
    """Filtra o DataFrame pelo período selecionado."""
    if not date_range or len(date_range) != 2:
        return df
    
    # Garantir que a coluna de timestamp está no formato datetime
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
    
    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[1])
    
    return df[
        (df['order_purchase_timestamp'] >= start_date) & 
        (df['order_purchase_timestamp'] <= end_date)
    ]

def calculate_acquisition_retention_kpis(df, marketing_spend=50000, date_range=None):
    """Calcula KPIs específicos para análise de aquisição e retenção."""
    
    # Filtrar dados pelo período
    df = filter_by_date_range(df, date_range)
    
    # Converter colunas de data para datetime
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
    
    # Identificar novos vs clientes recorrentes por mês
    df['month'] = df['order_purchase_timestamp'].dt.to_period('M')
    df['month_str'] = df['month'].astype(str)
    
    # Identificar primeira compra de cada cliente
    first_purchases = df.groupby('customer_unique_id')['order_purchase_timestamp'].min().reset_index()
    first_purchases['month'] = first_purchases['order_purchase_timestamp'].dt.to_period('M')
    
    # Novos clientes por mês (corrigido)
    new_customers = first_purchases.groupby('month')['customer_unique_id'].count().reset_index()
    new_customers['month'] = new_customers['month'].astype(str)
    
    # Total de novos clientes no período (corrigido)
    total_new_customers = first_purchases['customer_unique_id'].nunique()
    
    # Clientes recorrentes por mês (corrigido)
    # Primeiro, identificar todas as compras de cada cliente por mês
    customer_orders = df.groupby(['customer_unique_id', 'month'])['order_id'].count().reset_index()
    customer_orders['month'] = customer_orders['month'].astype(str)
    
    # Depois, identificar clientes que fizeram mais de uma compra no mês
    returning_customers = customer_orders[customer_orders['order_id'] > 1].groupby('month')['customer_unique_id'].nunique()
    returning_customers = returning_customers.reset_index()
    
    # Taxa de recompra (corrigido)
    total_customers = df['customer_unique_id'].nunique()
    customers_with_multiple_orders = df.groupby('customer_unique_id')['order_id'].nunique()
    customers_with_multiple_orders = customers_with_multiple_orders[customers_with_multiple_orders > 1].count()
    repurchase_rate = customers_with_multiple_orders / total_customers if total_customers > 0 else 0
    
    # Tempo médio até segunda compra (corrigido)
    customer_orders = df.groupby('customer_unique_id')['order_purchase_timestamp'].apply(list).reset_index()
    customer_orders['time_to_second_purchase'] = customer_orders['order_purchase_timestamp'].apply(
        lambda x: (x[1] - x[0]).days if len(x) > 1 else None
    )
    # Filtrar apenas clientes com múltiplas compras e tempo positivo
    valid_times = customer_orders['time_to_second_purchase'].dropna()
    valid_times = valid_times[valid_times > 0]  # Apenas tempos positivos
    avg_time_to_second = valid_times.mean() if not valid_times.empty else 0
    
    # CAC (corrigido)
    cac = marketing_spend / total_new_customers if total_new_customers > 0 else 0
    
    # LTV (corrigido)
    # Calcular receita total de pedidos não cancelados
    total_revenue = df[df["pedido_cancelado"] == 0]["price"].sum()
    # Calcular número total de clientes únicos
    total_customers = df["customer_unique_id"].nunique()
    # LTV = Receita total / Número total de clientes
    ltv = total_revenue / total_customers if total_customers > 0 else 0
    
    # Funil de conversão (simulado)
    funnel_data = {
        'Etapa': ['Visitantes', 'Carrinhos', 'Compras'],
        'Quantidade': [100000, 50000, total_new_customers]  # Valores simulados
    }
    
    return {
        "new_customers": new_customers,
        "returning_customers": returning_customers,
        "repurchase_rate": repurchase_rate,
        "avg_time_to_second": avg_time_to_second,
        "cac": cac,
        "ltv": ltv,
        "funnel_data": pd.DataFrame(funnel_data),
        "total_new_customers": total_new_customers
    }

def calculate_kpis(df, marketing_spend=50000, date_range=None):
    """Calcula os principais KPIs do negócio."""
    
    # Filtrar dados pelo período
    df = filter_by_date_range(df, date_range)
    
    # Converter colunas de data para datetime
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
    df['order_delivered_customer_date'] = pd.to_datetime(df['order_delivered_customer_date'])
    
    # Calcular KPIs
    total_revenue = df[df["pedido_cancelado"] == 0]["price"].sum()
    total_orders = df["order_id"].nunique()
    total_customers = df["customer_unique_id"].nunique()
    total_products = df["product_id"].nunique()
    unique_categories = df["product_category_name"].nunique()
    
    # Taxa de abandono (corrigido para considerar o período)
    total_cart_abandonments = df[df["pedido_cancelado"] == 1]["order_id"].nunique()
    total_carts = df["order_id"].nunique()
    abandonment_rate = total_cart_abandonments / total_carts if total_carts > 0 else 0
    
    # CSAT
    csat = df["review_score"].mean()
    
    # Ticket médio
    average_ticket = total_revenue / total_orders if total_orders > 0 else 0
    
    # Tempo médio de entrega
    df['delivery_time'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.days
    avg_delivery_time = df['delivery_time'].mean()
    
    # Taxa de cancelamento
    cancellation_rate = df["pedido_cancelado"].mean()
    
    # Receita perdida
    lost_revenue = df[df["pedido_cancelado"] == 1]["price"].sum()
    
    return {
        "total_revenue": total_revenue,
        "total_orders": total_orders,
        "total_customers": total_customers,
        "total_products": total_products,
        "unique_categories": unique_categories,
        "abandonment_rate": abandonment_rate,
        "csat": csat,
        "average_ticket": average_ticket,
        "avg_delivery_time": avg_delivery_time,
        "cancellation_rate": cancellation_rate,
        "lost_revenue": lost_revenue
    }

def calculate_churn_features(df, cutoff_date):
    """Calcula as features derivadas para análise de churn."""
    # Converter colunas de data para datetime
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
    df['order_delivered_customer_date'] = pd.to_datetime(df['order_delivered_customer_date'])
    
    # Filtrar dados antes da data de corte
    df_before_cutoff = df[df['order_purchase_timestamp'] <= cutoff_date]
    
    # Calcular total gasto por cliente
    total_spent = df_before_cutoff.groupby('customer_unique_id')['payment_value'].sum()
    
    # Calcular número de pedidos únicos por cliente
    num_orders = df_before_cutoff.groupby('customer_unique_id')['order_id'].nunique()
    
    # Calcular ticket médio por cliente
    avg_order_value = total_spent / num_orders
    
    # Calcular variação dos tickets por cliente
    std_order_value = df_before_cutoff.groupby('customer_unique_id')['payment_value'].std()
    
    # Calcular média de parcelas por cliente
    avg_installments = df_before_cutoff.groupby('customer_unique_id')['payment_installments'].mean()
    
    # Calcular média das avaliações por cliente
    avg_review = df_before_cutoff.groupby('customer_unique_id')['review_score'].mean()
    
    # Calcular taxa de cancelamento por cliente
    cancel_rate = df_before_cutoff[df_before_cutoff['order_status'] == 'canceled'].groupby('customer_unique_id')['order_id'].count() / num_orders
    
    # Calcular recência (dias desde a última compra) por cliente
    last_purchase_date = df_before_cutoff.groupby('customer_unique_id')['order_purchase_timestamp'].max()
    recency = (cutoff_date - last_purchase_date).dt.days
    
    # Criar DataFrame com as features derivadas
    churn_features = pd.DataFrame({
        'total_spent': total_spent,
        'num_orders': num_orders,
        'avg_order_value': avg_order_value,
        'std_order_value': std_order_value,
        'avg_installments': avg_installments,
        'avg_review': avg_review,
        'cancel_rate': cancel_rate,
        'recency': recency
    }).reset_index()
    
    return churn_features

def define_churn(df, cutoff_date):
    """Define a variável de churn com base na data de corte."""
    # Converter colunas de data para datetime
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
    
    # Identificar clientes ativos antes da data de corte
    active_customers = df[df['order_purchase_timestamp'] <= cutoff_date]['customer_unique_id'].unique()
    
    # Identificar clientes que compraram após a data de corte
    customers_after_cutoff = df[df['order_purchase_timestamp'] > cutoff_date]['customer_unique_id'].unique()
    
    # Definir churn: 1 se não comprou após a data de corte, 0 caso contrário
    churn_status = {customer: 0 if customer in customers_after_cutoff else 1 for customer in active_customers}
    
    # Criar DataFrame com o status de churn
    churn_df = pd.DataFrame(list(churn_status.items()), columns=['customer_unique_id', 'churn'])
    
    return churn_df

def kpi_card(title, value, help_text=None):
    """Creates a KPI card with a glass effect."""
    text_color = "#FFFFFF"  # White text for readability
    border_color = "rgba(255, 255, 255, 0.3)"  # Semi-transparent white border
    bg_color = "rgba(255, 255, 255, 0.1)"  # Semi-transparent white background
    shadow_color = "rgba(0, 0, 0, 0.1)"  # Light shadow

    st.markdown(
        f"""
        <div style="
            backdrop-filter: blur(10px);
            background: {bg_color};
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            font-size: 22px;
            font-family: 'Inter', sans-serif;
            font-weight: bold;
            color: {text_color};
            border: 1px solid {border_color};
            box-shadow: 0 4px 30px {shadow_color};">
            {title}  
            <br>  
            <span style="font-size: 28px; font-weight: bold;">{value}</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    if help_text:
        st.markdown(
        f"""
        <p style="font-size: 18px; color: gray; text-align: center; font-style: italic;">
            ℹ️ {help_text}
        </p>
        """,
        unsafe_allow_html=True
    )

def render_kpi_block(title, kpi_dict, cols_per_row=3):
    """
    Renders multiple KPIs in a single glass-effect container.
    
    Args:
        title (str): The title of the KPI block
        kpi_dict (dict): Dictionary of KPI names and their values
        cols_per_row (int): Number of columns per row (default: 3)
    """
    # Determine text color based on theme
    is_dark_theme = st.get_option("theme.base") == "dark"
    text_color = "rgba(255,255,255,0.9)" if is_dark_theme else "rgba(0,0,0,0.9)"
    subtitle_color = "rgba(255,255,255,0.7)" if is_dark_theme else "rgba(0,0,0,0.7)"
    
    # Add responsive CSS separately
    st.markdown("""
        <style>
            @media (max-width: 768px) {
                div[data-testid="column"] {
                    width: 48% !important;
                    flex: 1 1 48% !important;
                    min-width: 48% !important;
                }
            }
        </style>
    """, unsafe_allow_html=True)
    
    # Render the container with glass effect
    st.markdown(f"""
    <div style="
        backdrop-filter: blur(10px);
        background: rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 30px;
        margin: 20px 0;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
    ">
        <h2 style="color: {text_color}; text-align: center; margin-bottom: 30px; font-family: 'Inter', sans-serif;">{title}</h2>
    """, unsafe_allow_html=True)

    # Calculate number of rows needed
    kpi_items = list(kpi_dict.items())
    rows = (len(kpi_items) + cols_per_row - 1) // cols_per_row

    # Create rows and columns
    for i in range(rows):
        cols = st.columns(cols_per_row)
        for j in range(cols_per_row):
            idx = i * cols_per_row + j
            if idx < len(kpi_items):
                kpi_title, kpi_value = kpi_items[idx]
                with cols[j]:
                    st.markdown(f"""
                        <div style="
                            backdrop-filter: blur(10px);
                            background: rgba(255, 255, 255, 0.1);
                            border: 1px solid rgba(255, 255, 255, 0.3);
                            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
                            text-align: center;
                            padding: 20px;
                            border-radius: 15px;
                            transition: all 0.3s ease;
                            height: 100%;
                            margin: 5px;
                        "
                        onmouseover="this.style.transform='scale(1.05)'; this.style.boxShadow='0 8px 25px rgba(0, 0, 0, 0.2)';"
                        onmouseout="this.style.transform='scale(1)'; this.style.boxShadow='0 4px 15px rgba(0, 0, 0, 0.1)';">
                            <p style="
                                font-size: 18px;
                                color: {subtitle_color};
                                margin-bottom: 10px;
                                font-family: 'Inter', sans-serif;
                            ">{kpi_title}</p>
                            <h3 style="
                                font-size: 26px;
                                color: {text_color};
                                margin: 0;
                                font-family: 'Inter', sans-serif;
                                font-weight: bold;
                            ">{kpi_value}</h3>
                        </div>
                    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

def render_plotly_glass_card(title, fig, height=620):
    """
    Renders a Plotly figure inside a glass-effect container.
    
    Args:
        title (str): The title of the chart
        fig: The Plotly figure object
        height (int): Height of the container in pixels (default: 620)
    """
    # Get theme to adjust text color
    is_dark_theme = st.get_option("theme.base") == "dark"
    text_color = "rgba(255,255,255,0.9)" if is_dark_theme else "rgba(0,0,0,0.9)"
    
    # Update layout for better spacing and label positioning
    fig.update_layout(
        margin=dict(l=80, r=20, t=40, b=80),  # Increased bottom margin
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=14)
    )
    
    # Update Y-axis properties
    fig.update_yaxes(
        automargin=True,
        ticks="outside",
        ticklabelposition="outside left"
    )
    
    # Update X-axis properties
    fig.update_xaxes(
        ticklabelposition="outside bottom",
        ticks="outside"
    )
    
    # Convert Plotly figure to HTML
    html_graph = fig.to_html(full_html=False, include_plotlyjs='cdn')
    
    # Create glass container with the graph
    html_content = f"""
    <div style="
        backdrop-filter: blur(10px);
        background: rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 30px 40px;
        margin: 20px 0;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        color: {text_color};
        ">
        <h3 style="
            text-align: center;
            margin-bottom: 20px;
            font-family: 'Inter', sans-serif;
            font-size: 24px;
            ">{title}</h3>
        {html_graph}
    </div>
    """
    
    # Render the HTML content with increased height
    components.html(html_content, height=height, scrolling=False)

if __name__ == "__main__":
    df = load_data()
    kpis = calculate_kpis(df)
    print("KPIs Calculados:", kpis)

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
    is_dark_theme = st.get_option("theme.base") == "dark"
    text_color = "rgba(255,255,255,0.9)" if is_dark_theme else "rgba(0,0,0,0.9)"
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

def render_kpi_block(kpi_values=None, cols_per_row=3):
    """
    Renderiza um bloco de KPIs com efeito glass.
    
    Args:
        kpi_values (dict): Dicionário com os valores dos KPIs
        cols_per_row (int): Número de colunas por linha (padrão: 3)
    """
    if kpi_values:
        # Calcular número de linhas necessárias
        num_kpis = len(kpi_values)
        num_rows = (num_kpis + cols_per_row - 1) // cols_per_row
        
        # Criar layout de colunas
        cols = st.columns(cols_per_row)
        
        # Preencher as colunas com os KPIs
        for i, (kpi_name, kpi_value) in enumerate(kpi_values.items()):
            row = i // cols_per_row
            col = i % cols_per_row
            
            with cols[col]:
                kpi_card(kpi_name, kpi_value)

def render_kpi_block_title(title, cols_per_row=3):
    """
    Renderiza um título estilizado para um bloco de KPIs.
    
    Args:
        title (str): Título do bloco de KPIs
        cols_per_row (int): Número de colunas por linha (padrão: 3)
    """
    is_dark_theme = st.get_option("theme.base") == "dark"
    text_color = "rgba(255,255,255,0.9)" if is_dark_theme else "rgba(0,0,0,0.9)"
    
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0.05));
        backdrop-filter: blur(10px);
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        
    ">
        <h2 style="
            margin: 0;
            font-size: 1.5em;
            text-align: center;
            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        ">{title}</h2>
    </div>
    """, unsafe_allow_html=True)

def render_plotly_glass_card(title, fig, height=620):
    """
    Renders a Plotly figure with a glass effect directly in the figure configuration.
    
    Args:
        title (str): The title of the chart
        fig: The Plotly figure object
        height (int): Height of the container in pixels (default: 620)
    """
    # Get theme to adjust colors
    is_dark_theme = st.get_option("theme.base") == "dark"
    text_color = "rgba(255,255,255,0.9)" if is_dark_theme else "rgba(0,0,0,0.9)"
    grid_color = "rgba(255, 255, 255, 0.30)"  # Grid branco com 30% de opacidade
    border_color = "rgba(255, 255, 255, 0.3)"  # Semi-transparent white border
    bg_color = "rgba(255, 255, 255, 0.1)"  # Semi-transparent white background
    
    # Update layout for better spacing and label positioning
    fig.update_layout(
        margin=dict(l=80, r=20, t=40, b=80),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(
            family="Inter, sans-serif",
            size=14,
        ),
        title=dict(
            text=title,
            font=dict(size=22),
            x=0.5,
            xanchor='center'
        ),
        # Add glass effect
        shapes=[
            dict(
                type="rect",
                xref="paper",
                yref="paper",
                x0=0,
                y0=0,
                x1=1,
                y1=1,
                line=dict(
                    color=border_color,
                    width=1
                ),
                fillcolor=bg_color,
                layer="below"
            )
        ]
    )
    
    # Update Y-axis properties
    fig.update_yaxes(
        automargin=True,
        ticks="outside",
        ticklabelposition="outside left",
        zerolinewidth=1
    )
    
    # Update X-axis properties
    fig.update_xaxes(
        ticklabelposition="outside bottom",
        ticks="outside",
        zerolinewidth=1
    )
    
    # Add hover effect
    fig.update_traces(
        hoverlabel=dict(
            bgcolor=bg_color,
            bordercolor=border_color,
        )
    )
    
    # Render the Plotly figure
    st.plotly_chart(fig, use_container_width=True, height=height)

if __name__ == "__main__":
    df = load_data()
    kpis = calculate_kpis(df)
    print("KPIs Calculados:", kpis)

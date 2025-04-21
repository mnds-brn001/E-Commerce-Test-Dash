import pandas as pd
from datetime import datetime
from typing import Dict, Any, Tuple, List
import streamlit as st
from utils.KPIs import render_kpi_block, render_plotly_glass_card
import plotly.graph_objects as go

def calculate_revenue_insights(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calcula insights relacionados à receita.
    
    Args:
        df: DataFrame com os dados filtrados
        
    Returns:
        Dict com insights sobre receita, incluindo:
        - growth_rate: Taxa de crescimento
        - best_month: Mês com maior receita
        - best_month_revenue: Valor da receita do melhor mês
        - trend: Tendência (crescimento, estável, queda)
        - monthly_revenue: DataFrame com receita mensal
    """
    # Calcular receita mensal
    monthly_revenue = df.groupby(pd.to_datetime(df['order_purchase_timestamp']).dt.to_period('M'))['price'].sum().reset_index()
    monthly_revenue['order_purchase_timestamp'] = monthly_revenue['order_purchase_timestamp'].astype(str)
    
    # Calcular crescimento
    if len(monthly_revenue) >= 2:
        first_month = monthly_revenue.iloc[0]['price']
        last_month = monthly_revenue.iloc[-1]['price']
        growth_rate = (last_month - first_month) / first_month * 100 if first_month > 0 else 0
        
        # Determinar tendência
        if abs(growth_rate) < 5:
            trend = "estável"
            trend_icon = "➡️"
        elif growth_rate > 0:
            trend = "crescimento"
            trend_icon = "📈"
        else:
            trend = "queda"
            trend_icon = "📉"
    else:
        growth_rate = 0
        trend = "indeterminado"
        trend_icon = "❓"
    
    # Identificar melhor mês
    best_month_idx = monthly_revenue['price'].idxmax()
    best_month = monthly_revenue.iloc[best_month_idx]
    
    return {
        "growth_rate": growth_rate,
        "trend": trend,
        "trend_icon": trend_icon,
        "best_month": best_month['order_purchase_timestamp'],
        "best_month_revenue": best_month['price'],
        "monthly_revenue": monthly_revenue
    }

def calculate_satisfaction_insights(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calcula insights relacionados à satisfação do cliente.
    
    Args:
        df: DataFrame com os dados filtrados
        
    Returns:
        Dict com insights sobre satisfação, incluindo:
        - avg_satisfaction: Satisfação média
        - satisfaction_trend: Tendência da satisfação
        - distribution: Distribuição das avaliações
        - monthly_satisfaction: DataFrame com satisfação mensal
    """
    # Calcular satisfação mensal
    monthly_satisfaction = df.groupby(pd.to_datetime(df['order_purchase_timestamp']).dt.to_period('M'))['review_score'].mean().reset_index()
    monthly_satisfaction['order_purchase_timestamp'] = monthly_satisfaction['order_purchase_timestamp'].astype(str)
    
    # Calcular média geral
    avg_satisfaction = df['review_score'].mean()
    
    # Calcular distribuição
    satisfaction_distribution = df['review_score'].value_counts(normalize=True).sort_index()
    
    # Analisar tendência
    if len(monthly_satisfaction) >= 3:
        recent_avg = monthly_satisfaction['review_score'].tail(3).mean()
        old_avg = monthly_satisfaction['review_score'].head(3).mean()
        satisfaction_change = ((recent_avg - old_avg) / old_avg) * 100 if old_avg > 0 else 0
        
        if abs(satisfaction_change) < 5:
            satisfaction_trend = "estável"
            trend_icon = "➡️"
        elif satisfaction_change > 0:
            satisfaction_trend = "melhorando"
            trend_icon = "📈"
        else:
            satisfaction_trend = "piorando"
            trend_icon = "📉"
    else:
        satisfaction_change = 0
        satisfaction_trend = "indeterminado"
        trend_icon = "❓"
    
    return {
        "avg_satisfaction": avg_satisfaction,
        "satisfaction_trend": satisfaction_trend,
        "trend_icon": trend_icon,
        "satisfaction_change": satisfaction_change,
        "distribution": satisfaction_distribution,
        "monthly_satisfaction": monthly_satisfaction,
        "top_score_percentage": satisfaction_distribution.get(5, 0),
        "lowest_score_percentage": satisfaction_distribution.get(1, 0)
    }

def calculate_cancellation_insights(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calcula insights relacionados a cancelamentos.
    
    Args:
        df: DataFrame com os dados filtrados
        
    Returns:
        Dict com insights sobre cancelamentos, incluindo:
        - cancellation_rate: Taxa de cancelamento
        - total_cancelled: Total de pedidos cancelados
        - lost_revenue: Receita perdida
        - monthly_cancellation: DataFrame com cancelamentos mensais
    """
    # Calcular taxa de cancelamento mensal
    monthly_cancellation = df.groupby(pd.to_datetime(df['order_purchase_timestamp']).dt.to_period('M'))['pedido_cancelado'].mean().reset_index()
    monthly_cancellation['order_purchase_timestamp'] = monthly_cancellation['order_purchase_timestamp'].astype(str)
    
    # Calcular métricas gerais
    cancellation_rate = df['pedido_cancelado'].mean()
    total_cancelled = df[df['pedido_cancelado'] == 1]['order_id'].nunique()
    lost_revenue = df[df['pedido_cancelado'] == 1]['price'].sum()
    
    # Analisar tendência
    if len(monthly_cancellation) >= 3:
        recent_rate = monthly_cancellation['pedido_cancelado'].tail(3).mean()
        old_rate = monthly_cancellation['pedido_cancelado'].head(3).mean()
        rate_change = ((recent_rate - old_rate) / old_rate) * 100 if old_rate > 0 else 0
        
        if abs(rate_change) < 5:
            cancellation_trend = "estável"
            trend_icon = "➡️"
        elif rate_change > 0:
            cancellation_trend = "aumentando"
            trend_icon = "📈"
        else:
            cancellation_trend = "diminuindo"
            trend_icon = "📉"
    else:
        rate_change = 0
        cancellation_trend = "indeterminado"
        trend_icon = "❓"
    
    return {
        "cancellation_rate": cancellation_rate,
        "total_cancelled": total_cancelled,
        "lost_revenue": lost_revenue,
        "cancellation_trend": cancellation_trend,
        "trend_icon": trend_icon,
        "rate_change": rate_change,
        "monthly_cancellation": monthly_cancellation
    }

def calculate_delivery_insights(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calcula insights relacionados a entregas.
    
    Args:
        df: DataFrame com os dados filtrados
        
    Returns:
        Dict com insights sobre entregas, incluindo:
        - avg_delivery_time: Tempo médio de entrega
        - delivery_trend: Tendência do tempo de entrega
        - delivery_stats: Estatísticas de entrega
    """
    # Calcular tempo de entrega
    df['delivery_time'] = (pd.to_datetime(df['order_delivered_customer_date']) - 
                         pd.to_datetime(df['order_purchase_timestamp'])).dt.days
    
    # Calcular médias mensais
    monthly_delivery = df.groupby(pd.to_datetime(df['order_purchase_timestamp']).dt.to_period('M'))['delivery_time'].mean().reset_index()
    monthly_delivery['order_purchase_timestamp'] = monthly_delivery['order_purchase_timestamp'].astype(str)
    
    # Calcular métricas gerais
    avg_delivery_time = df['delivery_time'].mean()
    
    # Definir limites para classificação de entregas
    FAST_DELIVERY = 7  # Entregas em até 7 dias são consideradas rápidas
    NORMAL_DELIVERY = 15  # Entregas em até 15 dias são consideradas normais
    
    # Calcular distribuição das entregas
    fast_deliveries = df[df['delivery_time'] <= FAST_DELIVERY]['order_id'].count()
    normal_deliveries = df[(df['delivery_time'] > FAST_DELIVERY) & (df['delivery_time'] <= NORMAL_DELIVERY)]['order_id'].count()
    slow_deliveries = df[df['delivery_time'] > NORMAL_DELIVERY]['order_id'].count()
    total_deliveries = df['order_id'].count()
    
    # Calcular percentuais
    fast_rate = fast_deliveries / total_deliveries if total_deliveries > 0 else 0
    normal_rate = normal_deliveries / total_deliveries if total_deliveries > 0 else 0
    slow_rate = slow_deliveries / total_deliveries if total_deliveries > 0 else 0
    
    # Analisar tendência
    if len(monthly_delivery) >= 3:
        recent_time = monthly_delivery['delivery_time'].tail(3).mean()
        old_time = monthly_delivery['delivery_time'].head(3).mean()
        time_change = ((recent_time - old_time) / old_time) * 100 if old_time > 0 else 0
        
        if abs(time_change) < 5:
            delivery_trend = "estável"
            trend_icon = "➡️"
        elif time_change > 0:
            delivery_trend = "aumentando"
            trend_icon = "📈"
        else:
            delivery_trend = "melhorando"
            trend_icon = "📉"
    else:
        time_change = 0
        delivery_trend = "indeterminado"
        trend_icon = "❓"
    
    return {
        "avg_delivery_time": avg_delivery_time,
        "delivery_trend": delivery_trend,
        "trend_icon": trend_icon,
        "time_change": time_change,
        "monthly_delivery": monthly_delivery,
        "delivery_stats": {
            "fast_rate": fast_rate,
            "normal_rate": normal_rate,
            "slow_rate": slow_rate,
            "fast_count": fast_deliveries,
            "normal_count": normal_deliveries,
            "slow_count": slow_deliveries,
            "total_deliveries": total_deliveries
        }
    }

def calculate_customer_behavior_insights(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calcula insights relacionados ao comportamento do cliente.
    
    Args:
        df: DataFrame com os dados filtrados
        
    Returns:
        Dict com insights sobre comportamento do cliente
    """
    # Análise de satisfação
    satisfaction_data = df.groupby(pd.to_datetime(df['order_purchase_timestamp']).dt.to_period('M'))['review_score'].agg([
        'mean',
        'count',
        'std'
    ]).reset_index()
    satisfaction_data['order_purchase_timestamp'] = satisfaction_data['order_purchase_timestamp'].astype(str)
    
    # Distribuição das avaliações
    review_distribution = df['review_score'].value_counts(normalize=True).sort_index()
    
    # Correlações
    price_satisfaction_corr = df.groupby('review_score')['price'].mean().corr(pd.Series([1,2,3,4,5]))
    repurchase_satisfaction_corr = df.groupby('review_score')['customer_unique_id'].nunique().corr(pd.Series([1,2,3,4,5]))
    
    # Análise de correlação
    correlations = {
        "ticket_vs_satisfaction": {
            "correlation": price_satisfaction_corr,
            "trend": "maior" if price_satisfaction_corr > 0 else "menor"
        },
        "repurchase_vs_satisfaction": {
            "correlation": repurchase_satisfaction_corr,
            "trend": "menor" if repurchase_satisfaction_corr > 0 else "maior"
        }
    }
    
    return {
        "satisfaction_evolution": satisfaction_data,
        "review_distribution": review_distribution,
        "correlations": correlations
    }

def generate_overview_insights(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Gera todos os insights para a página de Visão Geral.
    
    Args:
        df: DataFrame com os dados filtrados
        
    Returns:
        Dict com todos os insights organizados por categoria
    """
    revenue_insights = calculate_revenue_insights(df)
    satisfaction_insights = calculate_satisfaction_insights(df)
    cancellation_insights = calculate_cancellation_insights(df)
    delivery_insights = calculate_delivery_insights(df)
    
    # Identificar principais oportunidades de melhoria
    improvement_opportunities = []
    
    if cancellation_insights['cancellation_rate'] > 0.05:
        improvement_opportunities.append({
            "area": "Taxa de Cancelamento",
            "current": cancellation_insights['cancellation_rate'],
            "goal": "< 5%",
            "priority": "Alta" if cancellation_insights['cancellation_rate'] > 0.1 else "Média",
            "format_as_percentage": True
        })
    
    if delivery_insights['avg_delivery_time'] > 15:
        improvement_opportunities.append({
            "area": "Tempo de Entrega",
            "current": delivery_insights['avg_delivery_time'],
            "goal": "< 15 dias",
            "priority": "Alta" if delivery_insights['avg_delivery_time'] > 20 else "Média",
            "format_as_percentage": False
        })
    
    if satisfaction_insights['avg_satisfaction'] < 4.5:
        improvement_opportunities.append({
            "area": "Satisfação do Cliente",
            "current": satisfaction_insights['avg_satisfaction'],
            "goal": "> 4.5",
            "priority": "Alta" if satisfaction_insights['avg_satisfaction'] < 4.0 else "Média",
            "format_as_percentage": False
        })
    
    return {
        "revenue": revenue_insights,
        "satisfaction": satisfaction_insights,
        "cancellation": cancellation_insights,
        "delivery": delivery_insights,
        "improvement_opportunities": improvement_opportunities
    }

def format_insight_message(insights: Dict[str, Any]) -> str:
    """
    Formata a mensagem de insights para exibição.
    
    Args:
        insights: Dicionário com os insights calculados
        
    Returns:
        String formatada com os insights principais
    """
    revenue = insights['revenue']
    satisfaction = insights['satisfaction']
    cancellation = insights['cancellation']
    delivery = insights['delivery']
    
    message = f"""
    💰 **Desempenho Financeiro**
    - Receita está em {revenue['trend']} {revenue['trend_icon']} ({revenue['growth_rate']:.1f}%)
    - Melhor mês: {revenue['best_month']} (R$ {revenue['best_month_revenue']:,.2f})
    
    😊 **Satisfação do Cliente**
    - Média de {satisfaction['avg_satisfaction']:.2f}/5.0
    - Tendência {satisfaction['satisfaction_trend']} {satisfaction['trend_icon']}
    - {(satisfaction['top_score_percentage']*100):.1f}% deram nota máxima
    
    ❌ **Cancelamentos**
    - Taxa de {(cancellation['cancellation_rate']*100):.1f}%
    - Tendência {cancellation['cancellation_trend']} {cancellation['trend_icon']}
    - Receita perdida: R$ {cancellation['lost_revenue']:,.2f}
    
    📦 **Entregas**
    - Tempo médio: {delivery['avg_delivery_time']:.1f} dias
    - Tendência {delivery['delivery_trend']} {delivery['trend_icon']}
    - {(delivery['on_time_rate']*100):.1f}% no prazo
    """
    
    return message 

def render_insight_card(title: str, value: str, trend: str, trend_icon: str, help_text: str = None) -> str:
    """
    Cria um card de insight com efeito glass similar ao kpi_card.
    
    Args:
        title: Título do insight
        value: Valor principal do insight
        trend: Descrição da tendência
        trend_icon: Ícone da tendência
        help_text: Texto de ajuda opcional
    
    Returns:
        HTML formatado do card
    """
    is_dark_theme = st.get_option("theme.base") == "dark"
    text_color = "rgba(255,255,255,0.9)" if is_dark_theme else "rgba(0,0,0,0.9)"
    border_color = "rgba(255, 255, 255, 0.3)"
    bg_color = "rgba(255, 255, 255, 0.1)"
    shadow_color = "rgba(0, 0, 0, 0.1)"
    
    trend_colors = {
        "📈": "#2ecc71",  # Verde para crescimento
        "📉": "#e74c3c",  # Vermelho para queda
        "➡️": "#3498db",  # Azul para estável
        "❓": "#95a5a6"   # Cinza para indeterminado
    }
    
    trend_color = trend_colors.get(trend_icon, "#95a5a6")
    
    html = f"""
    <div style="
        position: relative;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        background: linear-gradient(135deg, {bg_color}, rgba(255, 255, 255, 0.05));
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-family: 'Inter', sans-serif;
        border: 1px solid {border_color};
        box-shadow: 0 4px 30px {shadow_color};
        margin-bottom: 15px;
        overflow: hidden;
        ">
        <div style="
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0.05));
            z-index: -1;
            "></div>
        <div style="font-size: 18px; margin-bottom: 10px; position: relative;">
            {title}
        </div>
        <div style="font-size: 24px; font-weight: bold; margin-bottom: 10px; position: relative;">
            {value}
        </div>
        <div style="
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 16px;
            color: {trend_color};
            position: relative;
            ">
            <span style="margin-right: 5px;">{trend_icon}</span>
            <span>{trend}</span>
        </div>
        {f'<div style="font-size: 14px; margin-top: 10px; opacity: 0.8; position: relative;">{help_text}</div>' if help_text else ''}
    </div>
    """
    return html

def render_revenue_insights(insights: Dict[str, Any]) -> None:
    """Renderiza insights de receita."""
    revenue = insights['revenue']

    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(
            render_insight_card(
                "Crescimento da Receita",
                f"{revenue['growth_rate']:.1f}%",
                revenue['trend'],
                revenue['trend_icon'],
                "Comparação com o período anterior"
            ),
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            render_insight_card(
                "Melhor Mês",
                f"R$ {revenue['best_month_revenue']:,.2f}",
                revenue['best_month'],
                "🏆",
                "Mês com maior faturamento"
            ),
            unsafe_allow_html=True
        )

def render_satisfaction_insights(insights: Dict[str, Any]) -> None:
    """Renderiza insights de satisfação."""
    satisfaction = insights['satisfaction']
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(
            render_insight_card(
                "Satisfação Média",
                f"{satisfaction['avg_satisfaction']:.1f}/5.0",
                satisfaction['satisfaction_trend'],
                satisfaction['trend_icon'],
                "Média geral das avaliações"
            ),
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            render_insight_card(
                "Avaliações 5 Estrelas",
                f"{(satisfaction['top_score_percentage']*100):.1f}%",
                "dos clientes",
                "⭐",
                "Porcentagem de notas máximas"
            ),
            unsafe_allow_html=True
        )
    
    with col3:
        st.markdown(
            render_insight_card(
                "Avaliações Baixas",
                f"{(satisfaction['lowest_score_percentage']*100):.1f}%",
                "dos clientes",
                "⚠️",
                "Porcentagem de notas mínimas"
            ),
            unsafe_allow_html=True
        )

def render_cancellation_insights(insights: Dict[str, Any]) -> None:
    """Renderiza insights de cancelamento."""
    cancellation = insights['cancellation']
    
    st.markdown("### ❌ Cancelamentos")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(
            render_insight_card(
                "Taxa de Cancelamento",
                f"{(cancellation['cancellation_rate']*100):.1f}%",
                cancellation['cancellation_trend'],
                cancellation['trend_icon'],
                "Percentual de pedidos cancelados"
            ),
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            render_insight_card(
                "Receita Perdida",
                f"R$ {cancellation['lost_revenue']:,.2f}",
                f"{cancellation['total_cancelled']} pedidos",
                "💸",
                "Valor total de cancelamentos"
            ),
            unsafe_allow_html=True
        )

def render_delivery_insights(insights: Dict[str, Any]) -> None:
    """Renderiza insights de entrega."""
    delivery = insights['delivery']
    stats = delivery['delivery_stats']
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(
            render_insight_card(
                "Tempo Médio de Entrega",
                f"{delivery['avg_delivery_time']:.1f} dias",
                delivery['delivery_trend'],
                delivery['trend_icon'],
                "Média do prazo de entrega"
            ),
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            render_insight_card(
                "Entregas Rápidas (até 7 dias)",
                f"{(stats['fast_rate']*100):.1f}%",
                f"{stats['fast_count']} pedidos",
                "🚀",
                "Entregas realizadas em até 7 dias"
            ),
            unsafe_allow_html=True
        )
    
    with col3:
        st.markdown(
            render_insight_card(
                "Entregas Atrasadas (>15 dias)",
                f"{(stats['slow_rate']*100):.1f}%",
                f"{stats['slow_count']} pedidos",
                "⚠️",
                "Entregas que levaram mais de 15 dias"
            ),
            unsafe_allow_html=True
        )

def render_improvement_opportunities(insights: Dict[str, Any]) -> None:
    """Renderiza oportunidades de melhoria."""
    opportunities = insights['improvement_opportunities']
    
    if opportunities:
        cols = st.columns(len(opportunities))
        
        for idx, opportunity in enumerate(opportunities):
            with cols[idx]:
                priority_colors = {
                    "Alta": "#e74c3c",
                    "Média": "#f1c40f"
                }
                priority_icons = {
                    "Alta": "🔥",
                    "Média": "⚠️"
                }
                
                # Formatar o valor atual baseado no tipo de métrica
                if opportunity['format_as_percentage']:
                    current_value = f"{(opportunity['current']*100):.1f}%"
                else:
                    current_value = f"{opportunity['current']:.1f}"
                
                st.markdown(
                    render_insight_card(
                        opportunity['area'],
                        current_value,
                        f"Meta: {opportunity['goal']}",
                        priority_icons[opportunity['priority']],
                        f"Prioridade {opportunity['priority']}"
                    ),
                    unsafe_allow_html=True
                )

def render_overview_insights(insights: Dict[str, Any]) -> None:
    """
    Renderiza todos os insights da visão geral de forma visual.
    
    Args:
        insights: Dicionário com todos os insights calculados
    """
    render_revenue_insights(insights)
    st.markdown("---")
    render_satisfaction_insights(insights)
    st.markdown("---")
    render_cancellation_insights(insights)
    st.markdown("---")
    render_delivery_insights(insights)
    st.markdown("---")
    render_improvement_opportunities(insights)

def render_customer_behavior_insights(df: pd.DataFrame) -> None:
    """
    Renderiza insights de comportamento do cliente.
    
    Args:
        df: DataFrame com os dados filtrados
    """
    insights = calculate_customer_behavior_insights(df)
    
    # Seção de Satisfação
    st.markdown("<h3 style='text-align: center;'>📊 Análise de Satisfação</h2>", unsafe_allow_html=True)
    
    # Métricas de Satisfação
    satisfaction_data = insights['satisfaction_evolution']
    latest_satisfaction = satisfaction_data.iloc[-1]
    
    satisfaction_kpis = {
        "😊 Nota Média Atual": f"{latest_satisfaction['mean']:.1f}/5.0",
        "📊 Desvio Padrão": f"{latest_satisfaction['std']:.2f}",
        "📝 Total Avaliações": f"{int(latest_satisfaction['count']):,}"
    }
    render_kpi_block(kpi_values=satisfaction_kpis, cols_per_row=3)
    st.markdown("---")
    # Distribuição das Avaliações

    distribution = insights['review_distribution']
    dist_cols = st.columns(5)
    
    for i, (score, percentage) in enumerate(distribution.items(), 1):
        with dist_cols[i-1]:
            st.markdown(
                render_insight_card(
                    f"{i} Estrela{'s' if i > 1 else ''}",
                    f"{percentage*100:.1f}%",
                    f"{int(percentage * len(df)):,} avaliações",
                    "⭐" * i,
                    "dos clientes"
                ),
                unsafe_allow_html=True
            )
    st.markdown("---")
    # Correlações e Insights
    st.markdown("<h3 style='text-align: center;'>🔍 Correlações e Insights</h2>", unsafe_allow_html=True)
        
    correlations = insights['correlations']
    corr_cols = st.columns(2)
    
    with corr_cols[0]:
        st.markdown(
            render_insight_card(
                "Ticket Médio vs Satisfação",
                "Correlação Positiva" if correlations['ticket_vs_satisfaction']['correlation'] > 0 else "Correlação Negativa",
                f"Clientes mais satisfeitos tendem a ter ticket {correlations['ticket_vs_satisfaction']['trend']}",
                "💰",
                "Relação entre gasto e satisfação"
            ),
            unsafe_allow_html=True
        )
    
    with corr_cols[1]:
        st.markdown(
            render_insight_card(
                "Recompra vs Satisfação",
                "Correlação Positiva" if correlations['repurchase_vs_satisfaction']['correlation'] > 0 else "Correlação Negativa",
                f"Clientes com notas mais baixas têm taxa de recompra {correlations['repurchase_vs_satisfaction']['trend']}",
                "🔄",
                "Relação entre satisfação e retenção"
            ),
            unsafe_allow_html=True
        )
    
    # Conclusões
    st.markdown("### 💡 Principais Conclusões")
    
    conclusions = f"""
    1. **Distribuição das Avaliações**
       - {(distribution[5]*100):.1f}% dos clientes deram nota máxima (5 estrelas)
       - {(distribution[1]*100):.1f}% dos clientes deram nota mínima (1 estrela)
       - Média atual de {latest_satisfaction['mean']:.1f}/5.0
    
    2. **Correlações Identificadas**
       - Ticket Médio: {correlations['ticket_vs_satisfaction']['trend']} para clientes mais satisfeitos
       - Taxa de Recompra: {correlations['repurchase_vs_satisfaction']['trend']} para clientes com notas baixas
    
    3. **Tendências**
       - Desvio padrão de {latest_satisfaction['std']:.2f} indica {'alta' if latest_satisfaction['std'] > 1 else 'baixa'} variabilidade nas avaliações
       - {latest_satisfaction['count']} avaliações no último período
    """
    
    st.markdown(conclusions)

def analyze_category_performance(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analisa o desempenho das categorias com base em múltiplas métricas.
    
    Args:
        df: DataFrame com os dados filtrados
        
    Returns:
        Dict com insights sobre categorias, incluindo:
        - top_categories: Categorias com melhor desempenho
        - bottom_categories: Categorias que precisam de atenção
        - category_metrics: Métricas detalhadas por categoria
    """
    # Calcular tempo de entrega
    df['delivery_time'] = (
        pd.to_datetime(df['order_delivered_customer_date']) - 
        pd.to_datetime(df['order_purchase_timestamp'])
    ).dt.days
    
    # Agrupar dados por categoria
    category_metrics = df.groupby('product_category_name').agg({
        'price': ['sum', 'mean', 'count'],  # Receita total, ticket médio, número de pedidos
        'review_score': ['mean', 'count'],  # Satisfação média, número de avaliações
        'order_id': 'nunique',  # Número de pedidos únicos
        'customer_unique_id': 'nunique',  # Número de clientes únicos
        'delivery_time': 'mean',  # Tempo médio de entrega
        'pedido_cancelado': 'mean',  # Taxa de cancelamento
        'payment_value': ['sum', 'mean']  # Valor total pago, valor médio pago
    }).reset_index()
    
    # Renomear colunas
    category_metrics.columns = [
        'category',
        'total_revenue',
        'avg_ticket',
        'total_items',
        'avg_satisfaction',
        'total_reviews',
        'unique_orders',
        'unique_customers',
        'avg_delivery_time',
        'cancellation_rate',
        'total_payment',
        'avg_payment'
    ]
    
    # Calcular métricas adicionais
    category_metrics['revenue_per_customer'] = category_metrics['total_revenue'] / category_metrics['unique_customers']
    category_metrics['items_per_order'] = category_metrics['total_items'] / category_metrics['unique_orders']
    category_metrics['review_rate'] = category_metrics['total_reviews'] / category_metrics['unique_orders']
    
    # Normalizar métricas para ranking
    metrics_to_normalize = [
        'total_revenue',
        'avg_ticket',
        'avg_satisfaction',
        'unique_orders',
        'unique_customers',
        'revenue_per_customer',
        'items_per_order',
        'review_rate',
        'avg_payment'
    ]
    
    # Criar colunas de score normalizadas
    for metric in metrics_to_normalize:
        min_val = category_metrics[metric].min()
        max_val = category_metrics[metric].max()
        range_val = max_val - min_val
        if range_val > 0:  # Evitar divisão por zero
            category_metrics[f'{metric}_score'] = (category_metrics[metric] - min_val) / range_val
        else:
            category_metrics[f'{metric}_score'] = 0.5  # Valor neutro quando não há variação
    
    # Calcular score composto
    weights = {
        'total_revenue': 0.20,
        'avg_ticket': 0.15,
        'avg_satisfaction': 0.20,
        'unique_orders': 0.10,
        'unique_customers': 0.10,
        'revenue_per_customer': 0.10,
        'items_per_order': 0.05,
        'review_rate': 0.05,
        'avg_payment': 0.05
    }
    
    # Calcular score composto usando as colunas de score normalizadas
    category_metrics['composite_score'] = sum(
        category_metrics[f'{metric}_score'] * weight
        for metric, weight in weights.items()
    )
    
    # Identificar categorias em destaque (top 5)
    top_categories = category_metrics.nlargest(5, 'composite_score')
    
    # Identificar categorias que precisam de atenção (bottom 5)
    bottom_categories = category_metrics.nsmallest(5, 'composite_score')
    
    return {
        'top_categories': top_categories,
        'bottom_categories': bottom_categories,
        'category_metrics': category_metrics
    }

def render_category_recommendations(analysis: Dict[str, Any]) -> None:
    """
    Renderiza as recomendações de categorias de forma visual.
    
    Args:
        analysis: Dicionário com a análise de categorias
    """
    # Criar duas colunas para exibição lado a lado
    col1, col2 = st.columns(2)
    
    # Categorias em Destaque
    with col1:
        st.markdown("### 🌟 Categorias em Destaque")
        
        top_kpis = {}
        for _, row in analysis['top_categories'].iterrows():
            category_name = row['category']
            top_kpis[category_name] = f"""
            💰 R$ {row['total_revenue']:,.2f}
            ⭐ {row['avg_satisfaction']:.1f}/5.0
            👥 {row['unique_customers']:,} clientes
            """
        
        render_kpi_block(kpi_values=top_kpis, cols_per_row=1)
    
    # Categorias que Precisam de Atenção
    with col2:
        st.markdown("### ⚠️ Categorias que Precisam de Atenção")
        
        bottom_kpis = {}
        for _, row in analysis['bottom_categories'].iterrows():
            category_name = row['category']
            bottom_kpis[category_name] = f"""
            💰 R$ {row['total_revenue']:,.2f}
            ⭐ {row['avg_satisfaction']:.1f}/5.0
            👥 {row['unique_customers']:,} clientes
            """
        
        render_kpi_block(kpi_values=bottom_kpis, cols_per_row=1)
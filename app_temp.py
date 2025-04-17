import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from utils.KPIs import load_data, calculate_kpis, calculate_acquisition_retention_kpis, filter_by_date_range, kpi_card, render_kpi_block, render_plotly_glass_card
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
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

# Funções auxiliares
def format_value(value, is_integer=False):
    """Formata um valor numérico com separador de milhares e duas casas decimais."""
    if is_integer:
        return f"{int(value):,}"
    return f"{value:,.2f}"

def format_percentage(value):
    """Formata um valor como porcentagem com duas casas decimais."""
    return f"{value*100:.2f}%"

def get_date_range(periodo):
    """Calcula o período selecionado."""
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

# Exibir a página selecionada
if pagina == "Visão Geral":
    st.title("Visão Geral")
    kpis = calculate_kpis(filtered_df, marketing_spend, date_range)
    
    # Preparar dicionário de KPIs
    kpi_values = {
        "💰 Receita Total": f"R$ {format_value(kpis['total_revenue'])}",
        "📦 Total de Pedidos": format_value(kpis['total_orders'], is_integer=True),
        "👥 Total de Clientes": format_value(kpis['total_customers'], is_integer=True),
        "🎯 Taxa de Abandono": format_percentage(kpis['abandonment_rate']),
        "😊 Satisfação do Cliente": format_value(kpis['csat']),
        "💰 Ticket Médio": f"R$ {format_value(kpis['average_ticket'])}",
        "📦 Tempo Médio de Entrega": f"{int(kpis['avg_delivery_time'])} dias",
        "❌ Taxa de Cancelamento": format_percentage(kpis['cancellation_rate']),
        "💸 Receita Perdida": f"R$ {format_value(kpis['lost_revenue'])}"
    }
    
    # Renderizar bloco de KPIs com efeito glass
    render_kpi_block("📊 Métricas de Performance", kpi_values, cols_per_row=3)

elif pagina == "Aquisição e Retenção":
    st.title("Aquisição e Retenção")
    kpis = calculate_kpis(filtered_df, marketing_spend, date_range)
    acquisition_kpis = calculate_acquisition_retention_kpis(filtered_df, marketing_spend, date_range)
    
    # Preparar dicionário de KPIs de Clientes
    customer_kpis = {
        "👥 Novos Clientes (Período)": format_value(acquisition_kpis['total_new_customers'], is_integer=True),
        "🔄 Taxa de Recompra": format_percentage(acquisition_kpis['repurchase_rate']),
        "⏳ Tempo até 2ª Compra": f"{int(acquisition_kpis['avg_time_to_second'])} dias"
    }
    
    # Renderizar bloco de KPIs de Clientes com efeito glass
    render_kpi_block("👥 Métricas de Clientes", customer_kpis, cols_per_row=3)
    
    # Preparar dicionário de KPIs Financeiros
    financial_kpis = {
        "💰 CAC": f"R$ {format_value(acquisition_kpis['cac'])}",
        "📈 LTV": f"R$ {format_value(acquisition_kpis['ltv'])}",
        "⚖️ LTV/CAC": format_value(acquisition_kpis['ltv'] / acquisition_kpis['cac'] if acquisition_kpis['cac'] > 0 else 0)
    }
    
    # Renderizar bloco de KPIs Financeiros com efeito glass
    render_kpi_block("💰 Métricas Financeiras", financial_kpis, cols_per_row=3)
    
    st.markdown("---")
    
    # 📈 Análise LTV/CAC
    st.header("📈 Análise LTV/CAC")
    
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
        status = "🚨 Crítico"
        status_color = "#dc3545"
    elif current_ratio == 1:
        status = "⚠️ Limite"
        status_color = "#ffc107"
    elif current_ratio < 3:
        status = "😬 Razoável"
        status_color = "#17a2b8"
    elif current_ratio == 3:
        status = "✅ Ideal"
        status_color = "#28a745"
    else:
        status = "💰 Alto"
        status_color = "#007bff"
    
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
                color: {text_color};
                text-align: center;
            ">
                <h3 style="margin-top: 0; color: {text_color};">Status Atual</h3>
                <p style="font-size: 24px; font-weight: bold; color: {status_color};">{status}</p>
                <p style="font-size: 18px;">Razão LTV/CAC: {format_value(current_ratio)}</p>
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
            elif current_ratio < 3:
                recommendations = [
                    ("🔍 Testar novos canais", "Explore canais com potencial de menor CAC"),
                    ("🔄 Melhorar retenção", "Implemente programas de fidelidade para aumentar o LTV"),
                    ("⚡ Otimizar funil", "Identifique e corrija gargalos no processo de aquisição")
                ]
                rec_color = "#f1c40f"  # Amarelo para situação de atenção
                rec_icon = "⚠️"
                rec_status = "Necessita Atenção"
            elif current_ratio > 5:
                recommendations = [
                    ("📈 Aumentar marketing", "Você pode estar subinvestindo em crescimento"),
                    ("🌍 Expandir mercados", "Aproveite a eficiência atual para escalar o negócio"),
                    ("🔄 Diversificar canais", "Explore novos canais para manter a eficiência")
                ]
                rec_color = "#3498db"  # Azul para oportunidade de crescimento
                rec_icon = "💰"
                rec_status = "Oportunidade de Crescimento"
            else:
                recommendations = [
                    ("⚖️ Manter equilíbrio", "Continue monitorando a razão LTV/CAC"),
                    ("📊 Testar aumentos", "Experimente aumentar o investimento em marketing"),
                    ("🔍 Otimizar processos", "Foque em melhorias incrementais")
                ]
                rec_color = "#2ecc71"  # Verde para situação saudável
                rec_icon = "✅"
                rec_status = "Situação Saudável"
            
            # Generate recommendations HTML as a separate string
            recs_html = ""
            for title, desc in recommendations:
                recs_html += (
                    f"<li style='margin-bottom: 15px;'>"
                    f"<strong style='color: {rec_color};'>{title}:</strong> "
                    f"<span style='color: {text_color};'>{desc}</span>"
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
                f"color: {text_color};"
                "'>"
                "<div style='"
                "display: flex;"
                "align-items: center;"
                "margin-bottom: 20px;"
                "padding-bottom: 15px;"
                f"border-bottom: 1px solid {rec_color};"
                "'>"
                f"<h3 style='margin: 0; color: {text_color};'>🎯 Ações Recomendadas</h3>"
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
            guide_table = """
            <div style="margin-top: 20px; padding-top: 20px; border-top: 1px solid rgba(255,255,255,0.2);">
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
                        <tr><td>= 1</td><td>Você empata</td><td style="color: #f39c12;">⚠️ Limite</td></tr>
                        <tr><td>1 &lt; x &lt; 3</td><td>Lucro baixo</td><td style="color: #f1c40f;">😬 Razoável</td></tr>
                        <tr><td>= 3</td><td>Ponto ideal (clássico)</td><td style="color: #2ecc71;">✅ Saudável</td></tr>
                        <tr><td>&gt; 3</td><td>Lucro alto</td><td style="color: #3498db;">💰 Excelente</td></tr>
                    </tbody>
                </table>
            </div>
            """
            
            # Passo 2: Montar o bloco de tendência como string segura
            trend_card = f"""
            <div style="
                backdrop-filter: blur(10px);
                background: rgba(255, 255, 255, 0.08);
                border-radius: 20px;
                padding: 25px;
                margin: 20px 0;
                border: 1px solid rgba(255,255,255,0.3);
                box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
                color: {text_color};
            ">
                <div style="display: flex; justify-content: space-between; margin-bottom: 20px;">
                    <div style="flex: 1; padding-right: 20px; border-right: 1px solid rgba(255,255,255,0.2);">
                        <h3 style="margin-top: 0;">📈 Análise de Tendência</h3>
                        <p style="font-size: 1.1em;">{trend_icon} A razão LTV/CAC está em <strong style='color:{trend_color};'>{trend_text}</strong></p>
                        <p style="font-size: 1.1em;">Variação de <strong>{delta_percent:+.1f}%</strong> {periodo_texto}</p>
                    </div>
                    <div style="flex: 1; padding-left: 20px;">
                        <h3 style="margin-top: 0;">📊 Detalhamento da Análise</h3>
                        <ul style="list-style-type: none; padding-left: 0; margin: 0;">
                            <li style="margin: 8px 0;">📅 Período analisado: <strong>{start_date.strftime('%b/%Y')} a {end_date.strftime('%b/%Y')}</strong></li>
                            <li style="margin: 8px 0;">📉 LTV/CAC médio período inicial: <strong>{format_value(older_ratio)}</strong></li>
                            <li style="margin: 8px 0;">📈 LTV/CAC médio período recente: <strong>{format_value(recent_ratio)}</strong></li>
                            <li style="margin: 8px 0;">📊 Meses considerados por período: <strong>{n_months}</strong></li>
                        </ul>
                    </div>
                </div>
                {guide_table}
            </div>
            """
            
            # Passo 3: Renderizar no Streamlit
            st.markdown(trend_card, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Período insuficiente para análise de tendência (mínimo 2 meses)")
    
    st.markdown("---")

elif pagina == "Análise de Churn":
    st.title("Análise de Churn")
    
    # Calcular métricas de churn
    churn_metrics = calculate_churn_metrics(filtered_df, date_range)
    
    # Preparar dicionário de KPIs de Churn
    churn_kpis = {
        "🔄 Taxa de Churn": format_percentage(churn_metrics['churn_rate']),
        "⏳ Tempo Médio até Churn": f"{int(churn_metrics['avg_time_to_churn'])} dias",
        "💰 Receita Perdida": f"R$ {format_value(churn_metrics['lost_revenue'])}"
    }
    
    # Renderizar bloco de KPIs de Churn com efeito glass
    render_kpi_block("📊 Métricas de Churn", churn_kpis, cols_per_row=3)
    
    st.markdown("---")
    
    # Análise de Tendência de Churn
    st.header("📈 Análise de Tendência de Churn")
    
    # Calcular churn por mês
    monthly_churn = filtered_df.groupby(filtered_df['order_purchase_timestamp'].dt.to_period('M')).agg({
        'customer_unique_id': 'nunique',
        'churned': 'sum'
    }).reset_index()
    
    monthly_churn['order_purchase_timestamp'] = monthly_churn['order_purchase_timestamp'].astype(str)
    monthly_churn['churn_rate'] = monthly_churn['churned'] / monthly_churn['customer_unique_id']
    
    # Gráfico de Evolução do Churn
    fig_churn = go.Figure()
    
    fig_churn.add_trace(go.Scatter(
        x=monthly_churn['order_purchase_timestamp'],
        y=monthly_churn['churn_rate'],
        name='Taxa de Churn',
        fill='tozeroy',
        line=dict(color='rgba(231, 76, 60, 0.7)'),
        fillcolor='rgba(231, 76, 60, 0.3)'
    ))
    
    fig_churn.update_layout(
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
    render_plotly_glass_card("📈 Evolução da Taxa de Churn ao Longo do Tempo", fig_churn)
    
    # Análise de tendência dinâmica
    if len(monthly_churn) >= 2:
        # Calcular período analisado
        start_date = pd.to_datetime(monthly_churn['order_purchase_timestamp'].iloc[0])
        end_date = pd.to_datetime(monthly_churn['order_purchase_timestamp'].iloc[-1])
        meses_filtrados = ((end_date.year - start_date.year) * 12 + end_date.month - start_date.month) + 1
        
        # Calcular médias para diferentes períodos
        n_months = min(3, len(monthly_churn))
        recent_churn = monthly_churn['churn_rate'].tail(n_months).mean()
        older_churn = monthly_churn['churn_rate'].head(n_months).mean()
        
        # Calcular variação percentual
        delta_percent = ((recent_churn - older_churn) / abs(older_churn)) * 100 if older_churn != 0 else 0
        
        # Determinar direção da tendência e ícone
        if abs(delta_percent) < 1:
            trend_icon = "➡️"
            trend_color = "#808080"
            trend_text = "estável"
        elif delta_percent > 0:
            trend_icon = "⬆️"
            trend_color = "#dc3545"
            trend_text = "aumento"
        else:
            trend_icon = "⬇️"
            trend_color = "#28a745"
            trend_text = "redução"
        
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
            current_churn = churn_metrics['churn_rate']
            
            # Determinar status e cor
            if current_churn < 0.05:
                status = "✅ Excelente"
                status_color = "#28a745"
            elif current_churn < 0.10:
                status = "😊 Bom"
                status_color = "#17a2b8"
            elif current_churn < 0.15:
                status = "⚠️ Atenção"
                status_color = "#ffc107"
            else:
                status = "🚨 Crítico"
                status_color = "#dc3545"
            
            st.markdown(f"""
            <div style="
                backdrop-filter: blur(10px);
                background: rgba(255, 255, 255, 0.08);
                border-radius: 20px;
                padding: 25px;
                margin: 20px 0;
                border: 1px solid {status_color};
                box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
                color: {text_color};
                text-align: center;
            ">
                <h3 style="margin-top: 0; color: {text_color};">Status Atual</h3>
                <p style="font-size: 24px; font-weight: bold; color: {status_color};">{status}</p>
                <p style="font-size: 18px;">Taxa de Churn: {format_percentage(current_churn)}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Ações Recomendadas com efeito glass
            if current_churn >= 0.15:
                recommendations = [
                    ("🚨 Ação Imediata", "Implemente um programa de retenção urgente"),
                    ("📞 Contato Direto", "Entre em contato com clientes em risco"),
                    ("💰 Incentivos", "Ofereça descontos ou benefícios especiais")
                ]
                rec_color = "#dc3545"  # Vermelho para situação crítica
                rec_icon = "🚨"
                rec_status = "Situação Crítica"
            elif current_churn >= 0.10:
                recommendations = [
                    ("📊 Análise Profunda", "Identifique os principais motivos de churn"),
                    ("🎯 Segmentação", "Foque em grupos com maior risco"),
                    ("📈 Monitoramento", "Acompanhe indicadores de satisfação")
                ]
                rec_color = "#ffc107"  # Amarelo para situação de atenção
                rec_icon = "⚠️"
                rec_status = "Necessita Atenção"
            elif current_churn >= 0.05:
                recommendations = [
                    ("🔄 Manter Estratégia", "Continue com as ações atuais"),
                    ("📈 Otimização", "Busque melhorias incrementais"),
                    ("👥 Feedback", "Mantenha canal aberto com clientes")
                ]
                rec_color = "#17a2b8"  # Azul para situação boa
                rec_icon = "✅"
                rec_status = "Situação Boa"
            else:
                recommendations = [
                    ("🌟 Excelência", "Documente as práticas de sucesso"),
                    ("📊 Benchmark", "Compartilhe métricas com a equipe"),
                    ("🎯 Inovação", "Teste novas estratégias de retenção")
                ]
                rec_color = "#28a745"  # Verde para situação excelente
                rec_icon = "💫"
                rec_status = "Situação Excelente"
            
            # Generate recommendations HTML
            recs_html = ""
            for title, desc in recommendations:
                recs_html += (
                    f"<li style='margin-bottom: 15px;'>"
                    f"<strong style='color: {rec_color};'>{title}:</strong> "
                    f"<span style='color: {text_color};'>{desc}</span>"
                    f"</li>"
                )
            
            # Build the recommendations block
            recommendations_block = (
                "<div style='"
                "backdrop-filter: blur(10px);"
                "background: rgba(255,255,255,0.08);"
                "border-radius: 20px;"
                "padding: 25px;"
                "margin: 30px 0;"
                f"border: 1px solid {rec_color};"
                f"box-shadow: 0 4px 20px rgba(0,0,0,0.1);"
                f"color: {text_color};"
                "'>"
                "<div style='"
                "display: flex;"
                "align-items: center;"
                "margin-bottom: 20px;"
                "padding-bottom: 15px;"
                f"border-bottom: 1px solid {rec_color};"
                "'>"
                f"<h3 style='margin: 0; color: {text_color};'>🎯 Ações Recomendadas</h3>"
                f"<div style='margin-left: auto; padding: 5px 12px; background: rgba({rec_color.replace('#', '')}, 0.1); border-radius: 15px; border: 1px solid {rec_color};'>"
                f"<span style='color: {rec_color};'>{rec_icon} {rec_status}</span>"
                "</div></div>"
                "<ul style='font-size: 1.1em; padding-left: 20px; line-height: 1.7; margin: 0;'>"
                f"{recs_html}"
                "</ul></div>"
            )
            
            st.markdown(recommendations_block, unsafe_allow_html=True)
        
        with col2:
            # Criar a tabela do guia
            guide_table = """
            <div style="margin-top: 20px; padding-top: 20px; border-top: 1px solid rgba(255,255,255,0.2);">
                <h3 style="margin-top: 0;">📋 Guia de Interpretação: Taxa de Churn</h3>
                <table style="width: 100%; border-collapse: collapse; font-size: 1.05em;">
                    <thead>
                        <tr style="border-bottom: 1px solid rgba(255,255,255,0.3);">
                            <th align="left">Faixa</th>
                            <th align="left">Interpretação</th>
                            <th align="left">Situação</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr><td>&lt; 5%</td><td>Retenção excepcional</td><td style="color: #28a745;">✅ Excelente</td></tr>
                        <tr><td>5-10%</td><td>Boa retenção</td><td style="color: #17a2b8;">😊 Bom</td></tr>
                        <tr><td>10-15%</td><td>Requer atenção</td><td style="color: #ffc107;">⚠️ Atenção</td></tr>
                        <tr><td>&gt; 15%</td><td>Problema sério</td><td style="color: #dc3545;">🚨 Crítico</td></tr>
                    </tbody>
                </table>
            </div>
            """
            
            # Montar o bloco de tendência
            trend_card = f"""
            <div style="
                backdrop-filter: blur(10px);
                background: rgba(255, 255, 255, 0.08);
                border-radius: 20px;
                padding: 25px;
                margin: 20px 0;
                border: 1px solid rgba(255,255,255,0.3);
                box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
                color: {text_color};
            ">
                <div style="display: flex; justify-content: space-between; margin-bottom: 20px;">
                    <div style="flex: 1; padding-right: 20px; border-right: 1px solid rgba(255,255,255,0.2);">
                        <h3 style="margin-top: 0;">📈 Análise de Tendência</h3>
                        <p style="font-size: 1.1em;">{trend_icon} A taxa de churn está em <strong style='color:{trend_color};'>{trend_text}</strong></p>
                        <p style="font-size: 1.1em;">Variação de <strong>{delta_percent:+.1f}%</strong> {periodo_texto}</p>
                    </div>
                    <div style="flex: 1; padding-left: 20px;">
                        <h3 style="margin-top: 0;">📊 Detalhamento da Análise</h3>
                        <ul style="list-style-type: none; padding-left: 0; margin: 0;">
                            <li style="margin: 8px 0;">📅 Período analisado: <strong>{start_date.strftime('%b/%Y')} a {end_date.strftime('%b/%Y')}</strong></li>
                            <li style="margin: 8px 0;">📉 Churn médio período inicial: <strong>{format_percentage(older_churn)}</strong></li>
                            <li style="margin: 8px 0;">📈 Churn médio período recente: <strong>{format_percentage(recent_churn)}</strong></li>
                            <li style="margin: 8px 0;">📊 Meses considerados por período: <strong>{n_months}</strong></li>
                        </ul>
                    </div>
                </div>
                {guide_table}
            </div>
            """
            
            st.markdown(trend_card, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Período insuficiente para análise de tendência (mínimo 2 meses)")
    
    st.markdown("---")
    
    # Análise de Cohorts
    st.header("📊 Análise de Cohorts")
    
    # Calcular dados de cohort
    cohort_data = calculate_cohort_data(filtered_df)
    
    if not cohort_data.empty:
        # Criar heatmap de cohorts
        fig_cohort = go.Figure(data=go.Heatmap(
            z=cohort_data.values,
            x=cohort_data.columns,
            y=cohort_data.index,
            colorscale='RdYlGn',
            reversescale=True
        ))
        
        fig_cohort.update_layout(
            title='Taxa de Retenção por Cohort (%)',
            xaxis_title='Mês',
            yaxis_title='Cohort'
        )
        
        # Renderizar heatmap com efeito glass
        render_plotly_glass_card("🔥 Heatmap de Retenção por Cohort", fig_cohort)
        
        # Análise de cohorts
        st.subheader("📈 Insights de Cohorts")
        
        # Calcular métricas de cohort
        retention_m1 = cohort_data[1].mean()
        retention_m3 = cohort_data[3].mean() if 3 in cohort_data.columns else None
        retention_m6 = cohort_data[6].mean() if 6 in cohort_data.columns else None
        
        # Criar colunas para métricas
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Retenção M1",
                f"{retention_m1:.1f}%",
                delta=f"{retention_m1 - 100:.1f}%" if retention_m1 else None,
                delta_color="inverse"
            )
        
        with col2:
            if retention_m3 is not None:
                st.metric(
                    "Retenção M3",
                    f"{retention_m3:.1f}%",
                    delta=f"{retention_m3 - 100:.1f}%",
                    delta_color="inverse"
                )
            else:
                st.info("Dados insuficientes para M3")
        
        with col3:
            if retention_m6 is not None:
                st.metric(
                    "Retenção M6",
                    f"{retention_m6:.1f}%",
                    delta=f"{retention_m6 - 100:.1f}%",
                    delta_color="inverse"
                )
            else:
                st.info("Dados insuficientes para M6")
        
        # Análise de insights
        best_cohort = cohort_data.mean(axis=1).idxmax()
        worst_cohort = cohort_data.mean(axis=1).idxmin()
        
        st.markdown(f"""
        <div style="
            backdrop-filter: blur(10px);
            background: rgba(255, 255, 255, 0.08);
            border-radius: 20px;
            padding: 25px;
            margin: 20px 0;
            border: 1px solid rgba(255,255,255,0.3);
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
            color: {text_color};
        ">
            <h3 style="margin-top: 0; color: {text_color};">🔍 Principais Insights</h3>
            <ul style="list-style-type: none; padding-left: 0;">
                <li style="margin: 10px 0;">✨ <strong>Melhor Cohort:</strong> {best_cohort}</li>
                <li style="margin: 10px 0;">⚠️ <strong>Pior Cohort:</strong> {worst_cohort}</li>
                <li style="margin: 10px 0;">📊 <strong>Retenção Média M1:</strong> {retention_m1:.1f}%</li>
                {f'<li style="margin: 10px 0;">📈 <strong>Retenção Média M3:</strong> {retention_m3:.1f}%</li>' if retention_m3 is not None else ''}
                {f'<li style="margin: 10px 0;">📉 <strong>Retenção Média M6:</strong> {retention_m6:.1f}%</li>' if retention_m6 is not None else ''}
            </ul>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Dados insuficientes para análise de cohorts")

# ... (restante do código) 
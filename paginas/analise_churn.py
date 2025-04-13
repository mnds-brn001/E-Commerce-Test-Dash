import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from datetime import datetime
from utils.KPIs import load_data, calculate_churn_features, define_churn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import joblib
import requests
from io import BytesIO

def read_results_file(file_path):
    """
    Tenta ler o arquivo de resultados com diferentes codificações.
    Retorna o conteúdo do arquivo ou None se não conseguir ler.
    """
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    return None

@st.cache_resource
def load_model():
    """
    Carrega o modelo de churn. Se estiver em produção (Streamlit Cloud),
    baixa do storage. Se estiver local, carrega do arquivo.
    """
    if 'STREAMLIT_SHARING' in os.environ:
        # URL do seu modelo (pode ser S3, Google Drive, etc)
        model_url = st.secrets["MODEL_URL"]
        response = requests.get(model_url)
        model = joblib.load(BytesIO(response.content))
    else:
        # Carrega localmente
        model = joblib.load('churn_model.pkl')
    return model

def app():
    # Configuração da página
    #st.set_page_config(layout="wide")
    
    # Título principal com ícone
    st.title("🔄 Análise de Churn")
    
    # Adicionar abas para diferentes funcionalidades
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Visão Geral", 
        "⚙️ Configurar Análise", 
        "📈 Resultados do Modelo", 
        "🔮 Previsão"
    ])
    
    # Carregar dados
    df = load_data()
    
    # TAB 1: VISÃO GERAL
    with tab1:
        # Cabeçalho com descrição
        st.header("📊 Visão Geral do Churn")
        
        # Layout em duas colunas para o conteúdo principal
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                <h3 style="margin-top: 0;">O que é Churn?</h3>
                <p>O churn (cancelamento) é um indicador crítico para negócios, pois indica a taxa com que os clientes 
                param de comprar ou usar os serviços. Monitorar e prever o churn permite que empresas tomem medidas 
                proativas para reter clientes valiosos.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Exibir distribuição de compras ao longo do tempo
            st.subheader("📅 Distribuição de Compras ao Longo do Tempo")
            
            monthly_orders = df.groupby(pd.to_datetime(df['order_purchase_timestamp']).dt.to_period('M'))['order_id'].count().reset_index()
            monthly_orders['order_purchase_timestamp'] = monthly_orders['order_purchase_timestamp'].astype(str)
            
            fig = px.line(
                monthly_orders, 
                x='order_purchase_timestamp', 
                y='order_id',
                title="Número de Pedidos por Mês",
                labels={'order_purchase_timestamp': 'Mês', 'order_id': 'Número de Pedidos'}
            )
            fig.update_layout(
                xaxis=dict(tickangle=45),
                yaxis=dict(title="Número de Pedidos"),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Mostrar informações básicas dos dados
            st.subheader("📋 Informações dos Dados")
            
            max_date = pd.to_datetime(df['order_purchase_timestamp']).max()
            min_date = pd.to_datetime(df['order_purchase_timestamp']).min()
            total_customers = df['customer_unique_id'].nunique()
            total_orders = df['order_id'].nunique()
            
            st.markdown(f"""
            <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                <p><strong>Período dos Dados:</strong> {min_date.strftime('%d/%m/%Y')} a {max_date.strftime('%d/%m/%Y')}</p>
                <p><strong>Total de Clientes:</strong> {total_customers:,}</p>
                <p><strong>Total de Pedidos:</strong> {total_orders:,}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Verificar se o modelo já foi treinado
            model_exists = os.path.exists('churn_model.pkl')
            results_exist = os.path.exists('churn_analysis_results.txt')
            
            if results_exist:
                st.success("✅ Um modelo de churn já foi treinado. Veja os resultados na aba 'Resultados do Modelo'.")
                
                # Extrair informações básicas do arquivo de resultados
                results_text = read_results_file('churn_analysis_results.txt')
                
                if results_text is None:
                    st.error("❌ Não foi possível ler o arquivo de resultados. Por favor, treine o modelo novamente.")
                else:
                    # Procurar taxa de churn
                    churn_rate_line = [line for line in results_text.split('\n') if "Taxa de churn:" in line]
                    if churn_rate_line:
                        churn_rate = churn_rate_line[0].split(": ")[1]
                        st.info(f"📊 A taxa de churn atual é de {churn_rate}")
            else:
                st.warning("⚠️ Nenhum modelo de churn foi treinado ainda. Acesse a aba 'Configurar Análise' para criar um modelo.")
            
            # Exemplo ilustrativo de como funciona a definição de churn
            st.subheader("ℹ️ Como Funciona a Definição de Churn")
            
            st.markdown("""
            <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                <p>Para esta análise, consideramos:</p>
                <ul>
                    <li><strong>Data de corte padrão:</strong> 17 de abril de 2018 (6 meses antes do final dos dados)</li>
                    <li><strong>Clientes com churn (1):</strong> Aqueles que não compraram após a data de corte</li>
                    <li><strong>Clientes sem churn (0):</strong> Aqueles que compraram após a data de corte</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Exemplo ilustrativo de como funciona a definição de churn
        st.subheader("📝 Exemplo de Definição de Churn")
        
        example_data = pd.DataFrame({
            'Cliente': ['Cliente A', 'Cliente B', 'Cliente C', 'Cliente D'],
            'Últimas Compras': [
                'Jan/2018, Mar/2018, Mai/2018, Jul/2018', 
                'Jan/2018, Fev/2018, Mar/2018',
                'Dez/2017, Fev/2018, Set/2018',
                'Nov/2017, Jan/2018, Fev/2018, Abr/2018'
            ],
            'Churn': ['Não', 'Sim', 'Não', 'Sim'],
            'Explicação': [
                'Comprou em Julho (após data de corte)',
                'Última compra em Março (antes da data de corte)',
                'Comprou em Setembro (após data de corte)',
                'Última compra em Abril (mesma data de corte, ainda é churn)'
            ]
        })
        
        st.table(example_data)
        
        st.markdown("""
        <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-top: 20px;">
            <p><strong>Nota:</strong> A data de corte padrão é 17 de abril de 2018. Clientes que não fizeram compras
            após essa data são considerados como tendo abandonado (churn = 1).</p>
        </div>
        """, unsafe_allow_html=True)
        
    # TAB 2: CONFIGURAR ANÁLISE
    with tab2:
        st.header("⚙️ Configurar Análise de Churn")
        
        # Layout em três colunas para melhor organização
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            with st.form("churn_config_form"):
                st.subheader("📋 Parâmetros da Análise")
                
                # Definição da data de corte
                cutoff_date = st.date_input(
                    "Data de Corte",
                    value=datetime(2018, 4, 17),
                    min_value=datetime(2017, 1, 1),
                    max_value=max_date.to_pydatetime()
                )
                
                # Método de rebalanceamento
                col1_form, col2_form = st.columns(2)
                with col1_form:
                    rebalance_method = st.selectbox(
                        "Método de Rebalanceamento",
                        options=["smote", "undersample", "none"],
                        help="""
                        - SMOTE: Gera exemplos sintéticos da classe minoritária
                        - Undersample: Remove exemplos da classe majoritária
                        - None: Não realiza rebalanceamento
                        """
                    )
                
                with col2_form:
                    class_weight = st.selectbox(
                        "Peso das Classes",
                        options=["balanced", "none"],
                        help="""
                        - Balanced: Atribui pesos inversamente proporcionais à frequência da classe
                        - None: Sem pesos
                        """
                    )
                
                # Tipo de modelo
                model_type = st.selectbox(
                    "Tipo de Modelo",
                    options=["random_forest", "xgboost", "logistic_regression"],
                    help="""
                    - Random Forest: Conjunto de árvores de decisão
                    - XGBoost: Implementação de Gradient Boosting
                    - Logistic Regression: Regressão logística
                    """
                )
                
                # Validação cruzada
                col1_form, col2_form = st.columns(2)
                with col1_form:
                    use_cv = st.number_input(
                        "Número de Folds para Validação Cruzada",
                        min_value=0,
                        max_value=10,
                        value=5,
                        help="0 para não usar validação cruzada"
                    )
                
                with col2_form:
                    test_size = st.slider(
                        "Proporção de Dados para Teste",
                        min_value=0.1,
                        max_value=0.5,
                        value=0.3,
                        step=0.05,
                        help="Porcentagem dos dados que será usada para teste"
                    )
                
                # Opção de grid search
                grid_search = st.checkbox(
                    "Realizar Grid Search",
                    value=False,
                    help="Busca exaustiva pelos melhores hiperparâmetros (pode demorar)"
                )
                
                # Botão para executar a análise
                submit_button = st.form_submit_button("🚀 Executar Análise de Churn")
        
        with col2:
            st.subheader("ℹ️ Sobre os Parâmetros")
            
            # CSS personalizado para os parâmetros
            st.markdown("""
                <style>
                    .param-box {
                        background-color: #f0f2f6;
                        padding: 20px;
                        border-radius: 10px;
                        margin-bottom: 20px;
                    }
                    .param-title {
                        color: #1f77b4;
                        font-size: 1.1em;
                        font-weight: bold;
                        margin-bottom: 5px;
                    }
                    .param-desc {
                        color: #2c3e50;
                        margin-bottom: 15px;
                    }
                </style>
            """, unsafe_allow_html=True)

            # Parâmetros em containers separados
            params = [
                {
                    "icon": "📅",
                    "title": "Data de Corte",
                    "desc": "Define o ponto no tempo a partir do qual consideramos que um cliente abandonou o serviço."
                },
                {
                    "icon": "⚖️",
                    "title": "Rebalanceamento",
                    "desc": "Métodos para lidar com o desequilíbrio entre classes (churn vs. não-churn)."
                },
                {
                    "icon": "🤖",
                    "title": "Tipo de Modelo",
                    "desc": "Algoritmos de machine learning para prever o churn."
                },
                {
                    "icon": "🔄",
                    "title": "Validação Cruzada",
                    "desc": "Método para avaliar a performance do modelo em diferentes subconjuntos dos dados."
                }
            ]

            # Gerar todo o HTML em uma única string, sem quebras de linha extras
            params_html = [
                '<div class="param-box">',
                *[f'<div class="param-title">{p["icon"]} {p["title"]}</div><div class="param-desc">{p["desc"]}</div>' for p in params],
                '</div>'
            ]

            # Renderizar todo o HTML de uma vez, juntando as partes
            st.markdown(''.join(params_html), unsafe_allow_html=True)
            
        with col3:
            st.subheader("📊 Distribuição Atual")
            
            # Calcular distribuição atual de churn
            cutoff_date_obj = datetime.combine(cutoff_date, datetime.min.time())
            churn_df = define_churn(df, cutoff_date_obj)
            
            if 'churn' in churn_df.columns:
                churn_counts = churn_df['churn'].value_counts()
                total = churn_counts.sum()
                
                fig = go.Figure(data=[go.Pie(
                    labels=['Não Churn', 'Churn'],
                    values=[churn_counts.get(0, 0), churn_counts.get(1, 0)],
                    hole=.3,
                    marker_colors=['#3366CC', '#DC3912']
                )])
                
                fig.update_layout(
                    title_text=f"Baseado na data de corte<br>selecionada",
                    height=300,
                    margin=dict(t=30, b=0, l=0, r=0)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                churn_rate = churn_counts.get(1, 0) / total * 100 if total > 0 else 0
                st.info(f"Taxa de churn atual: {churn_rate:.2f}%")
        
        if submit_button:
            # Formatar data e parâmetros
            cutoff_date_str = cutoff_date.strftime("%Y-%m-%d")
            class_weight_param = "None" if class_weight == "none" else class_weight
            cv_param = 0 if use_cv == 0 else use_cv
            
            # Construir comando para executar o script
            grid_search_param = "--grid_search" if grid_search else ""
            
            command = f"""python churn_analysis.py \
                --cutoff_date {cutoff_date_str} \
                --rebalance {rebalance_method} \
                --model {model_type} \
                --class_weight {class_weight} \
                --cv {cv_param} \
                --test_size {test_size} \
                {grid_search_param}"""
            
            st.info(f"Executando análise com o comando: `{command}`")
            
            # Aqui você pode executar o comando via subprocess, mas como estamos no Streamlit,
            # vamos apenas mostrar uma mensagem de progresso para simular a execução
            
            with st.spinner("Executando análise de churn... (Este processo pode demorar alguns minutos)"):
                st.markdown("""
                <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                    <p><strong>Nota:</strong> Em um ambiente de produção, este comando seria executado em segundo plano.</p>
                    <p>Para fins de demonstração, acesse o terminal e execute o comando acima manualmente.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Aqui você pode implementar a chamada real para o script de análise
                # Por exemplo:
                # import subprocess
                # result = subprocess.run(command, shell=True, capture_output=True, text=True)
                # st.code(result.stdout)
                
                st.success("✅ Análise concluída! Navegue para a aba 'Resultados do Modelo' para ver os resultados.")
                st.balloons()


    # TAB 3: RESULTADOS DO MODELO
    with tab3:
        st.header("📈 Resultados do Modelo de Churn")
        
        # Verificar se existe um modelo treinado
        if not os.path.exists('churn_analysis_results.txt'):
            st.warning("⚠️ Nenhum modelo foi treinado ainda. Acesse a aba 'Configurar Análise' para criar um modelo.")
            return
        
            # Carregar resultados do arquivo
            with open('churn_analysis_results.txt', 'r') as f:
                results_text = f.read()
            
        # Criar tabs para diferentes aspectos dos resultados
        results_tab1, results_tab2, results_tab3, results_tab4 = st.tabs([
            "📊 Visão Geral",
            "🎯 Importância das Features",
            "📉 Métricas Detalhadas",
            "📑 Relatório Técnico"
        ])
        
        # Extrair métricas principais
        metrics = {}
        # Substituir a carga direta do modelo pelo uso da função
        model = load_model()
        for line in results_text.split('\n'):
            if ': ' in line:
                for metric in ['Accuracy', 'Precision (weighted)', 'Recall (weighted)', 
                             'F1 (weighted)', 'AUC-ROC']:
                    if line.startswith(metric):
                        metrics[metric] = float(line.split(': ')[1])
        
        # TAB 1: VISÃO GERAL
        with results_tab1:
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.subheader("Métricas Principais")
                
                # Função para determinar cor do delta
                def get_delta_color(value):
                    if value >= 0.9:
                        return "normal"  # Verde (positivo)
                    elif value >= 0.7:
                        return "off"     # Neutro
                    return "inverse"     # Vermelho (negativo)
                
                # Métricas em colunas
                m1, m2, m3 = st.columns(3)
                m1.metric(
                    "Acurácia",
                    f"{metrics.get('Accuracy', 0):.1%}",
                    delta="modelo atual",
                    delta_color=get_delta_color(metrics.get('Accuracy', 0))
                )
                
                m2.metric(
                    "Precisão",
                    f"{metrics.get('Precision (weighted)', 0):.1%}",
                    delta="ponderada",
                    delta_color=get_delta_color(metrics.get('Precision (weighted)', 0))
                )
                
                m3.metric(
                    "Recall",
                    f"{metrics.get('Recall (weighted)', 0):.1%}",
                    delta="ponderado",
                    delta_color=get_delta_color(metrics.get('Recall (weighted)', 0))
                )
                
                # Matriz de Confusão
                st.markdown("#### Matriz de Confusão")
                conf_matrix = None
                if "Matriz de confusão:" in results_text:
                    matrix_text = results_text.split("Matriz de confusão:")[1].split("\n\n")[0]
                    try:
                        matrix_lines = [line.strip() for line in matrix_text.split('\n') if '[' in line]
                        matrix_values = []
                        for line in matrix_lines:
                            values = [int(x) for x in line.strip('[]').split()]
                            matrix_values.append(values)
                        conf_matrix = np.array(matrix_values)
                    except:
                        st.error("Erro ao processar matriz de confusão")
                
                if conf_matrix is not None:
                    total = conf_matrix.sum()
                    percentages = conf_matrix / total * 100
                    
                    fig = go.Figure(data=go.Heatmap(
                        z=conf_matrix,
                        x=['Previsto Não-Churn', 'Previsto Churn'],
                        y=['Real Não-Churn', 'Real Churn'],
                        text=[[f'{val:,d}<br>({pct:.1f}%)' for val, pct in zip(row, pct_row)] 
                              for row, pct_row in zip(conf_matrix, percentages)],
                        texttemplate="%{text}",
                        textfont={"size": 14},
                        colorscale='RdYlBu_r',
                        showscale=False
                    ))
                    
                    fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Distribuição de Churn")
                
                # Extrair distribuição
                churn_dist = {}
                for line in results_text.split('\n'):
                    if "Não-churn (0):" in line:
                        churn_dist['Não Churn'] = int(line.split(': ')[1])
                    elif "Churn (1):" in line:
                        churn_dist['Churn'] = int(line.split(': ')[1])
                
                if churn_dist:
                    fig = go.Figure(data=[go.Pie(
                        labels=list(churn_dist.keys()),
                        values=list(churn_dist.values()),
                        hole=.4,
                        marker_colors=['#3366CC', '#DC3912'],
                        textinfo='label+percent',
                        textposition='inside'
                    )])
                    
                    fig.update_layout(
                        showlegend=False,
                        height=300,
                        margin=dict(l=0, r=0, t=30, b=0)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    total = sum(churn_dist.values())
                    churn_rate = churn_dist['Churn'] / total if total > 0 else 0
                    st.metric("Taxa de Churn", f"{churn_rate:.1%}", delta="do total de clientes")
        
        # TAB 2: IMPORTÂNCIA DAS FEATURES
        with results_tab2:
            st.subheader("🎯 Análise de Importância das Features")
            
            # Extrair importância das features
            if "Importância das features:" in results_text:
                importance_section = results_text.split("Importância das features:")[1].split("\n\n")[0]
                importance_lines = importance_section.strip().split("\n")
                
                importance_data = []
                for line in importance_lines:
                    if ":" in line:
                        feature_part, importance = line.split(":", 1)
                        if "(" in feature_part and ")" in feature_part:
                            original_name = feature_part.split("(")[0].strip()
                            display_name = feature_part.split("(")[1].split(")")[0].strip()
                        else:
                            original_name = feature_part.strip()
                            display_name = original_name
                        
                        importance = float(importance.strip())
                        importance_data.append({
                            "Feature": display_name,
                            "Original_Feature": original_name,
                            "Importância": importance
                        })
                
                if importance_data:
                    importance_df = pd.DataFrame(importance_data)
                    importance_df = importance_df.sort_values("Importância", ascending=True)
                    
                    # Gráfico de barras horizontais
                    fig = go.Figure()
                    
                    fig.add_trace(go.Bar(
                        x=importance_df["Importância"],
                        y=importance_df["Feature"],
                        orientation='h',
                        marker_color='#1f77b4',
                        text=[f"{x:.1%}" for x in importance_df["Importância"]],
                        textposition='auto',
                        hovertemplate="<b>%{y}</b><br>" +
                                    "Importância: %{x:.1%}<br>" +
                                    "<extra></extra>"
                    ))
                    
                    fig.update_layout(
                        title="Importância Relativa das Features",
                        xaxis_title="Importância",
                        yaxis_title="Feature",
                        height=400,
                        margin=dict(l=10, r=10, t=30, b=10),
                        xaxis=dict(tickformat=".1%"),
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Interpretação das features
                    col1, col2 = st.columns([3, 2])
                    
                    with col1:
                        st.markdown("#### 💡 Interpretação das Features Mais Importantes")
                        
                        feature_explanations = {
                            "última compra": "Quanto tempo o cliente está sem comprar é um forte indicador de churn.",
                            "número de compras": "A frequência de compras indica o nível de engajamento do cliente.",
                            "valor total": "O valor total gasto mostra o valor do cliente para o negócio.",
                            "valor médio": "O ticket médio indica o padrão de consumo do cliente.",
                            "variação": "A variação nos valores mostra a consistência do comportamento.",
                            "parcelas": "O padrão de parcelamento indica o perfil financeiro.",
                            "cancelamento": "Histórico de cancelamentos é um indicador de insatisfação.",
                            "avaliação": "A satisfação do cliente medida através das avaliações."
                        }
                        
                        for idx, row in importance_df.iloc[-3:].iloc[::-1].iterrows():
                            feature = row["Feature"]
                            importance = row["Importância"]
                            original = row["Original_Feature"]
                            
                            explanation = next(
                                (exp for key, exp in feature_explanations.items() if key in feature.lower()),
                                "Esta feature contribui significativamente para a previsão de churn."
                            )
                            
                            st.markdown(f"""
                                <div style='
                                    background-color: #f0f2f6;
                                    padding: 15px;
                                    border-radius: 5px;
                                    margin-bottom: 10px;
                                '>
                                    <strong>{feature}</strong> ({importance:.1%})
                                    <br><em>Nome técnico: {original}</em>
                                    <br>{explanation}
                                </div>
                            """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("#### 📊 Correlações com Churn")
                        
                        # Extrair correlações
                        correlations = {}
                        if "Top correlações com churn:" in results_text:
                            corr_section = results_text.split("Top correlações com churn:")[1].split("\n\n")[0]
                            for line in corr_section.strip().split("\n"):
                                if ":" in line and "churn:" not in line:
                                    feature, corr = line.split(":")
                                    correlations[feature.strip()] = float(corr)
                        
                        if correlations:
                            corr_df = pd.DataFrame(
                                list(correlations.items()),
                                columns=['Feature', 'Correlação']
                            ).sort_values('Correlação', key=abs, ascending=True)
                            
                            fig = go.Figure()
                            
                            fig.add_trace(go.Bar(
                                x=corr_df['Correlação'],
                                y=corr_df['Feature'],
                                orientation='h',
                                marker_color=['#DC3912' if x < 0 else '#3366CC' for x in corr_df['Correlação']],
                                text=[f"{x:+.2f}" for x in corr_df['Correlação']],
                                textposition='auto'
                            ))
                        
                        fig.update_layout(
                                title="Correlação com Churn",
                                height=300,
                                margin=dict(l=10, r=10, t=30, b=10),
                                showlegend=False
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
        
        # TAB 3: MÉTRICAS DETALHADAS
        with results_tab3:
            st.subheader("📉 Métricas de Performance Detalhadas")
            
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.markdown("#### Performance do Modelo")
                
                # Criar tabela de métricas
                metrics_df = pd.DataFrame([
                    ["Acurácia", metrics.get('Accuracy', 0)],
                    ["Precisão (weighted)", metrics.get('Precision (weighted)', 0)],
                    ["Recall (weighted)", metrics.get('Recall (weighted)', 0)],
                    ["F1 Score (weighted)", metrics.get('F1 (weighted)', 0)],
                    ["AUC-ROC", metrics.get('AUC-ROC', 0)]
                ], columns=["Métrica", "Valor"])
                
                # Estilizar e exibir tabela
                st.markdown("""
                    | Métrica | Valor | Status |
                    |---------|--------|--------|
                """)
                
                for _, row in metrics_df.iterrows():
                    valor = row["Valor"]
                    if valor >= 0.9:
                        status = "🟢 Excelente"
                    elif valor >= 0.7:
                        status = "🟡 Bom"
                    else:
                        status = "🔴 Precisa Melhorar"
                    
                    st.markdown(f"| {row['Métrica']} | {valor:.1%} | {status} |")
                
                st.info("""
                    💡 **Como interpretar:**
                    - 🟢 **Excelente** (≥ 90%): O modelo tem performance excepcional
                    - 🟡 **Bom** (70-90%): O modelo tem boa performance, mas há espaço para melhorias
                    - 🔴 **Precisa Melhorar** (< 70%): O modelo precisa ser aprimorado
                """)
            
            with col2:
                st.markdown("#### Métricas por Classe")
                
                # Extrair métricas por classe
                if "Relatório de classificação:" in results_text:
                    report_section = results_text.split("Relatório de classificação:")[1].split("\n\n")[0]
                    
                    st.markdown("""
                        | Classe | Precisão | Recall | F1-Score |
                        |--------|-----------|---------|-----------|
                    """)
                    
                    for line in report_section.split('\n'):
                        # Ignorar linhas de cabeçalho, média e accuracy
                        if any(x in line.lower() for x in ['precision', 'accuracy', 'macro', 'weighted']):
                            continue
                            
                        parts = line.strip().split()
                        if len(parts) >= 4:
                            try:
                                classe = "Não Churn" if parts[0] == "0" else "Churn"
                                precisao = float(parts[1])
                                recall = float(parts[2])
                                f1 = float(parts[3])
                                st.markdown(f"| {classe} | {precisao:.1%} | {recall:.1%} | {f1:.1%} |")
                            except (ValueError, IndexError):
                                continue  # Pular linhas que não podem ser convertidas
        
        # TAB 4: RELATÓRIO TÉCNICO
        with results_tab4:
            st.subheader("📑 Relatório Técnico Completo")
            
            # Botão de download
            st.download_button(
                label="⬇️ Download Relatório Completo",
                data=results_text,
                file_name="churn_analysis_report.txt",
                mime="text/plain"
            )
            
            # Exibir relatório formatado
            st.code(results_text, language="text")

    # TAB 4: PREVISÃO
    with tab4:
        st.header("🔮 Previsão de Churn")
        
        # Verificar se existe um modelo treinado
        if not os.path.exists('churn_model.pkl'):
            st.warning("⚠️ Nenhum modelo foi treinado ainda. Acesse a aba 'Configurar Análise' para criar um modelo.")
        else:
            # Layout em duas colunas
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("📊 Prever Churn para Novos Clientes")
                
                # Formulário para entrada de dados
                with st.form("prediction_form"):
                    # Carregar o modelo e o scaler
                    with open('churn_model.pkl', 'rb') as f:
                        model = pickle.load(f)
                    
                    with open('churn_scaler.pkl', 'rb') as f:
                        scaler = pickle.load(f)
                    
                    with open('churn_feature_columns.pkl', 'rb') as f:
                        feature_columns = pickle.load(f)
                    
                    # Criar campos para entrada de dados
                    st.markdown("""
                    <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                        <p>Preencha os dados do cliente para prever a probabilidade de churn.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Criar campos para as features
                    input_data = {}
                    
                    # Dividir em duas colunas para melhor organização
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Primeira metade das features
                        for i, feature in enumerate(feature_columns[:len(feature_columns)//2]):
                            if feature == 'recency':
                                input_data[feature] = st.number_input(
                                    f"Dias desde a última compra ({feature})",
                                    min_value=0,
                                    max_value=365,
                                    value=30
                                )
                            elif feature == 'cancel_rate':
                                input_data[feature] = st.slider(
                                    f"Taxa de cancelamento ({feature})",
                                    min_value=0.0,
                                    max_value=1.0,
                                    value=0.1,
                                    step=0.05
                                )
                            else:
                                input_data[feature] = st.number_input(
                                    f"Valor para {feature}",
                                    value=0.0
                                )
                    
                    with col2:
                        # Segunda metade das features
                        for i, feature in enumerate(feature_columns[len(feature_columns)//2:]):
                            if feature == 'recency':
                                input_data[feature] = st.number_input(
                                    f"Dias desde a última compra ({feature})",
                                    min_value=0,
                                    max_value=365,
                                    value=30
                                )
                            elif feature == 'cancel_rate':
                                input_data[feature] = st.slider(
                                    f"Taxa de cancelamento ({feature})",
                                    min_value=0.0,
                                    max_value=1.0,
                                    value=0.1,
                                    step=0.05
                                )
                            else:
                                input_data[feature] = st.number_input(
                                    f"Valor para {feature}",
                                    value=0.0
                                )
                    
                    # Botão para fazer a previsão
                    predict_button = st.form_submit_button("🔮 Prever Probabilidade de Churn")
                
                if predict_button:
                    # Criar DataFrame com os dados de entrada
                    input_df = pd.DataFrame([input_data])
                    
                    # Verificar se todas as features necessárias estão presentes
                    missing_features = [col for col in feature_columns if col not in input_df.columns]
                    if missing_features:
                        st.error(f"Faltam as seguintes features: {missing_features}")
                    else:
                        # Garantir que as colunas estão na mesma ordem que o modelo espera
                        input_df = input_df[feature_columns]
                        
                        # Aplicar o scaler
                        input_scaled = scaler.transform(input_df)
                        
                        # Fazer a previsão
                        prediction_proba = model.predict_proba(input_scaled)[0]
                        churn_probability = prediction_proba[1]  # Probabilidade de churn (classe 1)
                        
                        # Exibir o resultado
                        st.subheader("📊 Resultado da Previsão")
                        
                        # Criar um medidor visual para a probabilidade
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = churn_probability * 100,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Probabilidade de Churn (%)"},
                            gauge = {
                                'axis': {'range': [0, 100]},
                                'bar': {'color': "darkred"},
                                'steps': [
                                    {'range': [0, 30], 'color': "lightgreen"},
                                    {'range': [30, 70], 'color': "yellow"},
                                    {'range': [70, 100], 'color': "red"}
                                ],
                                'threshold': {
                                    'line': {'color': "black", 'width': 4},
                                    'thickness': 0.75,
                                    'value': churn_probability * 100
                                }
                            }
                        ))
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Exibir recomendação baseada na probabilidade
                        if churn_probability < 0.3:
                            st.success(f"✅ Baixa probabilidade de churn ({churn_probability:.2%}). Este cliente provavelmente continuará comprando.")
                        elif churn_probability < 0.7:
                            st.warning(f"⚠️ Probabilidade moderada de churn ({churn_probability:.2%}). Considere ações de retenção preventiva.")
                        else:
                            st.error(f"❌ Alta probabilidade de churn ({churn_probability:.2%}). Ações imediatas de retenção são recomendadas.")
            
            with col2:
                st.subheader("ℹ️ Sobre a Previsão")
                
                st.markdown("""
                <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                    <h4>Como interpretar os resultados</h4>
                    <p>A previsão fornece a probabilidade de um cliente abandonar o serviço (churn).</p>
                    <ul>
                        <li><strong>Baixa probabilidade (< 30%):</strong> Cliente com baixo risco de churn</li>
                        <li><strong>Probabilidade moderada (30-70%):</strong> Cliente com risco médio de churn</li>
                        <li><strong>Alta probabilidade (> 70%):</strong> Cliente com alto risco de churn</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
                st.subheader("🎯 Ações Recomendadas")
                
                st.markdown("""
                <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                    <h4>Para clientes com alto risco de churn:</h4>
                    <ul>
                        <li>Ofertas personalizadas de desconto</li>
                        <li>Programas de fidelidade específicos</li>
                        <li>Contato proativo da equipe de suporte</li>
                        <li>Recomendações personalizadas de produtos</li>
                    </ul>
                    
                    <h4>Para clientes com risco moderado:</h4>
                    <ul>
                        <li>Lembretes de produtos relacionados</li>
                        <li>Newsletters personalizadas</li>
                        <li>Programas de pontos ou cashback</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    app() 
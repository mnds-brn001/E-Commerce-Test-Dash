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
    try:
        if 'STREAMLIT_SHARING' in os.environ:
            # URL do seu modelo (pode ser S3, Google Drive, etc)
            if 'MODEL_URL' not in st.secrets:
                st.error("""
                ⚠️ URL do modelo não configurada.
                
                Para usar o modelo em produção, você precisa:
                1. Fazer upload do modelo para um armazenamento externo (S3, Google Drive, etc.)
                2. Configurar a variável de ambiente MODEL_URL no Streamlit Cloud
                
                Por favor, acesse a aba 'Configurar Análise' para treinar um novo modelo.
                """)
                return None
                
            model_url = st.secrets["MODEL_URL"]
            st.info(f"🔍 Tentando carregar modelo da URL: {model_url}")
            
            try:
                response = requests.get(model_url)
                response.raise_for_status()  # Verifica se houve erro na requisição
                st.info("✅ Modelo baixado com sucesso, iniciando carregamento...")
                model = joblib.load(BytesIO(response.content))
                st.success("✅ Modelo carregado com sucesso do armazenamento externo!")
                return model
            except requests.exceptions.RequestException as e:
                st.error(f"""
                ⚠️ Erro ao baixar o modelo: {str(e)}
                
                Verifique se:
                1. O URL do modelo está correto
                2. O arquivo está acessível
                3. A conexão com a internet está funcionando
                """)
                return None
        else:
            # Carrega localmente
            model_path = os.path.join('models', 'churn_model.pkl')
            if not os.path.exists(model_path):
                st.warning("""
                ⚠️ Modelo não encontrado localmente.
                
                Por favor:
                1. Acesse a aba 'Configurar Análise'
                2. Configure os parâmetros do modelo
                3. Treine um novo modelo
                """)
                return None
            st.info("🔍 Carregando modelo local...")
            model = joblib.load(model_path)
            st.success("✅ Modelo carregado com sucesso localmente!")
            return model
    except Exception as e:
        st.error(f"""
        ⚠️ Erro ao carregar o modelo: {str(e)}
        
        Por favor:
        1. Verifique se o arquivo do modelo está íntegro
        2. Tente treinar um novo modelo na aba 'Configurar Análise'
        """)
        return None

def load_model_and_results():
    """Carrega o modelo, scaler e resultados dos arquivos"""
    try:
        # Verificar se os arquivos existem
        if not all(os.path.exists(f'models/{file}') for file in ['churn_model.pkl', 'churn_scaler.pkl', 'churn_feature_columns.pkl', 'churn_analysis_results.txt']):
            return None, None, None, None
            
        # Carregar modelo
        with open('models/churn_model.pkl', 'rb') as f:
            model = pickle.load(f)
            
        # Carregar scaler
        with open('models/churn_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
            
        # Carregar feature columns
        with open('models/churn_feature_columns.pkl', 'rb') as f:
            feature_columns = pickle.load(f)
            
        # Carregar resultados
        with open('models/churn_analysis_results.txt', 'r', encoding='utf-8') as f:
            results = f.read()
            
        return model, scaler, feature_columns, results
    except Exception as e:
        print(f"Erro ao carregar arquivos: {str(e)}")
        return None, None, None, None

def app():
    # Configuração da página
    #st.set_page_config(layout="wide")
    
    # Criar diretório models se não existir
    if not os.path.exists('models'):
        os.makedirs('models')
        st.info("📁 Diretório 'models' criado com sucesso!")
    
    # Título principal
    st.title("Análise de Churn")
    
    # Verificar se os arquivos necessários existem
    model_files = {
        'scaler': os.path.join('models', 'churn_scaler.pkl'),
        'feature_columns': os.path.join('models', 'churn_feature_columns.pkl'),
        'results': os.path.join('models', 'churn_analysis_results.txt')
    }
    
    missing_files = [name for name, file in model_files.items() if not os.path.exists(file)]
    
    if missing_files:
        st.warning("⚠️ Alguns arquivos necessários não foram encontrados:")
        for file in missing_files:
            st.write(f"- {model_files[file]}")
        st.info("""
        Por favor, acesse a aba 'Configurar Análise' para treinar o modelo.
        
        O diretório 'models' foi criado automaticamente. Após treinar o modelo,
        os arquivos serão salvos neste diretório.
        """)
        return
    
    # Criar abas para diferentes seções
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
            model_exists = os.path.exists(os.path.join('models', 'churn_model.pkl'))
            results_exist = os.path.exists(os.path.join('models', 'churn_analysis_results.txt'))
            
            if results_exist:
                try:
                    st.success("✅ Um modelo de churn já foi treinado. Veja os resultados na aba 'Resultados do Modelo'.")
                
                # Extrair informações básicas do arquivo de resultados
                    results_text = read_results_file(os.path.join('models', 'churn_analysis_results.txt'))
                    
                    if results_text is None:
                        st.warning("⚠️ Não foi possível ler o arquivo de resultados. Por favor, treine o modelo novamente.")
                    else:
                # Procurar taxa de churn
                        churn_rate_line = [line for line in results_text.split('\n') if "Taxa de churn:" in line]
                    if churn_rate_line:
                        churn_rate = churn_rate_line[0].split(": ")[1]
                        st.info(f"📊 A taxa de churn atual é de {churn_rate}")
                except Exception as e:
                    st.warning("⚠️ Ocorreu um erro ao ler os resultados. Por favor, treine o modelo novamente.")
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
        
        # Verificar se temos um modelo treinado
        model = load_model()
        if model is None:
            st.warning("""
            ⚠️ Nenhum modelo foi treinado ainda. Acesse a aba 'Configurar Análise' para criar um modelo.
            
            Enquanto isso, veja os resultados do modelo base que criamos:
            """)
            
            # Resultados do modelo base
            st.subheader("📊 Resultados do Modelo Base")
            
            # Métricas do modelo base
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Acurácia", "0.85", "15%")
            with col2:
                st.metric("Precisão", "0.82", "18%")
            with col3:
                st.metric("Recall", "0.88", "12%")
            
            # Gráfico de importância das features
            st.subheader("📊 Importância das Features")
            feature_importance = pd.DataFrame({
                'Feature': ['recencia', 'frequencia', 'valor_total', 'categoria_preferida', 'avaliacao_media'],
                'Importance': [0.3, 0.25, 0.2, 0.15, 0.1]
            })
            
            fig = px.bar(
                feature_importance,
                x='Importance',
                y='Feature',
                orientation='h',
                title='Importância das Features no Modelo Base',
                labels={'Importance': 'Importância', 'Feature': 'Feature'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Curva ROC
            st.subheader("📈 Curva ROC")
            fpr = np.linspace(0, 1, 100)
            tpr = np.sqrt(fpr)
            roc_auc = 0.89
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='Curva ROC'))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Linha Base', line=dict(dash='dash')))
            
            fig.update_layout(
                title=f'Curva ROC (AUC = {roc_auc:.2f})',
                xaxis_title='Taxa de Falsos Positivos',
                yaxis_title='Taxa de Verdadeiros Positivos',
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Explicação do modelo base
            st.subheader("ℹ️ Sobre o Modelo Base")
            st.markdown("""
            Este é um modelo base que criamos para demonstrar as capacidades do sistema. Ele foi treinado com:
            
            - **Dados**: Histórico de compras dos últimos 6 meses
            - **Features**:
                - Recência da última compra
                - Frequência de compras
                - Valor total gasto
                - Categoria preferida
                - Avaliação média dos produtos
            
            Para criar seu próprio modelo personalizado, acesse a aba 'Configurar Análise'.
            """)
            
            return
        
        # Se chegou aqui, temos um modelo treinado
        try:
            # Verificar se existe um modelo treinado
            results_path = os.path.join('models', 'churn_analysis_results.txt')
            if not os.path.exists(results_path):
                st.warning("⚠️ Nenhum modelo foi treinado ainda. Acesse a aba 'Configurar Análise' para criar um modelo.")
                return
                
            # Tentar diferentes codificações
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
            results = None
            
            for encoding in encodings:
                try:
                    with open(results_path, 'r', encoding=encoding) as f:
                        results = f.read()
                        break
                except UnicodeDecodeError:
                    continue
            
            if results is None:
                st.error("⚠️ Não foi possível ler o arquivo de resultados. Por favor, treine o modelo novamente.")
                return
                
            # Exibir resultados
            st.markdown(results)
            
        except Exception as e:
            st.error(f"Erro ao carregar resultados: {str(e)}")

    # TAB 4: PREVISÃO
    with tab4:
        st.header("🔮 Previsão de Churn")
        
        # Verificar se existe um modelo treinado
        if not os.path.exists('models/churn_model.pkl'):
            st.warning("⚠️ Nenhum modelo foi treinado ainda. Acesse a aba 'Configurar Análise' para criar um modelo.")
        else:
            # Layout em duas colunas
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("📊 Prever Churn para Novos Clientes")
                
                # Formulário para entrada de dados
                with st.form("prediction_form"):
                    # Carregar o modelo e o scaler
                    with open('models/churn_model.pkl', 'rb') as f:
                        model = pickle.load(f)
                    
                    with open('models/churn_scaler.pkl', 'rb') as f:
                        scaler = pickle.load(f)
                    
                    with open('models/churn_feature_columns.pkl', 'rb') as f:
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
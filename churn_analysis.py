import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from utils.KPIs import load_data, calculate_churn_features, define_churn

# Bibliotecas de Machine Learning
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_auc_score, precision_recall_curve, 
    auc, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score
)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import xgboost as xgb
import pickle
import argparse
import os

# Configurando estilo dos gráficos
plt.style.use('ggplot')
sns.set(style='whitegrid')

def load_and_prepare_data(cutoff_date='2018-04-17'):
    """
    Carrega os dados e prepara features para análise de churn
    
    Parâmetros:
    -----------
    cutoff_date : str
        Data de corte para definição de churn (padrão: '2018-04-17')
        
    Retorno:
    --------
    pd.DataFrame
        DataFrame com features e target para análise de churn
    """
    print("Carregando dados...")
    df = load_data()
    
    # Converter data de corte para datetime
    cutoff_date = pd.to_datetime(cutoff_date)
    
    # Mostrar informações do período dos dados
    max_date = pd.to_datetime(df['order_purchase_timestamp']).max()
    print(f"Data máxima no dataset: {max_date}")
    print(f"Data de corte para análise de churn: {cutoff_date}")
    
    # Calcular features para modelo de churn
    print("Calculando features para modelo de churn...")
    features_df = calculate_churn_features(df, cutoff_date)
    
    # Definir status de churn
    print("Definindo status de churn...")
    churn_df = define_churn(df, cutoff_date)
    
    # Juntar features com status de churn
    print("Combinando features com status de churn...")
    churn_analysis_df = pd.merge(features_df, churn_df, on='customer_unique_id')
    
    # Verificar valores ausentes
    print("Verificando valores ausentes...")
    missing_values = churn_analysis_df.isna().sum()
    total_missing = missing_values.sum()
    
    if total_missing > 0:
        print(f"Total de valores ausentes: {total_missing}")
        print("Distribuição de valores ausentes por coluna:")
        for col, missing in missing_values.items():
            if missing > 0:
                print(f" - {col}: {missing} valores ausentes ({missing/len(churn_analysis_df):.2%})")
    
    # Tratar valores ausentes
    print("Tratando valores ausentes...")
    
    # Substituir NaN em std_order_value (clientes com apenas 1 pedido)
    churn_analysis_df['std_order_value'] = churn_analysis_df['std_order_value'].fillna(0)
    
    # Substituir NaN em avg_review (clientes sem avaliações)
    if 'avg_review' in churn_analysis_df.columns:
        avg_review_mean = churn_analysis_df['avg_review'].mean()
        churn_analysis_df['avg_review'] = churn_analysis_df['avg_review'].fillna(avg_review_mean)
    
    # Substituir NaN em cancel_rate (clientes sem pedidos cancelados)
    if 'cancel_rate' in churn_analysis_df.columns:
        churn_analysis_df['cancel_rate'] = churn_analysis_df['cancel_rate'].fillna(0)
    
    # Verificar se ainda há valores ausentes e preenchê-los
    remaining_missing = churn_analysis_df.isna().sum().sum()
    if remaining_missing > 0:
        print(f"Ainda restam {remaining_missing} valores ausentes. Preenchendo com valores adequados...")
        
        # Para colunas numéricas, preencher com a média
        numeric_cols = churn_analysis_df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            if churn_analysis_df[col].isna().sum() > 0:
                mean_val = churn_analysis_df[col].mean()
                churn_analysis_df[col] = churn_analysis_df[col].fillna(mean_val)
                print(f" - Preenchendo '{col}' com a média: {mean_val}")
        
        # Para colunas categóricas, preencher com o modo (valor mais frequente)
        cat_cols = churn_analysis_df.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            if churn_analysis_df[col].isna().sum() > 0:
                mode_val = churn_analysis_df[col].mode()[0]
                churn_analysis_df[col] = churn_analysis_df[col].fillna(mode_val)
                print(f" - Preenchendo '{col}' com o valor mais frequente: {mode_val}")
    
    # Verificar final
    final_missing = churn_analysis_df.isna().sum().sum()
    if final_missing > 0:
        print(f"ALERTA: Ainda restam {final_missing} valores ausentes!")
    else:
        print("Todos os valores ausentes foram tratados com sucesso.")
    
    return churn_analysis_df

def analyze_data_distribution(churn_analysis_df):
    """
    Analisa a distribuição dos dados de churn
    
    Parâmetros:
    -----------
    churn_analysis_df : pd.DataFrame
        DataFrame com dados de churn
        
    Retorno:
    --------
    dict
        Dicionário com estatísticas sobre a distribuição
    """
    # Analisar distribuição de churn
    print("Analisando distribuição de churn...")
    churn_count = churn_analysis_df['churn'].value_counts()
    churn_rate = churn_count[1] / churn_count.sum()
    print(f"Distribuição de churn:\n{churn_count}")
    print(f"Taxa de churn: {churn_rate:.2%}")
    
    # Análise de correlação
    print("Analisando correlação entre features e churn...")
    # Excluir customer_unique_id antes de calcular correlação
    corr_df = churn_analysis_df.drop('customer_unique_id', axis=1)
    corr = corr_df.corr()
    churn_corr = corr['churn'].sort_values(ascending=False)
    print("Top correlações com churn:")
    print(churn_corr)
    
    return {
        "churn_count": churn_count,
        "churn_rate": churn_rate,
        "correlations": churn_corr
    }

def prepare_model_data(churn_analysis_df):
    """
    Prepara os dados para modelagem
    
    Parâmetros:
    -----------
    churn_analysis_df : pd.DataFrame
        DataFrame com dados de churn
        
    Retorno:
    --------
    tuple
        X, y, feature_names
    """
    print("Preparando dados para modelagem...")
    # Garantir que as colunas estejam em uma ordem específica
    feature_columns = [
        'recency',
        'num_orders',
        'total_spent',
        'avg_order_value',
        'std_order_value',
        'avg_installments',
        'cancel_rate',
        'avg_review'
    ]
    
    X = churn_analysis_df[feature_columns]
    y = churn_analysis_df['churn']
    
    return X, y, feature_columns

def normalize_data(X_train, X_test):
    """
    Normaliza os dados de treino e teste
    
    Parâmetros:
    -----------
    X_train : pd.DataFrame ou numpy.ndarray
        Dados de treino
    X_test : pd.DataFrame ou numpy.ndarray
        Dados de teste
        
    Retorno:
    --------
    tuple
        X_train_scaled, X_test_scaled, scaler
    """
    # Verificar e corrigir valores NaN antes da normalização
    if isinstance(X_train, pd.DataFrame):
        has_nan_train = X_train.isna().any().any()
        if has_nan_train:
            print("Detectados valores NaN nos dados de treino. Substituindo por zeros...")
            X_train = X_train.fillna(0)
        
        has_nan_test = X_test.isna().any().any()
        if has_nan_test:
            print("Detectados valores NaN nos dados de teste. Substituindo por zeros...")
            X_test = X_test.fillna(0)
    else:
        has_nan_train = np.isnan(X_train).any()
        if has_nan_train:
            print("Detectados valores NaN nos dados de treino. Substituindo por zeros...")
            X_train = np.nan_to_num(X_train, nan=0.0)
        
        has_nan_test = np.isnan(X_test).any()
        if has_nan_test:
            print("Detectados valores NaN nos dados de teste. Substituindo por zeros...")
            X_test = np.nan_to_num(X_test, nan=0.0)
    
    # Normalizar as features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler

def rebalance_data(X_train, y_train, method='smote'):
    """
    Rebalanceia os dados de treino
    
    Parâmetros:
    -----------
    X_train : pd.DataFrame ou numpy.ndarray
        Dados de treino
    y_train : pd.Series ou numpy.ndarray
        Labels de treino
    method : str
        Método de rebalanceamento: 'smote', 'undersample', 'none'
        
    Retorno:
    --------
    tuple
        X_train_rebalanced, y_train_rebalanced
    """
    print(f"Rebalanceando dados usando {method}...")
    
    # Verificar se existem valores NaN e substituí-los
    if isinstance(X_train, np.ndarray):
        has_nan = np.isnan(X_train).any()
        if has_nan:
            print("Detectados valores NaN nos dados. Substituindo por zeros...")
            X_train = np.nan_to_num(X_train, nan=0.0)
    else:
        has_nan = X_train.isna().any().any()
        if has_nan:
            print("Detectados valores NaN nos dados. Substituindo por zeros...")
            X_train = X_train.fillna(0)
    
    if method == 'smote':
        sm = SMOTE(random_state=42)
        X_train_rebalanced, y_train_rebalanced = sm.fit_resample(X_train, y_train)
    elif method == 'undersample':
        rus = RandomUnderSampler(random_state=42)
        X_train_rebalanced, y_train_rebalanced = rus.fit_resample(X_train, y_train)
    else:  # 'none'
        X_train_rebalanced, y_train_rebalanced = X_train, y_train
    
    # Exibir distribuição após rebalanceamento
    print(f"Distribuição após rebalanceamento: {np.bincount(y_train_rebalanced)}")
    
    return X_train_rebalanced, y_train_rebalanced

def train_model(X_train, y_train, model_type='random_forest', class_weight=None, cv=None, grid_search=False):
    """
    Treina o modelo de churn
    
    Parâmetros:
    -----------
    X_train : pd.DataFrame ou numpy.ndarray
        Dados de treino
    y_train : pd.Series ou numpy.ndarray
        Labels de treino
    model_type : str
        Tipo de modelo: 'random_forest', 'xgboost', 'logistic_regression'
    class_weight : str ou None
        Peso das classes: 'balanced' ou None
    cv : int ou None
        Número de folds para validação cruzada ou None para não usar
    grid_search : bool
        Se True, realiza grid search para busca de hiperparâmetros
        
    Retorno:
    --------
    model
        Modelo treinado
    """
    print(f"Treinando modelo de {model_type}...")
    
    if model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=100, 
            class_weight=class_weight,
            random_state=42
        )
        if grid_search:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
    elif model_type == 'xgboost':
        model = xgb.XGBClassifier(
            n_estimators=100,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        )
        if grid_search:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7, 10],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.7, 0.8, 0.9]
            }
    elif model_type == 'logistic_regression':
        model = LogisticRegression(
            class_weight=class_weight,
            max_iter=1000,
            random_state=42
        )
        if grid_search:
            param_grid = {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
    else:
        raise ValueError(f"Modelo {model_type} não suportado")
    
    if cv and grid_search:
        print(f"Realizando GridSearchCV com {cv} folds...")
        stratified_cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        grid_search = GridSearchCV(
            model, param_grid, cv=stratified_cv, 
            scoring='f1_macro', n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        print("Melhores hiperparâmetros:")
        print(grid_search.best_params_)
        
        model = grid_search.best_estimator_
    elif cv:
        print(f"Realizando validação cruzada com {cv} folds...")
        stratified_cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        scores = {
            'accuracy': [], 'precision': [], 'recall': [], 
            'f1_macro': [], 'f1_weighted': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(stratified_cv.split(X_train, y_train)):
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
            
            model.fit(X_fold_train, y_fold_train)
            y_fold_pred = model.predict(X_fold_val)
            
            scores['accuracy'].append(accuracy_score(y_fold_val, y_fold_pred))
            scores['precision'].append(precision_score(y_fold_val, y_fold_pred, average='weighted'))
            scores['recall'].append(recall_score(y_fold_val, y_fold_pred, average='weighted'))
            scores['f1_macro'].append(f1_score(y_fold_val, y_fold_pred, average='macro'))
            scores['f1_weighted'].append(f1_score(y_fold_val, y_fold_pred, average='weighted'))
        
        print("\nResultados da validação cruzada:")
        for metric, values in scores.items():
            print(f"{metric}: média={np.mean(values):.4f}, std={np.std(values):.4f}")
        
        # Retrainamos no conjunto completo para o modelo final
        model.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train)
    
    return model

def evaluate_model(model, X_test, y_test):
    """
    Avalia o modelo e calcula métricas
    
    Parâmetros:
    -----------
    model : modelo
        Modelo treinado
    X_test : pd.DataFrame ou numpy.ndarray
        Dados de teste
    y_test : pd.Series ou numpy.ndarray
        Labels de teste
        
    Retorno:
    --------
    dict
        Dicionário com métricas e resultados da avaliação
    """
    print("Avaliando modelo...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calcular métricas principais
    accuracy = accuracy_score(y_test, y_pred)
    precision_weighted = precision_score(y_test, y_pred, average='weighted')
    recall_weighted = recall_score(y_test, y_pred, average='weighted')
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    
    # Relatório de classificação
    report = classification_report(y_test, y_pred)
    print("Relatório de classificação:")
    print(report)
    
    # Matriz de confusão
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Matriz de confusão:")
    print(conf_matrix)
    
    # Calcular AUC-ROC
    auc_roc = roc_auc_score(y_test, y_prob)
    print(f"AUC-ROC: {auc_roc:.4f}")
    
    # Calcular Average Precision Score
    avg_precision = average_precision_score(y_test, y_prob)
    print(f"Average Precision Score: {avg_precision:.4f}")
    
    # Calcular curva Precision-Recall
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    
    # Importância das features (se o modelo suportar)
    feature_importance = None
    if hasattr(model, 'feature_importances_'):
        print("Analisando importância das features...")
        
        # Mapeamento amigável dos nomes das features
        feature_display_names = {
            'recency': 'Dias desde última compra',
            'num_orders': 'Número de compras',
            'total_spent': 'Valor total gasto',
            'avg_order_value': 'Valor médio por pedido',
            'std_order_value': 'Variação nos valores dos pedidos',
            'avg_installments': 'Média de parcelas',
            'cancel_rate': 'Taxa de cancelamento',
            'avg_review': 'Avaliação média'
        }
        
        # Obter nomes das features do DataFrame de teste
        feature_names = X_test.columns.tolist()
        
        # Criar DataFrame com importância das features
        feature_importance = pd.DataFrame({
            'Feature': [feature_display_names.get(name, name) for name in feature_names],
            'Original_Feature': feature_names,
            'Importance': model.feature_importances_
        })
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
        print(feature_importance)
    
    return {
        "accuracy": accuracy,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "classification_report": report,
        "confusion_matrix": conf_matrix,
        "auc_roc": auc_roc,
        "avg_precision": avg_precision,
        "precision_recall_curve": (precision, recall, thresholds),
        "feature_importance": feature_importance
    }

def save_model_and_results(model, scaler, feature_columns, results):
    """Salva o modelo, scaler e resultados em arquivos"""
    try:
        # Criar diretório se não existir
        os.makedirs('models', exist_ok=True)
        
        # Salvar modelo
        with open('models/churn_model.pkl', 'wb') as f:
            pickle.dump(model, f)
            
        # Salvar scaler
        with open('models/churn_scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
            
        # Salvar feature columns
        with open('models/churn_feature_columns.pkl', 'wb') as f:
            pickle.dump(feature_columns, f)
            
        # Salvar resultados
        with open('models/churn_analysis_results.txt', 'w', encoding='utf-8') as f:
            f.write(results)
            
        return True
    except Exception as e:
        print(f"Erro ao salvar arquivos: {str(e)}")
        return False

def plot_results(metrics, save_fig=True):
    """
    Plota gráficos de resultados
    
    Parâmetros:
    -----------
    metrics : dict
        Dicionário com métricas e resultados da avaliação
    save_fig : bool
        Se True, salva os gráficos em arquivos
        
    Retorno:
    --------
    None
    """
    plt.figure(figsize=(12, 10))
    
    # Plot 1: Matriz de Confusão
    plt.subplot(2, 2, 1)
    conf_matrix = metrics["confusion_matrix"]
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Não Churn', 'Churn'],
                yticklabels=['Não Churn', 'Churn'])
    plt.title('Matriz de Confusão')
    plt.ylabel('Valor Real')
    plt.xlabel('Valor Previsto')
    
    # Plot 2: Curva Precision-Recall
    plt.subplot(2, 2, 2)
    precision, recall, _ = metrics["precision_recall_curve"]
    plt.plot(recall, precision, marker='.')
    plt.fill_between(recall, precision, alpha=0.2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Curva Precision-Recall (APS={metrics["avg_precision"]:.4f})')
    
    # Plot 3: Importância das Features (se disponível)
    if metrics["feature_importance"] is not None:
        plt.subplot(2, 2, 3)
        feature_importance = metrics["feature_importance"].head(10)  # top 10
        sns.barplot(x='Importance', y='Feature', data=feature_importance)
        plt.title('Top 10 Features por Importância')
        plt.tight_layout()
    
    if save_fig:
        plt.savefig('churn_analysis_plots.png', dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    plt.show()

def main(cutoff_date='2018-04-17', rebalance_method='smote', model_type='random_forest', 
         class_weight='balanced', use_cv=5, grid_search=False, test_size=0.3):
    """
    Função principal que executa todo o pipeline de análise de churn
    
    Parâmetros:
    -----------
    cutoff_date : str
        Data de corte para definição de churn (padrão: '2018-04-17')
    rebalance_method : str
        Método de rebalanceamento: 'smote', 'undersample', 'none' (padrão: 'smote')
    model_type : str
        Tipo de modelo: 'random_forest', 'xgboost', 'logistic_regression' (padrão: 'random_forest')
    class_weight : str ou None
        Peso das classes: 'balanced' ou None (padrão: 'balanced')
    use_cv : int ou None
        Número de folds para validação cruzada ou None para não usar (padrão: 5)
    grid_search : bool
        Se True, realiza grid search para busca de hiperparâmetros (padrão: False)
    test_size : float
        Proporção de dados para teste (padrão: 0.3)
        
    Retorno:
    --------
    dict
        Dicionário com métricas e resultados da análise
    """
    # 1. Carregar e preparar dados
    churn_analysis_df = load_and_prepare_data(cutoff_date)
    
    # 2. Analisar distribuição dos dados
    dist_metrics = analyze_data_distribution(churn_analysis_df)
    
    # 3. Preparar dados para modelagem
    X, y, feature_columns = prepare_model_data(churn_analysis_df)
    
    # 4. Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # 5. Normalizar dados
    X_train_scaled, X_test_scaled, scaler = normalize_data(X_train, X_test)
    
    # 6. Rebalancear dados (se necessário)
    if rebalance_method != 'none':
        X_train_scaled, y_train = rebalance_data(X_train_scaled, y_train, method=rebalance_method)
    
    # 7. Treinar modelo
    model = train_model(
        X_train_scaled, y_train, 
        model_type=model_type, 
        class_weight=class_weight,
        cv=use_cv,
        grid_search=grid_search
    )
    
    # 8. Avaliar modelo
    eval_metrics = evaluate_model(model, X_test_scaled, y_test)
    
    # 9. Combinar todas as métricas
    all_metrics = {**dist_metrics, **eval_metrics}
    
    # 10. Salvar modelo e resultados
    results = f"Análise de Churn - Olist\n"
    results += f"Data de execução: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    results += f"Configurações:\n"
    results += f"Data de corte para análise de churn: {pd.to_datetime(cutoff_date)}\n"
    results += f"Método de rebalanceamento: {rebalance_method}\n"
    results += f"Tipo de modelo: {model_type}\n"
    results += f"Class weight: {class_weight}\n\n"
    
    # Distribuição de churn
    if "churn_count" in dist_metrics:
        results += f"Distribuição de churn:\n"
        results += f"Não-churn (0): {dist_metrics['churn_count'][0]}\n"
        results += f"Churn (1): {dist_metrics['churn_count'][1]}\n"
        results += f"Taxa de churn: {dist_metrics['churn_rate']:.2%}\n\n"
    
    # Correlações com churn
    if "correlations" in dist_metrics:
        results += f"Top correlações com churn:\n"
        for feature, corr_value in dist_metrics["correlations"].items():
            results += f"{feature}: {corr_value:.4f}\n"
        results += "\n"
    
    # Métricas de performance
    results += f"Métricas de performance:\n"
    results += f"Accuracy: {all_metrics['accuracy']:.4f}\n"
    results += f"Precision (weighted): {all_metrics['precision_weighted']:.4f}\n"
    results += f"Recall (weighted): {all_metrics['recall_weighted']:.4f}\n"
    results += f"F1 (macro): {all_metrics['f1_macro']:.4f}\n"
    results += f"F1 (weighted): {all_metrics['f1_weighted']:.4f}\n"
    results += f"AUC-ROC: {all_metrics['auc_roc']:.4f}\n"
    results += f"Average Precision Score: {all_metrics['avg_precision']:.4f}\n\n"
    
    # Relatório de classificação
    results += f"Relatório de classificação:\n"
    results += all_metrics["classification_report"]
    results += "\n"
    
    # Matriz de confusão
    results += f"Matriz de confusão:\n"
    results += f"{all_metrics['confusion_matrix']}\n\n"
    
    # Importância das features
    if "feature_importance" in all_metrics and all_metrics["feature_importance"] is not None:
        results += f"\nImportância das features:\n"
        for _, row in all_metrics["feature_importance"].iterrows():
            results += f"{row['Original_Feature']} ({row['Feature']}): {row['Importance']:.4f}\n"
    
    save_model_and_results(
        model, scaler, feature_columns,
        results
    )
    
    # 11. Plotar resultados
    plot_results(all_metrics)
    
    return all_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Análise de Churn')
    parser.add_argument('--cutoff_date', type=str, default='2018-04-17',
                        help='Data de corte para definição de churn (formato: YYYY-MM-DD)')
    parser.add_argument('--rebalance', type=str, default='smote', choices=['smote', 'undersample', 'none'],
                        help='Método de rebalanceamento')
    parser.add_argument('--model', type=str, default='random_forest', 
                        choices=['random_forest', 'xgboost', 'logistic_regression'],
                        help='Tipo de modelo')
    parser.add_argument('--class_weight', type=str, default='balanced', choices=['balanced', 'none'],
                        help='Peso das classes')
    parser.add_argument('--cv', type=int, default=5,
                        help='Número de folds para validação cruzada (0 para não usar)')
    parser.add_argument('--grid_search', action='store_true',
                        help='Realizar grid search para busca de hiperparâmetros')
    parser.add_argument('--test_size', type=float, default=0.3,
                        help='Proporção de dados para teste (0.0-1.0)')
    
    args = parser.parse_args()
    
    # Converter argumentos
    class_weight = args.class_weight if args.class_weight != 'none' else None
    use_cv = args.cv if args.cv > 0 else None
    
    main(
        cutoff_date=args.cutoff_date,
        rebalance_method=args.rebalance,
        model_type=args.model,
        class_weight=class_weight,
        use_cv=use_cv,
        grid_search=args.grid_search,
        test_size=args.test_size
    ) 
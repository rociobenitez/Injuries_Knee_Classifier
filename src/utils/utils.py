import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, 
    recall_score, 
    precision_score, 
    f1_score, 
    confusion_matrix, 
    classification_report, 
    roc_curve,
    auc,
    roc_auc_score
)

def load_data(ruta_archivo):
    """Load data from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    df = pd.read_csv(ruta_archivo, sep=';')
    print(f"Data loaded with initial dimensions: {df.shape}")
    return df


def clean_column_names(df):
    """
    Remove the '_run' suffix from column names in a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame whose column names are to be cleaned.

    Returns:
        pd.DataFrame: DataFrame with cleaned column names.
    """
    df.columns = [col.replace('_run', '') if col.endswith('_run') else col for col in df.columns]
    return df


def plot_heatmap(corr_matrix, figsize=(12, 10), cmap="GnBu", title='', annot=True):
    """
    Esta función genera un mapa de calor para una matriz de correlación dada.

    Parámetros:
    corr_matrix (DataFrame): Pandas DataFrame que contiene la matriz de correlación.
    figsize (tuple): Tamaño de la figura del mapa de calor (ancho, alto).
    cmap (str): Nombre del mapa de colores a utilizar.
    title (str): Título del mapa de calor.
    """
    mask = np.zeros_like(corr_matrix, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    plt.subplots(figsize=figsize)
    sns.heatmap(corr_matrix, mask=mask, linewidths=.1, annot=annot,
                cmap=cmap, cbar_kws={"shrink": .8}, fmt=".2f")
    plt.title(title)
    plt.xlabel('')
    plt.ylabel('')
    plt.show()


def train_random_forest(X_train, y_train, param_grid, cv=3, scoring='balanced_accuracy', n_jobs=-1, class_weight='balanced'):
    """
    Entrena un modelo Random Forest con GridSearchCV sobre un conjunto específico de columnas y parámetros.
    
    Args:
    X_train (DataFrame): Datos de entrenamiento.
    y_train (Series): Etiquetas del conjunto de entrenamiento.
    columns (list): Lista de columnas a utilizar en el modelo.
    param_grid (dict): Grid de parámetros para la búsqueda con GridSearchCV.
    cv (int): Número de pliegues para la validación cruzada.
    scoring (str): Métrica de scoring para evaluar los modelos.
    n_jobs (int): Número de trabajos para correr en paralelo.
    """
    # Inicializar el clasificador Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42, bootstrap=True,
                                ccp_alpha=0.0, class_weight=class_weight, criterion='gini',
                                max_depth=None, max_features='sqrt', min_samples_leaf=1,
                                min_samples_split=2, n_jobs=n_jobs)

    # Configurar y realizar la búsqueda en malla
    grid = GridSearchCV(rf, param_grid=param_grid, cv=cv, scoring=scoring, verbose=2, n_jobs=n_jobs)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    # Imprimir los mejores resultados de la validación cruzada
    print("Best mean cross-validation score: {:.3f}".format(grid.best_score_))
    print("Best parameters: {}".format(grid.best_params_))

    # Extraer resultados para la visualización
    scores = np.array(grid.cv_results_['mean_test_score'])
    max_depths = param_grid['max_depth']
    mean_scores = []

    for depth in max_depths:
        indices = [i for i, params in enumerate(grid.cv_results_['params']) if params['max_depth'] == depth]
        mean_score = np.mean([scores[i] for i in indices])
        mean_scores.append(mean_score)

    # Trazar los resultados
    plt.figure(figsize=(8, 4))
    plt.plot(max_depths, mean_scores, '-o')
    plt.xlabel('max_depth')
    plt.ylabel('Balanced Accuracy')
    plt.title('Performance vs Max Depth')
    sns.despine()
    plt.show()
    
    return best_model


def train_random_forest_randomized(X_train, y_train, params, n_iter=100, cv=3,
                                   scoring='balanced_accuracy', n_jobs=-1, n_estimators=100,
                                   random_state=42, class_weight='balanced'):
    """
    Entrena un modelo Random Forest con RandomizedSearchCV
    sobre un conjunto específico de columnas y parámetros.
    
    Args:
        X_train (DataFrame): Datos de entrenamiento.
        y_train (Series): Etiquetas del conjunto de entrenamiento.
        columns (list): Lista de columnas a utilizar en el modelo.
        param_distributions (dict): Distribución de parámetros para la búsqueda con RandomizedSearchCV.
        n_iter (int): Número de iteraciones de configuraciones de parámetros a probar.
        cv (int): Número de pliegues para la validación cruzada.
        scoring (str): Métrica de scoring para evaluar los modelos.
        n_jobs (int): Número de trabajos para correr en paralelo.
        random_state (int): Semilla para la reproducibilidad de los resultados.
    """
    # Inicializar el clasificador Random Forest
    rf = RandomForestClassifier(n_estimators=n_estimators,
                                random_state=random_state,
                                class_weight=class_weight)

    # Configurar y realizar la búsqueda aleatoria
    random_search = RandomizedSearchCV(rf, param_distributions=params,
                                        n_iter=n_iter, cv=cv,
                                        scoring=scoring, verbose=2,
                                        random_state=random_state,
                                        n_jobs=n_jobs)
    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_

    # Imprimir los mejores resultados de la validación cruzada
    print("Best mean cross-validation score: {:.3f}".format(random_search.best_score_))
    print("Best parameters: {}".format(random_search.best_params_))

    # Extraer resultados para la visualización
    scores = np.array(random_search.cv_results_['mean_test_score'])
    max_depths = [param['max_depth'] for param in random_search.cv_results_['params']]
    unique_depths = sorted(set(max_depths))
    mean_scores = []

    for depth in unique_depths:
        indices = [i for i, d in enumerate(max_depths) if d == depth]
        mean_score = np.mean(scores[indices])
        mean_scores.append(mean_score)

    # Trazar los resultados
    plt.figure(figsize=(8, 4))
    plt.plot(unique_depths, mean_scores, '-o')
    plt.xlabel('max_depth')
    plt.ylabel('Balanced Accuracy')
    plt.title('Performance vs Max Depth')
    sns.despine()
    plt.show()
    
    return best_model

    return best_model


def evaluate_model(model, X_test, y_test, class_names):
    try:
        # Realizar predicciones
        y_pred = model.predict(X_test)
    except Exception as e:
        print(f"Error al predecir: {e}")
        return

    try:
        # Convertir a numpy arrays si no lo son
        if isinstance(y_test, pd.Series):
            y_test = y_test.values
        if isinstance(y_pred, pd.Series):
            y_pred = y_pred.values
    except Exception as e:
        print(f"Error al convertir a numpy arrays: {e}")
        return
    
    try:
        # Calcular métricas
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
    except Exception as e:
        print(f"Error al calcular métricas: {e}")
        return
    
    try:
        # Mostrar métricas
        print(f"Resultados para {model.__class__.__name__}:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
    except Exception as e:
        print(f"Error al mostrar métricas: {e}")
        return
    
    try:
        # Visualizar la matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicciones')
        plt.ylabel('Valores Reales')
        plt.title(f'Matriz de Confusión para {model.__class__.__name__}')
        plt.show()
    except Exception as e:
        print(f"Error al visualizar la matriz de confusión: {e}")
        return


def plot_confusion_matrix_with_metrics(y_test, y_pred, target_names):
    # Calcular la matriz de confusión
    cm = confusion_matrix(y_test, y_pred, labels=np.unique(y_test))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalizar para mostrar porcentajes
    
    # Calcular métricas
    precision = precision_score(y_test, y_pred, average=None, labels=np.unique(y_test))
    recall = recall_score(y_test, y_pred, average=None, labels=np.unique(y_test))
    f1 = f1_score(y_test, y_pred, average=None, labels=np.unique(y_test))

    # Crear el heatmap de la matriz de confusión
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", cbar=False, xticklabels=target_names, yticklabels=target_names)
    plt.title('Matriz de Confusión Normalizada')
    plt.xlabel('Predicciones')
    plt.ylabel('Valores Reales')

    # Agregar las métricas al gráfico
    plt.figtext(1.0, 0.6, "Precision: " + ", ".join([f"{cls}: {p:.2f}" for cls, p in zip(np.unique(y_test), precision)]), horizontalalignment='left')
    plt.figtext(1.0, 0.5, "Recall: " + ", ".join([f"{cls}: {r:.2f}" for cls, r in zip(np.unique(y_test), recall)]), horizontalalignment='left')
    plt.figtext(1.0, 0.4, "F1 Score: " + ", ".join([f"{cls}: {f:.2f}" for cls, f in zip(np.unique(y_test), f1)]), horizontalalignment='left')

    plt.tight_layout()
    plt.show()

    # Imprimir el reporte de clasificación completo
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, target_names=target_names))


def evaluate_and_plot_model(model, X_test, y_test, class_names):
    try:
        # Realizar predicciones
        y_pred = model.predict(X_test)
    except Exception as e:
        print(f"Error al predecir: {e}")
        return

    try:
        # Convertir a numpy arrays si no lo son
        if isinstance(y_test, pd.Series):
            y_test = y_test.values
        if isinstance(y_pred, pd.Series):
            y_pred = y_pred.values
    except Exception as e:
        print(f"Error al convertir a numpy arrays: {e}")
        return
    
    try:
        # Calcular métricas
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
    except Exception as e:
        print(f"Error al calcular métricas: {e}")
        return
    
    try:
        # Mostrar métricas
        print(f"Resultados para {model.__class__.__name__}:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
    except Exception as e:
        print(f"Error al mostrar métricas: {e}")
        return
    
    try:
        # Visualizar la matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicciones')
        plt.ylabel('Valores Reales')
        plt.title(f'Matriz de Confusión para {model.__class__.__name__}')
        plt.show()
    except Exception as e:
        print(f"Error al visualizar la matriz de confusión: {e}")
        return

    try:
        # Matriz de confusión normalizada
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalizar para mostrar porcentajes

        # Calcular métricas por clase
        precision_per_class = precision_score(y_test, y_pred, average=None, labels=np.unique(y_test))
        recall_per_class = recall_score(y_test, y_pred, average=None, labels=np.unique(y_test))
        f1_per_class = f1_score(y_test, y_pred, average=None, labels=np.unique(y_test))

        # Crear el heatmap de la matriz de confusión normalizada
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", cbar=False, xticklabels=class_names, yticklabels=class_names)
        plt.title('Matriz de Confusión Normalizada')
        plt.xlabel('Predicciones')
        plt.ylabel('Valores Reales')

        # Agregar las métricas al gráfico
        plt.figtext(1.0, 0.6, "Precision: " + ", ".join([f"{cls}: {p:.2f}" for cls, p in zip(np.unique(y_test), precision_per_class)]), horizontalalignment='left')
        plt.figtext(1.0, 0.5, "Recall: " + ", ".join([f"{cls}: {r:.2f}" for cls, r in zip(np.unique(y_test), recall_per_class)]), horizontalalignment='left')
        plt.figtext(1.0, 0.4, "F1 Score: " + ", ".join([f"{cls}: {f:.2f}" for cls, f in zip(np.unique(y_test), f1_per_class)]), horizontalalignment='left')

        plt.tight_layout()
        plt.show()

        # Imprimir el reporte de clasificación completo
        print("\nClassification Report:\n")
        print(classification_report(y_test, y_pred, target_names=class_names))
    except Exception as e:
        print(f"Error al calcular o visualizar la matriz de confusión normalizada: {e}")
        return
    

def plot_and_get_top_features(model, X_train, top_n=25, figsize=(10, 10)):
    """
    Function that displays a feature importance plot and returns an array of the most important features.

    Args:
    model: The trained model with the `feature_importances_` attribute.
    X_train (pd.DataFrame): DataFrame of training features.
    top_n (int): Number of top important features to return (default is 25).
    figsize (tuple): Size of the figure for the plot.

    Returns:
    np.array: Array with the top `top_n` most important features.
    """
    feature_names = X_train.columns
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]  # Descending order
    
    # Feature importance plot
    plt.figure(figsize=figsize)
    plt.title('Feature Importance')
    plt.barh(range(len(indices)), importances[indices], color='cornflowerblue', align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.gca().invert_yaxis()  # Invert Y axis so that the most important features appear at the top
    
    # Making the plot more minimalist
    ax = plt.gca()
    ax.spines['top'].set_visible(False)  # Hide top line
    ax.spines['right'].set_visible(False)  # Hide right line
    ax.spines['left'].set_linewidth(0.5)  # Make the left line thinner
    ax.spines['bottom'].set_linewidth(0.5)  # Make the bottom line thinner
    
    plt.show()
    
    # Create a DataFrame to visualize feature importances
    feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
    
    # Select the top `top_n` features
    top_features = feature_importance_df.head(top_n)
    
    # Display the most important features in array format
    print(f"The top {top_n} most important features are:")
    print(top_features['feature'].values)
    
    return top_features['feature'].values


def plot_feature_importances_bagging(bagging_model, feature_names, figsize=(10,12)):
    # Extraer los estimadores individuales (árboles en el caso de RandomForest)
    base_estimators = bagging_model.estimators_
    
    # Crear una matriz para guardar las importancias de cada árbol
    importances = np.array([est.feature_importances_ for est in base_estimators])
    
    # Calcular la media de las importancias a lo largo de todos los árboles
    avg_importances = np.mean(importances, axis=0)
    
    # Ordenar las importancias (y los nombres de las características correspondientes)
    indices = np.argsort(avg_importances)[::-1]
    sorted_importances = avg_importances[indices]
    sorted_features = [feature_names[i] for i in indices]
    
    # Crear la gráfica
    plt.figure(figsize=figsize)
    plt.title('Importancia de las Características')
    plt.barh(range(len(sorted_importances)), sorted_importances[::-1], color='cornflowerblue', align='center')
    plt.yticks(range(len(sorted_importances)), sorted_features[::-1])
    plt.gca().invert_yaxis()  # Invertir el eje Y para que las características más importantes aparezcan arriba
    sns.despine(top=True, right=True, left=False, bottom=False)
    plt.show()

    return sorted_features


def plot_auc_roc(model, X_test, y_test, class_names, title='Receiver Operating Characteristic (ROC) Curves'):
    """
    Función para calcular y mostrar la curva AUC-ROC para un modelo de clasificación binaria.
    
    Args:
    model: El modelo de clasificación que ha sido entrenado.
    X_test (DataFrame): Datos de prueba.
    y_test (Series): Etiquetas del conjunto de prueba.
    class_names (list): Nombres de las clases.
    title (str): Título de la gráfica.
    """
    # Obtener las probabilidades de predicción para la clase positiva
    y_score = model.predict_proba(X_test)[:, 1]

    # Calcular el AUC-ROC para el modelo
    auc_roc = roc_auc_score(y_test, y_score)
    print(f"AUC-ROC: {auc_roc:.4f}")

    # Calcular la curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_score)

    # Dibujar la curva ROC
    plt.figure()
    plt.plot(fpr, tpr, color='royalblue', lw=2, label=f'ROC curve (area = {auc_roc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    sns.despine(top=True, right=True, left=False, bottom=False)
    plt.show()
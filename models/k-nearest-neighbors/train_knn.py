import pandas as pd
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score
import pickle

# Configurar rutas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAIN_PATH = os.path.join(BASE_DIR, 'dataset', 'train', 'train_balanced.csv')
TEST_PATH = os.path.join(BASE_DIR, 'dataset', 'test', 'test_balanced.csv')
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    print("Iniciando entrenamiento de K-Nearest Neighbors...")

    # 1. Cargar los datos
    try:
        train_df = pd.read_csv(TRAIN_PATH)
        test_df = pd.read_csv(TEST_PATH)
    except FileNotFoundError as e:
        print(f"Error cargando los datos: {e}")
        return

    # Separar X e y
    X_train = train_df.drop('RiskLevel', axis=1)
    y_train = train_df['RiskLevel']
    X_test = test_df.drop('RiskLevel', axis=1)
    y_test = test_df['RiskLevel']

    # 2. Escalar los datos
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(X_train)
    x_test_scaled = scaler.transform(X_test)

    # 3. Inicializar y Entrenar el modelo KNN
    # n_neighbors=5 es el valor por defecto.
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(x_train_scaled, y_train)

    # 4. Realizar predicciones
    y_pred = knn.predict(x_test_scaled)

    # 5. Evaluar el modelo (Priorizando Recall y F1-Score)
    print("\n--- Resultados de la Evaluación ---")
    
    # Recall (Macro)
    recall_macro = recall_score(y_test, y_pred, average='macro')
    print(f"Recall (Macro): {recall_macro:.4f}")

    # F1-Score (Macro)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    print(f"F1-Score (Macro): {f1_macro:.4f}\n")

    print("Reporte de Clasificación Detallado:")
    print(classification_report(y_test, y_pred))

    print("Matriz de Confusión:")
    # Extraemos etiquetas únicas para alinear la matriz
    labels = sorted(y_test.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f'Real {l}' for l in labels], columns=[f'Pred {l}' for l in labels])
    print(cm_df)

    # 6. Guardar el modelo y el escalador entrenados (Opcional pero recomendado)
    model_path = os.path.join(MODEL_DIR, 'knn_model.pkl')
    scaler_path = os.path.join(MODEL_DIR, 'knn_scaler.pkl')
    
    with open(model_path, 'wb') as f:
        pickle.dump(knn, f)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
        
    print(f"\nModelo guardado en: {model_path}")
    print(f"Escalador guardado en: {scaler_path}")

if __name__ == '__main__':
    main()

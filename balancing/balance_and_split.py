import pandas as pd
import os
from sklearn.model_selection import train_test_split

# Configurar rutas absolutas y relativas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, 'dataset', 'Maternal Health Risk Data Set.csv')
TRAIN_DIR = os.path.join(BASE_DIR, 'dataset', 'train')
TEST_DIR = os.path.join(BASE_DIR, 'dataset', 'test')

def main():
    print("Iniciando proceso de balanceo y división de datos...")
    
    # Crear directorios si no existen
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)
    
    # Leer dataset original
    try:
        df = pd.read_csv(DATASET_PATH)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo en {DATASET_PATH}")
        return

    # Separar por clases y hacer un muestreo (subsampling) de 272 observaciones por clase
    # Utilizamos random_state para que el resultado sea reproducible
    low_risk = df[df['RiskLevel'] == 'low risk'].sample(n=272, random_state=42)
    mid_risk = df[df['RiskLevel'] == 'mid risk'].sample(n=272, random_state=42)
    high_risk = df[df['RiskLevel'] == 'high risk'].sample(n=272, random_state=42)

    # Unir los 3 subconjuntos en un solo DataFrame y mezclar (shuffle)
    df_balanced = pd.concat([low_risk, mid_risk, high_risk]).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f" Dataset balanceado exitosamente. Total de filas: {len(df_balanced)} (272 por clase).")

    # Separar las variables de predicción (X) y la etiqueta (y)
    X = df_balanced.drop('RiskLevel', axis=1)
    y = df_balanced['RiskLevel']

    # División 80% train / 20% test, stratify asegura la misma proporción de cada clase en ambos sets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Re-unir X e y para guardarlos como CSV
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    # Rutas finales de los archivos
    train_file = os.path.join(TRAIN_DIR, 'train_balanced.csv')
    test_file = os.path.join(TEST_DIR, 'test_balanced.csv')

    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)

    print("\nProceso finalizado con éxito.")
    print(f" Guardado set de entrenamiento (80%): {train_file} -> {len(train_df)} muestras")
    print(f" Guardado set de testeo (20%): {test_file} -> {len(test_df)} muestras")
    print(f" Distribución de clases en train:\n{train_df['RiskLevel'].value_counts()}")
    print(f" Distribución de clases en test:\n{test_df['RiskLevel'].value_counts()}")

if __name__ == '__main__':
    main()

import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# ======================================================
# CONFIGURACION
# ======================================================

SEED = 42

# Crear carpeta outputs si no existe
os.makedirs("outputs", exist_ok=True)

print("Librerías cargadas correctamente.")

# ======================================================
# CARGAR DATASET
# ======================================================

script_dir = os.path.dirname(os.path.abspath(__file__))
ruta_dataset = os.path.join(
    script_dir,
    "archive (1)",
    "healthcare-dataset-stroke-data.csv"
)

print(f"\nBuscando dataset en: {ruta_dataset}")

if not os.path.exists(ruta_dataset):
    raise FileNotFoundError(
        f"No se encontró el archivo de datos en: {ruta_dataset}\n"
        "Asegúrate de ejecutar el script desde la carpeta del proyecto y de que la carpeta 'archive (1)' exista."
    )


df = pd.read_csv(ruta_dataset)

print("\nDataset cargado correctamente.")
print(df.head())

# ======================================================
# LIMPIEZA DE DATOS
# ======================================================

df_clean = df.copy()

# Eliminar columna id
df_clean.drop(columns=["id"], inplace=True)
print("\nColumna 'id' eliminada.")

# Eliminar registros con gender = Other
n_before = len(df_clean)

df_clean = df_clean[df_clean["gender"] != "Other"]

print(f"Filas eliminadas con gender='Other': {n_before - len(df_clean)}")

# Imputar BMI con mediana
mediana_bmi = df_clean["bmi"].median()

nulos_bmi = df_clean["bmi"].isnull().sum()

df_clean["bmi"] = df_clean["bmi"].fillna(mediana_bmi)

print(f"\nBMI imputado con mediana: {mediana_bmi:.2f}")
print(f"Cantidad de nulos corregidos: {nulos_bmi}")

# Verificacion final
print("\nInformación final del dataset:")
print(df_clean.info())

print("\nNulos restantes:")
print(df_clean.isnull().sum())

# ======================================================
# ENCODING
# ======================================================

df_encoded = df_clean.copy()

# Label Encoding columnas binarias
binary_map = {
    "gender": {"Male": 1, "Female": 0},
    "ever_married": {"Yes": 1, "No": 0},
    "Residence_type": {"Urban": 1, "Rural": 0}
}

for col, mapping in binary_map.items():
    df_encoded[col] = df_encoded[col].map(mapping)
    print(f"\nEncoding aplicado en {col}")

# One Hot Encoding
df_encoded = pd.get_dummies(
    df_encoded,
    columns=["work_type", "smoking_status"],
    drop_first=True
)

print("\nOne-Hot Encoding aplicado.")
print(f"Shape final: {df_encoded.shape}")

# ======================================================
# FEATURES Y TARGET
# ======================================================

X = df_encoded.drop(columns=["stroke"])
y = df_encoded["stroke"]

# ======================================================
# SPLITS
# ======================================================

# Test = 15%
X_temp, X_test, y_temp, y_test = train_test_split(
    X,
    y,
    test_size=0.15,
    random_state=SEED,
    stratify=y
)

# Validation = 15% total
X_train, X_val, y_train, y_val = train_test_split(
    X_temp,
    y_temp,
    test_size=0.1765,
    random_state=SEED,
    stratify=y_temp
)

# ======================================================
# INFORMACION SPLITS
# ======================================================

print("\nDistribución de clases:")

print(f"{'Split':<12} {'N':>8} {'stroke=1':>10} {'% pos':>8}")
print("-" * 45)

for name, ys in [
    ("Train", y_train),
    ("Validation", y_val),
    ("Test", y_test)
]:
    print(
        f"{name:<12} "
        f"{len(ys):>8} "
        f"{ys.sum():>10} "
        f"{ys.mean()*100:>7.2f}%"
    )

# ======================================================
# ESCALADO
# ======================================================

feature_names = X_train.columns.tolist()

scaler = StandardScaler()

X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=feature_names
)

X_val_scaled = pd.DataFrame(
    scaler.transform(X_val),
    columns=feature_names
)

X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=feature_names
)

print("\nScaler aplicado correctamente.")

print(
    f"Media columna age: "
    f"{X_train_scaled['age'].mean():.4f}"
)

print(
    f"STD columna age: "
    f"{X_train_scaled['age'].std():.4f}"
)

# ======================================================
# SMOTE
# ======================================================

smote = SMOTE(random_state=SEED)

X_sm, y_sm = smote.fit_resample(
    X_train_scaled,
    y_train
)

X_train_smote = pd.DataFrame(
    X_sm,
    columns=feature_names
)

y_train_smote = pd.Series(
    y_sm,
    name="stroke"
)

print("\nDistribución antes y después de SMOTE:")

print(
    f"Antes  -> stroke=0: {(y_train == 0).sum()} | "
    f"stroke=1: {(y_train == 1).sum()}"
)

print(
    f"Después -> stroke=0: {(y_train_smote == 0).sum()} | "
    f"stroke=1: {(y_train_smote == 1).sum()}"
)

# ======================================================
# GUARDAR ARCHIVOS
# ======================================================

splits = {
    "X_train": X_train_scaled,
    "X_val": X_val_scaled,
    "X_test": X_test_scaled,
    "X_train_smote": X_train_smote,
    "y_train": y_train.reset_index(drop=True),
    "y_val": y_val.reset_index(drop=True),
    "y_test": y_test.reset_index(drop=True),
    "y_train_smote": y_train_smote
}

for name, data in splits.items():

    ruta_guardado = f"outputs/{name}.csv"

    data.to_csv(
        ruta_guardado,
        index=False
    )

    print(f"Guardado: {ruta_guardado}")

# Guardar scaler
joblib.dump(
    scaler,
    "outputs/scaler.pkl"
)

print("\nScaler guardado correctamente.")

print("\nPREPROCESAMIENTO FINALIZADO.")
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
from pycaret.classification import load_model

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(OUTPUT_DIR, "final_logistic_regression_smote")
X_PATH = os.path.join(OUTPUT_DIR, "X_train_smote.csv")
Y_PATH = os.path.join(OUTPUT_DIR, "y_train_smote.csv")

FEATURES = [
    "age",
    "avg_glucose_level",
    "bmi",
    "hypertension",
    "heart_disease",
    "smoking_status_smokes",
]

print(f"Cargando modelo desde: {MODEL_PATH}")
model = load_model(MODEL_PATH)
print("Modelo cargado correctamente.")

print(f"Leyendo datos desde: {X_PATH}")
X_train = pd.read_csv(X_PATH)
y_train = pd.read_csv(Y_PATH)
print("Datos cargados correctamente.")

def get_estimator(obj):
    # Obtener el estimador subyacente (pycaret wrapper)
    try:
        if hasattr(obj, "named_steps") and "actual_estimator" in obj.named_steps:
            return obj.named_steps["actual_estimator"]
    except Exception:
        pass
    if hasattr(obj, "estimator"):
        return obj.estimator
    return obj

est = get_estimator(model)

if hasattr(est, "coef_"):
    coefs = est.coef_
    if getattr(coefs, "ndim", 1) > 1:
        coefs = coefs[0]
    importance_df = pd.DataFrame({
        "feature": X_train.columns,
        "coef": coefs,
    })
    importance_df["abs_coef"] = importance_df["coef"].abs()
    importance_df = importance_df.sort_values(by="abs_coef", ascending=False)

    importance_path = os.path.join(OUTPUT_DIR, "lr_feature_coefficients.png")
    plt.figure(figsize=(10, 6))
    colors = ["tab:blue" if v >= 0 else "tab:red" for v in importance_df["coef"].iloc[::-1]]
    plt.barh(importance_df["feature"].iloc[::-1], importance_df["coef"].iloc[::-1], color=colors)
    plt.title("Logistic Regression Coefficients")
    plt.xlabel("Coefficient")
    plt.tight_layout()
    plt.savefig(importance_path, dpi=200)
    plt.close()

    print(f"Coeficientes guardados en: {importance_path}")
    print("Top features por magnitud de coeficiente:")
    print(importance_df.head(10).to_string(index=False))
else:
    print("El estimador no tiene coeficientes (coef_). No se pueden mostrar coeficientes de regresión.")

for feature in FEATURES:
    if feature not in X_train.columns:
        print(f"Advertencia: la columna '{feature}' no existe en X_train.")
        continue

    try:
        display = PartialDependenceDisplay.from_estimator(
            model,
            X_train,
            [feature],
            kind="average",
            grid_resolution=50,
        )
        plot_path = os.path.join(OUTPUT_DIR, f"pdp_{feature}.png")
        plt.title(f"Partial Dependence Plot para {feature}")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=200)
        plt.close()
        print(f"PDP guardado en: {plot_path}")
    except Exception as e:
        print(f"No se pudo generar PDP para {feature}: {e}")

print("\nInterpretación completada. Archivos generados en outputs/")

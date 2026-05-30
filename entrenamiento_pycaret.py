# ======================================================
# LIBRERIAS
# ======================================================

import os
import pandas as pd
from pycaret.classification import *

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================================================
# FUNCIONES AUXILIARES
# ======================================================

def load_dataset(x_path, y_path, test_x_path, test_y_path):
    X_train = pd.read_csv(x_path)
    y_train = pd.read_csv(y_path)
    X_test = pd.read_csv(test_x_path)
    y_test = pd.read_csv(test_y_path)

    train_df = X_train.copy().reset_index(drop=True)
    train_df["stroke"] = y_train.reset_index(drop=True)

    test_df = X_test.copy().reset_index(drop=True)
    test_df["stroke"] = y_test.reset_index(drop=True)

    return train_df, test_df


def get_plot_filename(plot_name):
    return {
        "feature": "Feature Importance.png",
        "confusion_matrix": "Confusion Matrix.png",
        "auc": "AUC.png",
        "error": "Error.png",
    }.get(plot_name, f"{plot_name}.png")


def save_pycaret_plot(model, plot_name, prefix):
    try:
        saved = plot_model(model, plot=plot_name, save=True, verbose=False)
        default_name = get_plot_filename(plot_name)
        dest_name = f"{prefix}_{default_name}"
        dest_path = os.path.join(OUTPUT_DIR, dest_name)

        if isinstance(saved, str) and os.path.exists(saved):
            os.replace(saved, dest_path)
        elif os.path.exists(default_name):
            os.replace(default_name, dest_path)
        else:
            print(f"Aviso: no se encontró archivo de salida para plot {plot_name}")
            return None

        print(f"Guardado plot {plot_name} en {dest_path}")
        return dest_path
    except Exception as e:
        print(f"No se pudo guardar el plot {plot_name}: {e}")
        return None


def extract_mean_metrics(results_df):
    if "Mean" in results_df.index:
        mean_row = results_df.loc["Mean"]
    else:
        mean_row = results_df.iloc[-1]
    return {
        "Accuracy": float(mean_row["Accuracy"]),
        "AUC": float(mean_row["AUC"]),
        "Recall": float(mean_row["Recall"]),
        "Prec.": float(mean_row["Prec."]),
        "F1": float(mean_row["F1"]),
        "Kappa": float(mean_row["Kappa"]),
        "MCC": float(mean_row["MCC"]),
    }


def model_key_from_name(class_name):
    text = str(class_name)
    if "Logistic" in text or "lr" in text.lower():
        return "LR"
    if "RandomForest" in text:
        return "RF"
    if "XGB" in text or "XGBoost" in text:
        return "XGBoost"
    return "LR"


def run_experiment(name, train_df, test_df, use_class_weight=False):
    print("\n" + "#" * 80)
    print(f"EXPERIMENTO: {name}")
    print("#" * 80)

    clf = setup(
        data=train_df,
        target="stroke",
        preprocess=False,
        session_id=42,
        test_data=test_df,
        index=False,
        verbose=False
    )

    if use_class_weight:
        positive = int((train_df["stroke"] == 1).sum())
        negative = int((train_df["stroke"] == 0).sum())
        scale_pos_weight = negative / positive if positive > 0 else 1.0

        print(f"\nclass_weight='balanced' activado para LR/RF y scale_pos_weight={scale_pos_weight:.2f} para XGBoost")

        lr = create_model("lr", class_weight="balanced", verbose=False)
        rf = create_model("rf", class_weight="balanced", verbose=False)
        xgb = create_model("xgboost", scale_pos_weight=scale_pos_weight, verbose=False)
    else:
        print("\nSMOTE dataset: modelos entrenados sin class_weight")

        lr = create_model("lr", verbose=False)
        rf = create_model("rf", verbose=False)
        xgb = create_model("xgboost", verbose=False)

    tuned_lr = tune_model(
        lr,
        optimize="F1",
        fold=10,
        choose_better=True,
        verbose=False
    )
    tuned_lr_results = pull().copy()
    print("\nResultados tuning LR:")
    print(tuned_lr_results)

    tuned_rf = tune_model(
        rf,
        optimize="F1",
        fold=10,
        choose_better=True,
        verbose=False
    )
    tuned_rf_results = pull().copy()
    print("\nResultados tuning RF:")
    print(tuned_rf_results)

    tuned_xgb = tune_model(
        xgb,
        optimize="F1",
        fold=10,
        choose_better=True,
        verbose=False
    )
    tuned_xgb_results = pull().copy()
    print("\nResultados tuning XGBoost:")
    print(tuned_xgb_results)

    best = compare_models(
        include=[tuned_lr, tuned_rf, tuned_xgb],
        sort="F1",
        verbose=False
    )

    print("\nMejor modelo en este experimento (reportado por PyCaret):")
    print(best)

    final_lr = finalize_model(tuned_lr)
    final_rf = finalize_model(tuned_rf)
    final_xgb = finalize_model(tuned_xgb)

    suffix = "class_weight" if use_class_weight else "smote"
    prefix = f"{suffix}"

    save_model(final_lr, os.path.join(OUTPUT_DIR, f"final_logistic_regression_{suffix}"))
    save_model(final_rf, os.path.join(OUTPUT_DIR, f"final_random_forest_{suffix}"))
    save_model(final_xgb, os.path.join(OUTPUT_DIR, f"final_xgboost_{suffix}"))

    save_pycaret_plot(best, "auc", prefix)
    save_pycaret_plot(best, "confusion_matrix", prefix)
    save_pycaret_plot(best, "feature", prefix)
    save_pycaret_plot(best, "error", prefix)

    metrics = pd.DataFrame([
        extract_mean_metrics(tuned_lr_results),
        extract_mean_metrics(tuned_rf_results),
        extract_mean_metrics(tuned_xgb_results),
    ], index=["LR", "RF", "XGBoost"])
    metrics.index.name = "Model"

    report_path = os.path.join(OUTPUT_DIR, f"summary_{suffix}.csv")
    metrics.to_csv(report_path)
    print(f"\nResumen de métricas guardado en {report_path}")

    # Devolver métricas y modelos finalizados, sin marcar un "best_model" global
    return {
        "name": name,
        "metrics": metrics,
        "final_lr": final_lr,
        "final_rf": final_rf,
        "final_xgb": final_xgb,
    }


# ======================================================
# CARGAR DATASETS PREPROCESADOS
# ======================================================

train_smote, test_smote = load_dataset(
    "outputs/X_train_smote.csv",
    "outputs/y_train_smote.csv",
    "outputs/X_test.csv",
    "outputs/y_test.csv"
)

train_original, test_original = load_dataset(
    "outputs/X_train.csv",
    "outputs/y_train.csv",
    "outputs/X_test.csv",
    "outputs/y_test.csv"
)

print("Datasets cargados correctamente.")
print(f"SMOTE train shape: {train_smote.shape}")
print(f"Original train shape: {train_original.shape}")

# ======================================================
# EJECUTAR EXPERIMENTOS
# ======================================================

results_smote = run_experiment("SMOTE", train_smote, test_smote, use_class_weight=False)
results_class_weight = run_experiment(
    "class_weight='balanced'",
    train_original,
    test_original,
    use_class_weight=True
)

print("\n" + "=" * 80)
print("RESUMEN DE EXPERIMENTOS")
print("=" * 80)

print("\nSMOTE métricas medias por modelo:")
print(results_smote["metrics"])
print("\nclass_weight métricas medias por modelo:")
print(results_class_weight["metrics"])

print("\nNota: se han guardado los resúmenes de métricas.")
print("- summary_smote.csv")
print("- summary_class_weight.csv")

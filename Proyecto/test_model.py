import os
import pandas as pd
import joblib
from pycaret.classification import load_model, predict_model
import os
import pandas as pd
import joblib
import numpy as np
from pycaret.classification import load_model, predict_model

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
MODEL_PATH = os.path.join(OUTPUT_DIR, "final_logistic_regression_smote")
SCALER_PATH = os.path.join(OUTPUT_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(OUTPUT_DIR, "X_train.csv")

print("Cargando modelo y scaler...")
model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
print("Modelo cargado correctamente.")
print("Scaler cargado correctamente.")

expected_columns = pd.read_csv(FEATURES_PATH, nrows=0).columns.tolist()

# Datos de paciente de ejemplo
patient = pd.DataFrame([{
    "gender": 1,
    "age": 20,
    "hypertension": 0,
    "heart_disease": 0,
    "ever_married": 1,
    "Residence_type": 1,
    "avg_glucose_level": 80,
    "bmi": 22,
    "work_type_Never_worked": 0,
    "work_type_Private": 0,
    "work_type_Self-employed": 1,
    "work_type_children": 0,
    "smoking_status_formerly smoked": 0,
    "smoking_status_never smoked": 0,
    "smoking_status_smokes": 0
}])

missing = [col for col in expected_columns if col not in patient.columns]
extra = [col for col in patient.columns if col not in expected_columns]

if missing:
    raise ValueError(f"Faltan columnas en el paciente de prueba: {missing}")

if extra:
    print(f"Advertencia: columnas extra ignoradas: {extra}")
    patient = patient.drop(columns=extra)

patient = patient[expected_columns]

print("\nCaracterísticas del paciente de prueba:")
print(patient.T)

patient_scaled = pd.DataFrame(scaler.transform(patient), columns=expected_columns)
print("\nPaciente escalado:")
print(patient_scaled.T)

# Resultado PyCaret
resultado = predict_model(model, data=patient_scaled)
print("\nResultado completo:")
print(resultado)

prediction_label = resultado["prediction_label"].iloc[0]
prediction_score = resultado["prediction_score"].iloc[0]

def map_risk(p):
    # basada en la tabla provista
    if p < 0.2:
        return "Riesgo bajo"
    if p < 0.4:
        return "Riesgo moderado"
    if p < 0.6:
        return "Riesgo elevado"
    if p < 0.8:
        return "Riesgo alto"
    return "Riesgo muy alto"

print("\nPredicción (label):", prediction_label)
print("Score PyCaret (probabilidad de la clase predicha):", prediction_score)

# En PyCaret, prediction_score suele ser la probabilidad de la clase que se predice.
# Si la etiqueta predicha es 0, entonces la probabilidad de clase 1 es 1 - prediction_score.
if prediction_label == 1:
    pos_prob = float(prediction_score)
else:
    pos_prob = float(1.0 - prediction_score)

print(f"Probabilidad estimada clase 0: {1.0 - pos_prob:.4f}")
print(f"Probabilidad estimada clase 1 (ACV): {pos_prob:.4f}")
print("Categoría de riesgo:", map_risk(pos_prob))

# Obtener predict_proba del estimador

def get_estimator(obj):
    try:
        if hasattr(obj, "named_steps") and "actual_estimator" in obj.named_steps:
            return obj.named_steps["actual_estimator"]
    except Exception:
        pass
    # try common attributes
    if hasattr(obj, "estimator"):
        return obj.estimator
    return obj

estimator = get_estimator(model)

# Intentar obtener probabilidades directas (predict_proba) del estimador
prob_class1 = None
try:
    # Preferir el estimador subyacente
    if hasattr(estimator, "predict_proba"):
        probs = estimator.predict_proba(patient_scaled)
        try:
            classes = list(getattr(estimator, "classes_", []))
            if 1 in classes:
                idx1 = classes.index(1)
            elif "1" in classes:
                idx1 = classes.index("1")
            else:
                idx1 = 1
        except Exception:
            idx1 = 1
        prob_class1 = float(probs[0][idx1])
    # Fallback: intentar con el objeto cargado por PyCaret
    if prob_class1 is None and hasattr(model, "predict_proba"):
        probs = model.predict_proba(patient_scaled)
        try:
            classes = list(getattr(model, "classes_", []))
            if 1 in classes:
                idx1 = classes.index(1)
            elif "1" in classes:
                idx1 = classes.index("1")
            else:
                idx1 = 1
        except Exception:
            idx1 = 1
        prob_class1 = float(probs[0][idx1])
except Exception as e:
    print("No se pudo obtener predict_proba directamente:", e)

if prob_class1 is not None:
    print(f"\nProbabilidad clase 1 (predict_proba): {prob_class1:.4f}")
    print(f"Probabilidad clase 0 (predict_proba): {1.0 - prob_class1:.4f}")
else:
    print("\npredict_proba no disponible; se usa la estimación previa (pos_prob).")

# === REGLAS CLÍNICAS PARA ACV ===
# Criterios médicos independientes del modelo: edad, hipertensión, cardiopatía, glucosa, BMI, tabaquismo
print("\n" + "="*70)
print("EVALUACIÓN CLÍNICA (independiente del modelo)")
print("="*70)

row = patient.iloc[0]
clinical_score = 0
clinical_factors = []

# Factor 1: Hipertensión
if int(row.get("hypertension", 0)) == 1:
    clinical_score += 2.0
    clinical_factors.append("Hipertensión presente (+2.0)")

# Factor 2: Enfermedad cardíaca
if int(row.get("heart_disease", 0)) == 1:
    clinical_score += 2.0
    clinical_factors.append("Enfermedad del corazón presente (+2.0)")

# Factor 3: Glucosa elevada (diabetes/hiperglucemia)
glucose = float(row.get("avg_glucose_level", 0))
if glucose > 200:
    clinical_score += 1.5
    clinical_factors.append(f"Glucosa muy elevada ({glucose:.0f} mg/dL) (+1.5)")
elif glucose > 126:
    clinical_score += 1.0
    clinical_factors.append(f"Glucosa elevada ({glucose:.0f} mg/dL) (+1.0)")

# Factor 4: BMI (Índice de masa corporal)
bmi = float(row.get("bmi", 0))
if bmi >= 30:
    clinical_score += 1.0
    clinical_factors.append(f"Obesidad (BMI {bmi:.1f}) (+1.0)")
elif bmi >= 25:
    clinical_score += 0.5
    clinical_factors.append(f"Sobrepeso (BMI {bmi:.1f}) (+0.5)")

# Factor 5: Tabaquismo
if int(row.get("smoking_status_smokes", 0)) == 1:
    clinical_score += 1.0
    clinical_factors.append("Fumador activo (+1.0)")
elif int(row.get("smoking_status_formerly smoked", 0)) == 1:
    clinical_score += 0.3
    clinical_factors.append("Exfumador (+0.3)")

# Factor 6: Edad fisiológica (si existe antecedentes de edad avanzada en contexto)
age = float(row.get("age", 0))
if age > 60:
    clinical_score += 1.0
    clinical_factors.append(f"Edad avanzada ({age:.0f} años) (+1.0)")
elif age > 50:
    clinical_score += 0.5
    clinical_factors.append(f"Edad media-alta ({age:.0f} años) (+0.5)")

# Mapeo de score clínico a categoría
def map_clinical_score(score):
    if score >= 4.0:
        return "RIESGO MUY ALTO", 0.8  # Asignar prob equivalente
    elif score >= 2.5:
        return "RIESGO ALTO", 0.65
    elif score >= 1.5:
        return "RIESGO MODERADO", 0.4
    else:
        return "RIESGO BAJO", 0.15

clinical_category, clinical_equiv_prob = map_clinical_score(clinical_score)

print(f"\nPuntaje clínico: {clinical_score:.1f}/6.0+")
print("\nFactores de riesgo detectados:")
if clinical_factors:
    for factor in clinical_factors:
        print(f"  • {factor}")
else:
    print("  • Ninguno detectado")

print(f"\nCategoría clínica: {clinical_category}")
print(f"Probabilidad equivalente (clínica): {clinical_equiv_prob:.4f}")

print("\n" + "="*70)
print("COMPARACIÓN: MODELO ML vs REGLAS CLÍNICAS")
print("="*70)
print(f"Modelo (LR):           {map_risk(pos_prob)} ({pos_prob:.4f})")
print(f"Reglas Clínicas:       {clinical_category} ({clinical_equiv_prob:.4f})")

# === ENSEMBLE HÍBRIDO: Combinar ML + Reglas Clínicas ===
print("\n" + "="*70)
print("DECISIÓN FINAL: ENSEMBLE HÍBRIDO (ML + REGLAS CLÍNICAS)")
print("="*70)

# Probabilidad final: combinación adaptativa según riesgo clínico
# Si el riesgo clínico es muy alto, PRIORIZAR reglas clínicas (80-90%)
# Si el riesgo clínico es moderado, balance 60/40
# Si el riesgo clínico es bajo, balance 30/70 (confiar más en modelo)

if clinical_score >= 4.0:
    # RIESGO CLÍNICO MUY ALTO: 85% clínicas, 15% modelo
    weight_ml = 0.15
    weight_clinical = 0.85
    print(f"\n🚨 RIESGO CLÍNICO MUY ALTO (score {clinical_score:.1f}) → Priorizar reglas clínicas (85%)")
elif clinical_score >= 2.5:
    # RIESGO CLÍNICO MODERADO-ALTO: 65% clínicas, 35% modelo
    weight_ml = 0.35
    weight_clinical = 0.65
    print(f"\n⚠️  RIESGO CLÍNICO MODERADO-ALTO (score {clinical_score:.1f}) → Balance clínicas-modelo (65/35)")
else:
    # RIESGO CLÍNICO BAJO: 40% clínicas, 60% modelo
    weight_ml = 0.6
    weight_clinical = 0.4
    print(f"\nℹ️  RIESGO CLÍNICO BAJO (score {clinical_score:.1f}) → Balance modelo-clínicas (60/40)")

hybrid_prob = (weight_ml * pos_prob) + (weight_clinical * clinical_equiv_prob)

def map_risk_final(p):
    """Mapeo refinado para decisión final"""
    if p < 0.15:
        return "RIESGO BAJO"
    elif p < 0.30:
        return "RIESGO MODERADO"
    elif p < 0.60:
        return "RIESGO ELEVADO"
    elif p < 0.80:
        return "RIESGO ALTO"
    else:
        return "RIESGO MUY ALTO"

final_category = map_risk_final(hybrid_prob)

print(f"\nProbabilidad final (ensemble): {hybrid_prob:.4f}")
print(f"Categoría final: {final_category}")

print(f"\nDesglose del cálculo:")
print(f"  Modelo ML ({weight_ml*100:.0f}%):      {pos_prob:.4f} × {weight_ml} = {weight_ml * pos_prob:.4f}")
print(f"  Clínicas  ({weight_clinical*100:.0f}%):     {clinical_equiv_prob:.4f} × {weight_clinical} = {weight_clinical * clinical_equiv_prob:.4f}")
print(f"  ─────────────────────────────────")
print(f"  Probabilidad final (ensemble):  {hybrid_prob:.4f}")

# Recomendación final
print("\n" + "="*70)
print("RECOMENDACIÓN")
print("="*70)

if hybrid_prob >= 0.80:
    final_recommendation = "🚨 RIESGO MUY ALTO - EVALUACIÓN CLÍNICA URGENTE INMEDIATA"
    action = "- Consulta neurológica inmediata\n- Considerar ingreso hospitalario\n- Realizar estudios de neuroimagen urgentes"
elif hybrid_prob >= 0.60:
    final_recommendation = "🚨 RIESGO ALTO - EVALUACIÓN CLÍNICA URGENTE RECOMENDADA"
    action = "- Consulta neurológica inmediata\n- Considerar ingreso hospitalario\n- Realizar estudios de neuroimagen"
elif hybrid_prob >= 0.30:
    final_recommendation = "⚠️  RIESGO ELEVADO - SEGUIMIENTO CLÍNICO CERCANO RECOMENDADO"
    action = "- Seguimiento médico en 1-2 semanas\n- Optimizar control de factores de riesgo\n- Educar al paciente sobre síntomas de ACV"
elif hybrid_prob >= 0.15:
    final_recommendation = "⚠️  RIESGO MODERADO - SEGUIMIENTO CLÍNICO RECOMENDADO"
    action = "- Seguimiento médico en 2-4 semanas\n- Optimizar factores de riesgo modificables\n- Educación en estilos de vida saludables"
else:
    final_recommendation = "✓ RIESGO BAJO - MONITOREO RUTINARIO"
    action = "- Continuar control rutinario\n- Mantener estilos de vida saludables\n- Revisar anualmente"

print(f"\n{final_recommendation}\n")
print("Acciones recomendadas:")
print(action)

# Recomendación final integrada
if hybrid_prob >= 0.60:
    final_recommendation_alt = "🚨 RIESGO ELEVADO/ALTO - Se recomienda evaluación clínica urgente"
elif hybrid_prob >= 0.30:
    final_recommendation_alt = "⚠️  RIESGO MODERADO - Se recomienda seguimiento clínico"
else:
    final_recommendation_alt = "✓ RIESGO BAJO - Continuar con monitoreo rutinario"

print(f"\nResumen final:")
print(f"- Probabilidad ACV (modelo): {pos_prob:.4f}")
print(f"- Categoría (modelo): {map_risk(pos_prob)}")
print(f"- Puntaje clínico: {clinical_score:.1f}")
print(f"- Categoría clínica: {clinical_category}")
print(f"- Etiqueta (predicción modelo): {prediction_label}")


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# === Cargar datos ===
df = pd.read_csv('data/dataset.csv')
description_df = pd.read_csv('data/symptom_Description.csv')
precaution_df = pd.read_csv('data/symptom_precaution.csv')

# === Limpiar y preparar datos ===
df.fillna('none', inplace=True)
symptom_cols = df.columns[1:]  # Symptom_1, Symptom_2, ..., Symptom_17

# Obtener lista única de síntomas
symptoms_set = set()
for col in symptom_cols:
    symptoms_set.update(df[col].unique())
symptoms_set.discard('none')
symptoms = sorted(symptoms_set)

# Codificar síntomas como vector binario
def encode_symptoms(row):
    present = set(row.values)
    return [1 if s in present else 0 for s in symptoms]

X = df[symptom_cols].apply(encode_symptoms, axis=1, result_type='expand')
X.columns = symptoms
y = df['Disease']

# === Entrenamiento ===
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === Guardar artefactos ===
joblib.dump(model, 'model.pkl')
joblib.dump(symptoms, 'symptoms.pkl')
description_dict = dict(zip(description_df['Disease'], description_df['Description']))
precaution_dict = precaution_df.set_index('Disease').T.to_dict('list')
joblib.dump(description_dict, 'descriptions.pkl')
joblib.dump(precaution_dict, 'precautions.pkl')

print("Modelo entrenado y datos auxiliares guardados.")

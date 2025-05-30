from flask import Flask, render_template, request
import joblib
from fuzzywuzzy import process
import spacy
from googletrans import Translator
import pandas as pd

app = Flask(__name__)

# Cargar modelo y datos
model = joblib.load('model.pkl')
symptoms = joblib.load('symptoms.pkl')
descriptions = joblib.load('descriptions.pkl')
precautions = joblib.load('precautions.pkl')

# Inicializar procesamiento y traductor
nlp = spacy.load("es_core_news_sm")
translator = Translator()

# Diccionario de sinónimos comunes en español
SYMPTOM_ALIASES = {
    "dolor de cabeza": "headache",
    "cefalea": "headache",
    "fiebre": "fever",
    "tos seca": "cough",
    "tos con flema": "cough",
    "congestión nasal": "congestion",
    "nariz tapada": "congestion",
    "dolor muscular": "muscle_pain",
    "náuseas": "nausea",
    "mareo": "dizziness",
    "vómito": "vomiting",
    "diarrea": "diarrhea",
    "fatiga": "fatigue"
}

# Función de traducción
def translate(text, src="en", dest="es"):
    try:
        translated = translator.translate(text, src=src, dest=dest)
        return translated.text
    except Exception as e:
        print(f"Error al traducir: {e}")
        return text

# Traducción de resultados
def translate_output(disease, description, precautions):
    disease_es = translate(disease, src="en", dest="es")
    description_es = translate(description, src="en", dest="es")
    precautions_es = [translate(p, src="en", dest="es") for p in precautions]
    return disease_es, description_es, precautions_es

# Procesar los síntomas desde texto EN INGLÉS
def extract_symptoms_from_text(text_en):
    doc = nlp(text_en.lower())
    lemmas = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    recognized = set()

    # Buscar sinónimos comunes en español → inglés
    for phrase, standard_symptom in SYMPTOM_ALIASES.items():
        if phrase in text_en.lower():
            recognized.add(standard_symptom)

    # Coincidencias aproximadas con los síntomas conocidos
    for lemma in lemmas:
        match, score = process.extractOne(lemma, symptoms)
        if score >= 80:
            recognized.add(match)

    return list(recognized)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    description = None
    precaution_list = None
    recognized_symptoms = []

    if request.method == 'POST':
        user_input_es = request.form['user_input']

        # 1. Traducir el input del usuario al inglés
        user_input_en = translate(user_input_es, src="es", dest="en")

        # 2. Extraer síntomas desde el input en inglés
        recognized_symptoms = extract_symptoms_from_text(user_input_en)

        # 3. Crear vector para el modelo
        input_vector = [1 if s in recognized_symptoms else 0 for s in symptoms]
        input_df = pd.DataFrame([input_vector], columns=symptoms)

        # 4. Predicción
        prediction = model.predict(input_df)[0]
        description = descriptions.get(prediction, "Descripción no disponible.")
        precaution_list = precautions.get(prediction, ["No hay precauciones registradas."])

        # 5. Traducir resultados al español
        prediction, description, precaution_list = translate_output(prediction, description, precaution_list)

    return render_template('chat.html', prediction=prediction, description=description,
                           precautions=precaution_list, recognized=recognized_symptoms)

if __name__ == '__main__':
    app.run(debug=True)

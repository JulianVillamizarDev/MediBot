# 🤖 MediBot - Chatbot de Diagnóstico Médico

MediBot es un asistente médico inteligente que permite a los usuarios describir sus síntomas en español y recibir un diagnóstico preliminar, información clínica y recomendaciones de precaución. Utiliza técnicas de aprendizaje automático y procesamiento de lenguaje natural para analizar los síntomas y ofrecer resultados relevantes.

## 🚀 Características

- Diagnóstico médico basado en síntomas.
- Traducción automática del español al inglés y viceversa usando `googletrans`.
- Visualización de historial de conversación tipo chatbot.
- Predicción usando modelo de árbol de decisión (RandomForest).
- Interfaz web simple desarrollada con Flask y Bootstrap.

## 🗃️ Estructura del Proyecto

```
Medibot/
│
├── app/
│   ├── templates/
│   │   └── index.html      # Interfaz del usuario
│   ├── static/             # Archivos estáticos (CSS, JS, imágenes)
│   ├── app.py              # Lógica principal del chatbot
│   └── utils.py            # Funciones auxiliares (traducción, limpieza)
│
├── model/
│   ├── train_model.py      # Entrenamiento del modelo RandomForest
│   └── model.pkl           # Modelo entrenado serializado
│
├── data/
│   ├── dataset.csv
│   ├── symptom_Description.csv
│   ├── symptom_precaution.csv
│   └── symptom-severity.csv
│
├── requirements.txt        # Dependencias necesarias
└── README.md               # Descripción del proyecto
```

## ⚙️ Instalación

1. Clona el repositorio:

```bash
git clone https://github.com/tu_usuario/medibot.git
cd medibot
```

2. Crea un entorno virtual e instala dependencias:

```bash
python -m venv venv
source venv/bin/activate  # en Windows: venv\Scripts\activate
pip install -r requirements.txts
```

3. Ejecuta la aplicación:

```bash
python app/app.py
```

Abre tu navegador en `http://localhost:5000`.

## 🧠 Modelo de IA

El modelo fue entrenado utilizando `RandomForestClassifier` de Scikit-learn sobre datos estructurados de síntomas y enfermedades. Los datos provienen del conjunto de Kaggle “Disease Symptom Prediction”.

## 🌍 Traducción

MediBot traduce las consultas de los usuarios del español al inglés usando la librería `googletrans`, permitiendo al modelo procesar la información correctamente.

## 👤 Autores

Desarrollado por Julian Villamizar y Diego Granados.

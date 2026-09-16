"""
FASTAPI — API de classification d'images.
Le modèle est entraîné dans le notebook et simplement chargé ici.
"""
from pathlib import Path
import io
import joblib
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException

app = FastAPI(title="AI Vision API", version="1.0.0",
              description="API de classification CAT/DOG.")

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "image_classifier.pkl"
IMAGE_SIZE = (32, 32)

try:
    model = joblib.load(MODEL_PATH)
    MODEL_ERROR = None
except Exception as exc:
    model = None
    MODEL_ERROR = str(exc)

def preprocess(image_bytes: bytes):
    # ÉTAPE : même preprocessing que dans le notebook.
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Fichier image invalide.")
    image = image.resize(IMAGE_SIZE)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return array.reshape(1, -1)

@app.get("/")
def root():
    # ÉTAPE : vérifier que l'API répond.
    return {"message":"AI Vision API is running","docs":"/docs","prediction_endpoint":"POST /predict"}

@app.get("/health")
def health():
    # ÉTAPE : vérifier que le modèle est chargé.
    if model is None:
        return {"status":"error","model_loaded":False,"error":MODEL_ERROR}
    return {"status":"ok","model_loaded":True,"classes":["cat","dog"]}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # ÉTAPE : recevoir l'image, prétraiter, prédire et retourner du JSON.
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Veuillez envoyer une image PNG/JPG/JPEG.")
    data = await file.read()
    X = preprocess(data)
    prediction = model.predict(X)[0]
    probs = model.predict_proba(X)[0]
    result_probs = {str(c): round(float(p),4) for c,p in zip(model.classes_,probs)}
    return {"filename":file.filename,"predicted_class":str(prediction),
            "confidence":round(float(max(probs)),4),"probabilities":result_probs}

# Réaliser par : OUARAS Khelil Rafik
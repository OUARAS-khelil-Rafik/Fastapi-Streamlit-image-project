# AI Vision — FastAPI + Streamlit + Jupyter

## 1. Installer
```bash
python -m venv .venv
source .venv/bin/activate
# Windows : .venv\\Scripts\\activate
pip install -r requirements.txt
pip install --upgrade pip
```

## 2. Entraîner dans Jupyter
```bash
jupyter nbconvert --to notebook --execute --inplace ./notebooks/train_image_classifier.ipynb
```
Ouvrir `notebooks/01_train_image_classifier.ipynb` et exécuter les cellules dans l'ordre.

Le notebook produit `model/image_classifier.pkl`.

## 3. Démarrer FastAPI
```bash
uvicorn app.main:app --reload
```
Swagger : `http://127.0.0.1:8000/docs`

## 4. Démarrer Streamlit (deuxième terminal)
```bash
streamlit run streamlit_app/app.py
```

## 5. Tester
Importer une image dans Streamlit puis cliquer sur **Analyser l'image**.

### Architecture
```text
Jupyter Notebook
      ↓
model/image_classifier.pkl
      ↓
FastAPI /predict
      ↓
Streamlit
```
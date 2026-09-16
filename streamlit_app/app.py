"""
STREAMLIT : Interface.
Streamlit envoie l'image à FastAPI et affiche la réponse.
Il ne charge pas directement le modèle.
"""
import requests
import streamlit as st
from PIL import Image

st.set_page_config(page_title="AI Vision | Cat vs Dog", page_icon="🐾", layout="wide")
API_URL="http://127.0.0.1:8000"

st.markdown("""
<style>
.block-container{padding-top:2rem}
.hero{padding:2rem;border-radius:24px;background:linear-gradient(135deg,#fff,#eef2ff);box-shadow:0 10px 30px rgba(0,0,0,.07);margin-bottom:1.5rem}
.card{padding:1.2rem;border-radius:18px;background:#fff;border:1px solid #e7eaf0}
</style>
""",unsafe_allow_html=True)

with st.sidebar:
    st.title("AI Vision")
    st.caption("Streamlit → FastAPI → Model")
    st.divider()
    st.markdown("### Pipeline")
    st.markdown("Upload → HTTP POST → FastAPI → Preprocessing → Model → JSON → UI")
    try:
        h=requests.get(f"{API_URL}/health",timeout=3).json()
        if h.get("model_loaded"): st.success("API + modèle : OK")
        else: st.warning("API OK, modèle indisponible")
    except requests.RequestException: st.error("FastAPI non démarré")
    st.divider(); st.code(API_URL)

st.markdown("""<div class="hero"><h1>🐾 AI Vision (Cat vs Dog)</h1>
<p>Importez une image puis envoyez-la au modèle via FastAPI.</p></div>""",unsafe_allow_html=True)

file=st.file_uploader("Choisissez une image",type=["png","jpg","jpeg"])
if file:
    image=Image.open(file)
    left,right=st.columns(2)
    with left:
        st.subheader("Image envoyée")
        st.image(image,use_container_width=True)
        st.markdown(f'<div class="card"><b>Format:</b> {image.format or "N/A"}<br><b>Dimensions:</b> {image.size[0]} × {image.size[1]} px<br><b>Type:</b> {file.type}</div>',unsafe_allow_html=True)
    with right:
        st.subheader("Résultat")
        if st.button("Analyser l'image",type="primary",use_container_width=True):
            with st.spinner("Analyse via FastAPI..."):
                try:
                    file.seek(0)
                    r=requests.post(f"{API_URL}/predict",files={"file":(file.name,file,file.type)},timeout=20)
                    if not r.ok:
                        st.error(f"FastAPI : {r.status_code}"); st.code(r.text)
                    else:
                        result=r.json(); label=result["predicted_class"]; conf=result["confidence"]
                        st.success("🐱 CAT" if label=="cat" else "🐶 DOG")
                        st.metric("Classe prédite",label.upper())
                        st.metric("Confiance",f"{conf*100:.2f}%"); st.progress(conf)
                        st.markdown("#### Probabilités")
                        for c,p in result["probabilities"].items():
                            st.write(f"**{c.upper()}** : {p*100:.2f}%"); st.progress(p)
                        with st.expander("Voir le JSON FastAPI"): st.json(result)
                except requests.RequestException as e:
                    st.error("Impossible de contacter FastAPI."); st.code(str(e))
else: st.info("Importez une image pour commencer.")

# Réaliser par : OUARAS Khelil Rafik
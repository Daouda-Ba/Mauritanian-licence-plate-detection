import streamlit as st
from ultralytics import YOLO
from paddleocr import PaddleOCR
import cv2
import numpy as np
import re
import tempfile
import pandas as pd
import os

#  --- Style page ---
st.set_page_config(page_title="🚘 Plate Detection & OCR", layout="wide")
st.markdown("<h1 style='text-align: center; color: #2c3e50;'>📸 License Plate Detection & OCR</h1>", unsafe_allow_html=True)

#  Sidebar settings
with st.sidebar:
    st.title(" Configuration")
    ocr_lang = st.radio("Choisissez la langue OCR :", ("English", "Arabic"))
    st.markdown("---")
    st.info(" Uploadez une image pour lancer la détection", icon="ℹ️")

#  Load models
@st.cache_resource
def load_models(selected_lang):
    yolo = YOLO("best.pt")
    ocr = PaddleOCR(use_angle_cls=True, lang='en' if selected_lang == "English" else 'ar')
    return yolo, ocr

yolo_model, ocr = load_models(ocr_lang)

#  Définir les patterns & régions
series_patterns = [
    ("Série normale", re.compile(r"^\d{4}\s*[A-Z]{2}\s*\d{2,3}$")),
    ("Série diplomatique (CD)", re.compile(r"^[A-Z]{2,3}\s*CD\s*\d{4}$")),
    ("Série diplomatique (CMD)", re.compile(r"^[A-Z]{2,3}\s*CMD\s*\d{4}$")),
    ("Série diplomatique (CC)", re.compile(r"^[A-Z]{2,3}\s*CC\s*\d{4}$")),
    ("Série ONU", re.compile(r"^ONU(?:\s*CMD)?\s*\d{4}$")),
    ("Série ASNA", re.compile(r"^ASNA(?:\s*CMD)?\s*\d{4}$")),
    ("Série TT", re.compile(r"^[A-Z]\s*\d{5}\s*TT(?:\s*ER)?$")),
    ("Série IT", re.compile(r"^\d{4}\s*IT$")),
    ("Série IF", re.compile(r"^\d{4,5}\s*IF$")),
    ("Série Zone Franche", re.compile(r"^ZFN\s*\d{5}$")),
    ("Série Service Gouvernemental", re.compile(r"^SG\s*\d{5}$")),
    ("Service Conseil Constitutionnel", re.compile(r"^SCC\s*\d{5}$")),
    ("Service parlementaire", re.compile(r"^SP\s*\d{5}$")),
    ("Numérotation Tricycle", re.compile(r"^WT\s*\d{5}$")),
    ("Format spécial (LNNN)", re.compile(r"^[A-Z]\d{3}$")),
]

regions_map = {
    "00": "Nouakchott", "01": "Hawd Charki", "02": "Hawd Karbi", "03": "Assaba", "04": "Gorgol",
    "05": "Brakna", "06": "Trarza", "07": "Adrar", "08": "Nouadhibou", "09": "Teganete",
    "10": "Gidimaka", "11": "Tiris Zemmour", "12": "Inchiri"
}

# 📤 Upload image
uploaded_file = st.file_uploader("📷 Choisissez une image :", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    original_img = img.copy()

    st.image(img, caption=" Image originale", use_container_width=True)

    with st.spinner("🔍 Détection en cours avec YOLO..."):
        results = yolo_model.predict(source=img, save=False)
        plates = results[0].boxes.xyxy.cpu().numpy()

    st.success(f"✅ {len(plates)} plaque(s) détectée(s).")

    results_summary = []

    for idx, (x1, y1, x2, y2) in enumerate(plates):
        h, w, _ = img.shape
        pad = 5
        x1p, y1p = max(int(x1) - pad, 0), max(int(y1) - pad, 0)
        x2p, y2p = min(int(x2) + pad, w), min(int(y2) + pad, h)

        plate_crop = img[y1p:y2p, x1p:x2p]
        if plate_crop.size == 0 or x2p <= x1p or y2p <= y1p:
            continue

        if plate_crop.shape[1] < 200:
            scale = 200 / plate_crop.shape[1]
            plate_crop = cv2.resize(plate_crop, (200, int(plate_crop.shape[0] * scale)))

        _, temp_path = tempfile.mkstemp(suffix=".png")
        cv2.imwrite(temp_path, plate_crop)

        try:
            result = ocr.predict(temp_path)
        except Exception as e:
            st.warning(f" OCR échoué pour la plaque #{idx+1} : {e}")
            continue

        page = result[0]
        cv2.rectangle(original_img, (x1p, y1p), (x2p, y2p), (0, 255, 0), 2)
        cv2.putText(original_img, f"#{idx+1}", (x1p, y1p - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        with st.expander(f"🔍 Résultat OCR pour la plaque #{idx+1}"):
            st.image(cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB), caption=f"Plaque #{idx+1}", use_container_width=True)

            for i, raw_text in enumerate(page['rec_texts']):
                text = raw_text.strip().upper().replace(" ", "")
                conf = page['rec_scores'][i]

                serie = "Inconnue"
                for name, pattern in series_patterns:
                    if pattern.match(text):
                        serie = name
                        break

                region = "Inconnue"
                if serie == "Série normale":
                    match = re.search(r"(\d{2})$", text)
                    if match:
                        region = regions_map.get(match.group(1), "Inconnue")

                st.markdown(f"""
                    - **Texte reconnu** : `{raw_text}`
                    - **Confiance OCR** : `{conf:.2f}`
                    - **Série** : `{serie}`
                    - **Région** : `{region}`
                """)

                results_summary.append({
                    "Plaque #": idx + 1,
                    "Texte détecté": raw_text,
                    "Confiance": round(conf, 2),
                    "Série": serie,
                    "Région": region
                })

    st.subheader("Image finale avec détections")
    st.image(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB), use_container_width=True)

    if results_summary:
        st.markdown("### Résumé des plaques détectées")
        df = pd.DataFrame(results_summary)
        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(" Télécharger les résultats en CSV", data=csv, file_name="resultats_plaques.csv", mime="text/csv")
        
# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 50px;">
    <p><strong>Developed by Visionary Minds</strong> – Datathon 2025 ESPDC</p>
</div>
""", unsafe_allow_html=True)
import streamlit as st
from ultralytics import YOLO
from paddleocr import PaddleOCR
import cv2
import numpy as np
import tempfile
import pandas as pd
import os
import re
import time

# === Configuration page ===
st.set_page_config(page_title="🚘 Plate Detection & OCR", layout="wide")
st.markdown("<h1 style='text-align: center; color: #2c3e50;'> Détection Plaques | OCR | Webcam</h1>", unsafe_allow_html=True)

# === Sidebar ===
with st.sidebar:
    st.title(" Configuration")
    app_mode = st.radio("Mode :", [" Vidéo", " Webcam"])
    ocr_lang = st.radio("Langue OCR :", ("English", "Arabic"))
    st.markdown("---")
    st.info("Choisissez un mode et uploadez un vidéo ou utilisez la webcam")

# === Charger les modèles ===
@st.cache_resource
def load_models(lang):
    yolo = YOLO("best.pt")
    ocr = PaddleOCR(use_angle_cls=True, lang='en' if lang == "English" else 'ar')
    return yolo, ocr

yolo_model, ocr = load_models(ocr_lang)

# === Définir les patterns ===
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

# === Détection & OCR ===
def process_frame(frame):
    results = yolo_model.predict(source=frame, save=False)
    plates = results[0].boxes.xyxy.cpu().numpy()
    results_summary = []

    for idx, (x1, y1, x2, y2) in enumerate(plates):
        h, w, _ = frame.shape
        pad = 5
        x1p, y1p = max(int(x1)-pad, 0), max(int(y1)-pad, 0)
        x2p, y2p = min(int(x2)+pad, w), min(int(y2)+pad, h)
        crop = frame[y1p:y2p, x1p:x2p]

        if crop.size == 0 or x2p <= x1p or y2p <= y1p:
            continue

        if crop.shape[1] < 200:
            scale = 200 / crop.shape[1]
            crop = cv2.resize(crop, (200, int(crop.shape[0] * scale)))

        _, tmp_path = tempfile.mkstemp(suffix=".png")
        cv2.imwrite(tmp_path, crop)

        try:
            result = ocr.predict(tmp_path)
            page = result[0]
        except:
            continue

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

            results_summary.append({
                "Texte": raw_text,
                "Confiance": round(conf, 2),
                "Série": serie,
                "Région": region
            })

            cv2.rectangle(frame, (x1p, y1p), (x2p, y2p), (0, 255, 0), 2)
            cv2.putText(frame, text, (x1p, y1p - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return frame, results_summary

# === Mode : Vidéo ===
if app_mode == " Vidéo":
    uploaded_video = st.file_uploader(" Uploadez une vidéo", type=["mp4", "mov", "avi"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        tfile.close()

        cap = cv2.VideoCapture(tfile.name)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_path = os.path.join(tempfile.gettempdir(), "annotated_output.mp4")
        fps = cap.get(cv2.CAP_PROP_FPS)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        summary_all = []

        with st.spinner("Traitement de la vidéo..."):
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % 10 == 0:  # une frame sur 10
                    processed, summary = process_frame(frame)
                    out.write(processed)
                    summary_all.extend(summary)
                    st.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

                frame_count += 1

        cap.release()
        out.release()

        st.success("✅ Traitement terminé !")
        st.video(output_path)

        if summary_all:
            df = pd.DataFrame(summary_all)

            # Supprimer les doublons basés sur la colonne "Texte"
            df_unique = df.drop_duplicates(subset="Texte")

            # Trier par la confiance (colonne "Confiance") en ordre décroissant
            df_unique = df_unique.sort_values(by="Confiance", ascending=False)

            st.markdown("### ✅ Résumé des plaques détectées (uniques)")
            st.dataframe(df_unique)

            # Téléchargement CSV
            csv = df_unique.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Télécharger les plaques détectées (CSV)",
                data=csv,
                file_name="plaques_uniques.csv",
                mime="text/csv"
            )

        with open(output_path, "rb") as file:
            btn = st.download_button(
                label="📥 Télécharger la vidéo annotée",
                data=file,
                file_name="video_annotée.mp4",
                mime="video/mp4"
            )

# === Mode : Webcam ===
elif app_mode == " Webcam":
    st.warning(" L'accès webcam ne fonctionne que localement (Streamlit CLI, pas via navigateur cloud).")
    run = st.checkbox(" Lancer la webcam")

    if run:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()

        while run:
            ret, frame = cap.read()
            if not ret:
                break

            frame, _ = process_frame(frame)
            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

        cap.release()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 50px;">
    <p><strong>Developed by Visionary Minds</strong> – Datathon 2025 ESPDC</p>
</div>
""", unsafe_allow_html=True)
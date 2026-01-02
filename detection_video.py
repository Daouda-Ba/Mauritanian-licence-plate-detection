import streamlit as st
import cv2
import numpy as np
import tempfile
import pandas as pd
import os

from src.config import MIN_CONF_YOLO_DEFAULT, MIN_SCORE_OCR_DEFAULT, OCR_FRAME_GAP_DEFAULT, VideoConfig
from src.models import load_ocr, load_yolo
from src.video_pipeline import VideoProcessor

# === Configuration page ===
st.set_page_config(page_title="🚘 Plate Detection & OCR", layout="wide")
st.markdown("<h1 style='text-align: center; color: #2c3e50;'> Détection Plaques | OCR | Webcam</h1>", unsafe_allow_html=True)

# === Sidebar ===
with st.sidebar:
    st.title(" Configuration")
    app_mode = st.radio("Mode :", [" Vidéo", " Webcam"])
    ocr_lang = st.radio("Langue OCR :", ("English", "Arabic"))
    min_conf_yolo = st.slider("Seuil de confiance YOLO", 0.1, 0.9, float(MIN_CONF_YOLO_DEFAULT), 0.05)
    min_score_ocr = st.slider("Seuil de score OCR", 0.1, 0.9, float(MIN_SCORE_OCR_DEFAULT), 0.05)
    ocr_frame_gap = st.slider("Rafraîchir l'OCR toutes les N frames", 1, 15, OCR_FRAME_GAP_DEFAULT, 1)
    aggressive_preprocess = st.checkbox("Prétraitement agressif", value=False)
    mode_rapide = st.checkbox("Mode rapide (annoter 1 frame sur N)", value=False)
    annotate_every_n = st.slider("Annoter 1 frame sur", 1, 10, 3) if mode_rapide else 1
    st.markdown("---")
    st.info("Choisissez un mode et uploadez un vidéo ou utilisez la webcam")

# === Charger les modèles ===
yolo_model = load_yolo()
ocr_model = load_ocr(ocr_lang)
video_config = VideoConfig(
    min_conf_yolo=min_conf_yolo,
    min_score_ocr=min_score_ocr,
    ocr_frame_gap=ocr_frame_gap,
    aggressive_preprocess=aggressive_preprocess,
    annotate_every_n=annotate_every_n,
)
processor = VideoProcessor(yolo_model, ocr_model, video_config, language=ocr_lang)

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
        with st.spinner("Traitement de la vidéo..."):
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                annotate = True
                if mode_rapide and video_config.annotate_every_n > 1:
                    annotate = frame_count % video_config.annotate_every_n == 0

                processed, _ = processor.process_frame(frame, frame_count, annotate=annotate)
                out.write(processed)

                if frame_count % max(1, int(fps // 2) or 1) == 0:
                    st.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

                frame_count += 1

        cap.release()
        out.release()

        st.success("✅ Traitement terminé !")
        st.video(output_path)

        summary_by_track = processor.summary()

        if summary_by_track:
            df = pd.DataFrame(summary_by_track)
            df_unique = df.drop_duplicates(subset="text").sort_values(by="score", ascending=False)

            st.markdown("### ✅ Résumé des plaques détectées (uniques)")
            st.dataframe(df_unique)

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
        frame_idx = 0

        while run:
            ret, frame = cap.read()
            if not ret:
                break

            annotate = True
            if mode_rapide and video_config.annotate_every_n > 1:
                annotate = frame_idx % video_config.annotate_every_n == 0

            processed, _ = processor.process_frame(frame, frame_idx, annotate=annotate)
            stframe.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            frame_idx += 1

        cap.release()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 50px;">
    <p><strong>Developed by Visionary Minds</strong> – Datathon 2025 ESPDC</p>
</div>
""", unsafe_allow_html=True)

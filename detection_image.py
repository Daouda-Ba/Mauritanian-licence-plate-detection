import streamlit as st
import cv2
import numpy as np
import pandas as pd

from src.config import MIN_CONF_YOLO_DEFAULT, MIN_SCORE_OCR_DEFAULT, TOP_K_DEFAULT, AGGRESSIVE_PREPROCESS_DEFAULT
from src.models import load_ocr, load_yolo
from src.ocr_pipeline import ocr_plate

#  --- Style page ---
st.set_page_config(page_title="🚘 Plate Detection & OCR", layout="wide")
st.markdown("<h1 style='text-align: center; color: #2c3e50;'>📸 License Plate Detection & OCR</h1>", unsafe_allow_html=True)

#  Sidebar settings
with st.sidebar:
    st.title(" Configuration")
    ocr_lang = st.radio("Choisissez la langue OCR :", ("English", "Arabic"))
    min_conf_yolo = st.slider("Seuil de confiance YOLO", 0.1, 0.9, float(MIN_CONF_YOLO_DEFAULT), 0.05)
    min_score_ocr = st.slider("Seuil de score OCR", 0.1, 0.9, float(MIN_SCORE_OCR_DEFAULT), 0.05)
    aggressive_preprocess = st.checkbox("Prétraitement agressif", value=AGGRESSIVE_PREPROCESS_DEFAULT)
    only_top = st.checkbox("OCR seulement sur meilleures plaques (top-k)")
    top_k = st.slider("Top-k détections YOLO", 1, 5, TOP_K_DEFAULT) if only_top else None
    st.markdown("---")
    st.info(" Uploadez une image pour lancer la détection", icon="ℹ️")

#  Load models
yolo_model = load_yolo()
ocr_model = load_ocr(ocr_lang)

# 📤 Upload image
uploaded_file = st.file_uploader("📷 Choisissez une image :", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    original_img = img.copy()

    st.image(img, caption=" Image originale", use_container_width=True)

    with st.spinner("🔍 Détection en cours avec YOLO..."):
        results = yolo_model.predict(source=img, save=False, conf=min_conf_yolo, verbose=False)
        boxes = results[0].boxes

    bboxes = boxes.xyxy.cpu().numpy() if hasattr(boxes, "xyxy") else []
    confs = boxes.conf.cpu().numpy() if hasattr(boxes, "conf") else []
    detections = list(zip(bboxes, confs))
    detections = sorted(detections, key=lambda d: d[1], reverse=True)
    if only_top and top_k:
        detections = detections[:top_k]

    st.success(f"✅ {len(detections)} plaque(s) analysée(s).")

    results_summary = []

    for idx, (bbox, det_conf) in enumerate(detections, start=1):
        x1, y1, x2, y2 = map(int, bbox[:4])
        cv2.rectangle(original_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        ocr_result = ocr_plate(
            img,
            (x1, y1, x2, y2),
            ocr_model,
            language=ocr_lang,
            aggressive_preprocess=aggressive_preprocess,
        )

        # 2) texte seulement si score OCR suffisant
        if ocr_result.get("score", 0) >= min_score_ocr:
            label = f"{ocr_result.get('norm','')} ({ocr_result.get('score',0):.2f})"
            cv2.putText(
                original_img,
                label,
                (x1, max(y1 - 10, 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
        else:
            # optionnel: afficher un petit label "OCR low"
            cv2.putText(
                original_img,
                f"OCR<{min_score_ocr:.2f}",
                (x1, max(y1 - 10, 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
        with st.expander(f"🔍 Résultat OCR pour la plaque #{idx}"):
            st.image(cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2RGB), caption=f"Plaque #{idx}", use_container_width=True)
            st.markdown(
                f"""
                - **Texte brut** : `{ocr_result.get('raw','')}`
                - **Score final** : `{ocr_result.get('score',0):.2f}`
                - **Série** : `{ocr_result.get('serie','')}`
                - **Région** : `{ocr_result.get('region','')}`
                """
            )

        results_summary.append({
            "Plaque #": idx,
            "Texte détecté": ocr_result.get("raw", ""),
            # "Texte normalisé": ocr_result.get("norm", ""),
            "Score": round(ocr_result.get("score", 0), 3),
            "Série": ocr_result.get("serie", "Inconnue"),
            "Région": ocr_result.get("region", "Inconnue"),
            "Confiance YOLO": round(float(det_conf), 3),
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
    <p><strong>Developed by Visionary Minds</strong></p>
</div>
""", unsafe_allow_html=True)

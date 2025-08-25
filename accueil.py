import streamlit as st

st.title("Détection & Lecture de Plaques Mauritaniennes – Datathon 2025 ESPDC")

# Introduction
st.markdown("""
<div style="padding: 20px; border-radius: 10px; margin: 20px 0;">
    <h3>Bienvenue dans notre application de surveillance routière intelligente</h3>
    <p>Ce projet combine YOLOv8 pour détecter les plaques d'immatriculation et PaddleOCR pour en extraire les caractères (français et arabe).</p>
</div>
""", unsafe_allow_html=True)

# Colonnes
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Fonctionnalités principales")

    st.markdown("""
    ** Détection des plaques (YOLOv8)**
    - Analyse d’images ou de vidéos
    - Localisation précise des plaques mauritaniennes

    ** Reconnaissance des caractères (OCR)**
    - Support multilingue : arabe et français
    - Extraction automatique des numéros

    ** Visualisation des résultats**
    - Images annotées
    - Statistiques de détection & lecture
    """)

with col2:
    st.markdown("### Objectif du projet")

    st.info("""
    **Digitalisation des infractions routières**

    Ce projet vise à automatiser l’identification des véhicules pour :
    - Réduire les erreurs humaines
    - Accélérer le traitement des contraventions
    - Créer une base de données fiable des plaques
    """)

    st.success("""
    **Étapes d’utilisation :**
    
    1. Téléversez une image/vidéo contenant une plaque
    2. La plaque est détectée via YOLOv8
    3. Le texte est reconnu avec PaddleOCR
    4. Les résultats sont affichés visuellement
    """)

# Infos techniques
st.markdown("---")
st.markdown("### Technologies utilisées")

col3, col4, col5 = st.columns(3)

with col3:
    st.metric("Détection", "YOLOv8", "Ultralytics")

with col4:
    st.metric("Reconnaissance", "PaddleOCR", "Multilangue")

with col5:
    st.metric("Interface", "Streamlit", "Python")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 50px;">
    <p><strong>Developed by Visionary Minds</strong> – Datathon 2025 ESPDC</p>
</div>
""", unsafe_allow_html=True)
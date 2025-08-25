import streamlit as st

# Navigation
page_0 = st.Page("accueil.py", title="Accueil")       
page_1 = st.Page("visualisation.py", title="Visualisation")     
page_2 = st.Page("detection_image.py", title="Détection sur Image") 
page_3 = st.Page("detection_video.py", title="Détection sur Vidéo/Webcam")

pg = st.navigation([page_0, page_1, page_2, page_3])
pg.run()
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import os
import cv2
import numpy as np
import random
import pandas as pd


st.set_page_config(page_title="Visualisation", layout="wide")

# Configuration générale
choix_visu = st.radio("Que souhaitez-vous visualiser ?", ["Visualisation des données", "Visualisation des métriques"])

if choix_visu == "Visualisation des données":
    st.subheader(" Visualisation du dataset de plaques d'immatriculation")

    import os
    import numpy as np
    import random
    import cv2

    img_dir = "./train/images"
    label_dir = "./train/labels"

    # Prétraitement pour récupérer les infos des bounding boxes
    bbox_widths, bbox_heights, aspect_ratios, centers_x, centers_y = [], [], [], [], []
    n_labels = []

    for label_file in os.listdir(label_dir):
        with open(os.path.join(label_dir, label_file)) as f:
            lines = f.readlines()
            n_labels.append(len(lines))
            for line in lines:
                _, x, y, w, h = map(float, line.strip().split())
                bbox_widths.append(w)
                bbox_heights.append(h)
                centers_x.append(x)
                centers_y.append(y)
                aspect_ratios.append(w / h)

    # Choix du type de visualisation
    visu_type = st.selectbox("Choisissez une visualisation :", [
        "Distribution des dimensions (normalisées)",
        "Heatmap des centres des plaques",
        "Nombre de plaques par image",
        "Carte de densité",
        "Échantillons d'images annotées"
    ])

    # === Distribution des dimensions ===
    if visu_type == "Distribution des dimensions (normalisées)":
        st.markdown("### 📏 Distribution des tailles des plaques")

        # Utilisation de Plotly Express pour les histogrammes
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=bbox_widths, name='Largeur', opacity=0.7, nbinsx=30))
        fig.add_trace(go.Histogram(x=bbox_heights, name='Hauteur', opacity=0.7, nbinsx=30))
        fig.update_layout(
            title="Distribution des tailles normalisées",
            xaxis_title="Taille",
            yaxis_title="Fréquence",
            barmode='overlay'
        )
        st.plotly_chart(fig, use_container_width=True)

        # Histogramme des ratios avec Plotly
        fig2 = px.histogram(x=aspect_ratios, nbins=30, title="Distribution des ratios largeur/hauteur")
        fig2.update_layout(xaxis_title="Ratio (L/H)", yaxis_title="Fréquence")
        st.plotly_chart(fig2, use_container_width=True)

    # === Heatmap des centres ===
    elif visu_type == "Heatmap des centres des plaques":
        st.markdown("### Heatmap des centres (coordonnées normalisées)")

        # Utilisation de Plotly Express pour la heatmap de densité
        fig = px.density_heatmap(
            x=centers_x, 
            y=centers_y,
            title="Répartition des centres des plaques",
            labels={'x': 'Position X', 'y': 'Position Y'}
        )
        fig.update_layout(yaxis=dict(autorange="reversed"))  # Inverser Y
        st.plotly_chart(fig, use_container_width=True)

    # === Nombre de plaques par image ===
    elif visu_type == "Nombre de plaques par image":
        st.markdown("### Nombre de plaques par image")

        # Utilisation de Plotly Express pour l'histogramme
        fig = px.histogram(
            x=n_labels, 
            title="Nombre de plaques par image",
            nbins=max(n_labels)
        )
        fig.update_layout(
            xaxis_title="Plaques",
            yaxis_title="Nombre d'images"
        )
        st.plotly_chart(fig, use_container_width=True)

    # === Carte de densité 3x3 ===
    elif visu_type == "Carte de densité":
        st.markdown("### Carte de densité (3x3)")

        zones = np.zeros((3, 3))
        for x, y in zip(centers_x, centers_y):
            row = min(int(y * 3), 2)
            col = min(int(x * 3), 2)
            zones[row, col] += 1

        # Utilisation de Plotly Express pour la heatmap
        fig = px.imshow(
            zones,
            title="Densité par zone (3x3)",
            color_continuous_scale='Blues',
            aspect='equal'
        )
        fig.update_xaxes(
            tickvals=[0, 1, 2],
            ticktext=['Gauche', 'Centre', 'Droite']
        )
        fig.update_yaxes(
            tickvals=[0, 1, 2],
            ticktext=['Haut', 'Milieu', 'Bas']
        )
        st.plotly_chart(fig, use_container_width=True)

    # === Échantillons annotés ===
    elif visu_type == "Échantillons d'images annotées":
        st.markdown("### Échantillons d'images avec annotations")

        sample_imgs = random.sample(os.listdir(img_dir), 9)
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))

        for ax, img_name in zip(axes.flatten(), sample_imgs):
            img_path = os.path.join(img_dir, img_name)
            lbl_path = os.path.join(label_dir, img_name.replace(".jpg", ".txt").replace(".png", ".txt"))

            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, _ = img.shape

            if os.path.exists(lbl_path):
                with open(lbl_path, "r") as f:
                    for line in f:
                        cls_id, x_center, y_center, bw, bh = map(float, line.strip().split())
                        x_center *= w
                        y_center *= h
                        bw *= w
                        bh *= h
                        x1 = int(x_center - bw / 2)
                        y1 = int(y_center - bh / 2)
                        x2 = int(x_center + bw / 2)
                        y2 = int(y_center + bh / 2)
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)

            ax.imshow(img)
            ax.axis('off')

        plt.tight_layout()
        st.pyplot(fig)


elif choix_visu == "Visualisation des métriques":
    st.subheader("Courbes de performance")
    st.markdown("Ici, vous pouvez visualiser les courbes de perte, de précision, de rappel, etc.")

    # Charger les données
    try:
        df = pd.read_csv("results_with_f1.csv")

        if "epoch" not in df.columns:
            st.error("La colonne 'epoch' est absente du fichier.")
        else:
            df['epoch'] = df['epoch'].astype(int)

            # Menu pour choisir la visualisation
            visu_type = st.selectbox("Choisissez une visualisation :", 
                                    ["Courbes de performance", "Courbes de pertes", 
                                    "Taux d'apprentissage", "Matrice de corrélation"])

            if visu_type == "Courbes de performance":
                st.markdown("### Courbes des métriques de détection")
                
                # Utilisation de Plotly Express pour les courbes
                fig = go.Figure()
                
                metrics = [
                    ("metrics/precision(B)", "Precision"),
                    ("metrics/recall(B)", "Recall"),
                    ("metrics/mAP50(B)", "mAP@0.5"),
                    ("metrics/mAP50-95(B)", "mAP@0.5:0.95")
                ]
                
                for metric_col, metric_name in metrics:
                    if metric_col in df.columns:
                        fig.add_trace(go.Scatter(
                            x=df["epoch"],
                            y=df[metric_col],
                            mode='lines+markers',
                            name=metric_name
                        ))
                
                fig.update_layout(
                    title="Évolution des métriques",
                    xaxis_title="Époques",
                    yaxis_title="Score",
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)

            elif visu_type == "Courbes de pertes":
                st.markdown("### Pertes d'entraînement et de validation")
                
                # Utilisation de Plotly Express pour les courbes de pertes
                fig = go.Figure()
                
                losses = [
                    ("train/box_loss", "Train Box Loss"),
                    ("val/box_loss", "Val Box Loss"),
                    ("train/cls_loss", "Train Class Loss"),
                    ("val/cls_loss", "Val Class Loss")
                ]
                
                for loss_col, loss_name in losses:
                    if loss_col in df.columns:
                        fig.add_trace(go.Scatter(
                            x=df["epoch"],
                            y=df[loss_col],
                            mode='lines+markers',
                            name=loss_name
                        ))
                
                fig.update_layout(
                    title="Pertes par époque",
                    xaxis_title="Époques",
                    yaxis_title="Loss",
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)

            elif visu_type == "Taux d'apprentissage":
                st.markdown("### Taux d'apprentissage")
                
                # Utilisation de Plotly Express pour les courbes de learning rate
                fig = go.Figure()
                
                lr_cols = ["lr/pg0", "lr/pg1", "lr/pg2"]
                
                for lr_col in lr_cols:
                    if lr_col in df.columns:
                        fig.add_trace(go.Scatter(
                            x=df["epoch"],
                            y=df[lr_col],
                            mode='lines+markers',
                            name=lr_col
                        ))
                
                fig.update_layout(
                    title="Learning Rate par époque",
                    xaxis_title="Époques",
                    yaxis_title="Taux d'apprentissage",
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)

            elif visu_type == "Matrice de corrélation":
                st.markdown("### Matrice de corrélation")
                
                # Utilisation de Plotly Express pour la matrice de corrélation
                corr = df.corr(numeric_only=True)
                
                fig = px.imshow(
                    corr,
                    title="Corrélation entre les métriques",
                    color_continuous_scale='RdBu',
                    aspect='auto'
                )
                fig.update_layout(
                    width=800,
                    height=600
                )
                st.plotly_chart(fig, use_container_width=True)
                
    except FileNotFoundError:
        st.error("Le fichier 'results.csv' est introuvable dans le répertoire.")
    except Exception as e:
        st.error(f"Une erreur est survenue : {e}")
        
# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 50px;">
    <p><strong>Developed by Visionary Minds</strong> – Datathon 2025 ESPDC</p>
</div>
""", unsafe_allow_html=True)
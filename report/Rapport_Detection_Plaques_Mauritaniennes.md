RAPPORT TECHNIQUE ET SCIENTIFIQUE
Détection et reconnaissance automatique des plaques d’immatriculation mauritaniennes
YOLOv8 + PaddleOCR + règles métier

Auteur : Équipe Visionary Minds
Date : 2025

TABLE DES MATIÈRES
1. Résumé exécutif
2. Introduction et contexte
3. État de l’art
4. Architecture globale du système
5. Détection des plaques (YOLOv8)
6. Pipeline OCR (PaddleOCR, prétraitement, normalisation, scoring)
7. Règles métier spécifiques aux plaques mauritaniennes
8. Gestion des erreurs, bruit visuel et faux positifs
9. Application Streamlit (UX, fonctionnalités, choix techniques)
10. Résultats expérimentaux et analyses qualitatives
11. Limites du système actuel
12. Perspectives d’amélioration (court / moyen / long terme)
13. Conclusion
14. Annexes techniques

1. RÉSUMÉ EXÉCUTIF
Ce projet propose une chaîne complète de détection et de reconnaissance des plaques
mauritaniennes, intégrant un détecteur YOLOv8, un OCR PaddleOCR multilingue et une
couche métier dédiée à la normalisation et à la validation des formats locaux. La
solution traite des images et des vidéos, fournit des exports CSV, gère un suivi
léger des plaques en vidéo et s’appuie sur une interface Streamlit pensée pour la
validation rapide et la démonstration. L’ensemble démontre un niveau de maturité
supérieur à un prototype académique : modularité claire, paramètres explicites,
choix techniques justifiés et intégration de connaissances réglementaires.

2. INTRODUCTION ET CONTEXTE
L’identification fiable des plaques d’immatriculation est un enjeu majeur pour la
sécurité routière, la lutte contre la fraude et la modernisation des services
publics. En Mauritanie, la diversité des séries (plaques normales, diplomatiques,
services gouvernementaux, zones franches, etc.) et la présence de textes bilingues
complexifient la reconnaissance automatique. Le projet répond à ce contexte par
une architecture unifiée permettant :
- la localisation robuste des plaques dans des conditions variées,
- la reconnaissance du texte en arabe et en français,
- la validation du format et la déduction régionale,
- l’exploitation opérationnelle via une interface interactive.

3. ÉTAT DE L’ART
Les systèmes modernes d’ALPR (Automatic License Plate Recognition) reposent
habituellement sur deux étapes : détection d’objets puis OCR. Les approches
classiques (segmentation par contours, règles fixes) se dégradent fortement sous
variations d’angle, d’éclairage et de bruit. Les détecteurs de type YOLO offrent
un compromis vitesse/précision robuste, tandis que les OCR contemporains (PaddleOCR,
Tesseract moderne, CRNN-CTC) permettent la lecture multi-lingue. Le choix d’un
pipeline YOLOv8 + PaddleOCR s’inscrit donc dans un standard académique et
industriel, tout en offrant une compatibilité immédiate avec un usage temps réel.

4. ARCHITECTURE GLOBALE DU SYSTÈME
Le système est organisé de façon modulaire pour favoriser la maintenance et
l’évolutivité. La logique est structurée autour des composants suivants :
- Configuration centralisée : seuils et paramètres vidéo (src/config.py).
- Chargement des modèles : YOLOv8 et PaddleOCR via cache Streamlit (src/models.py).
- OCR et normalisation : prétraitement, scoring, classification (src/ocr_pipeline.py).
- Règles métier : formats regex et mapping des régions (src/patterns.py).
- Pipeline vidéo : tracking léger, rafraîchissement OCR, agrégation (src/video_pipeline.py).
- Interface utilisateur : pages Streamlit pour image, vidéo, visualisation (detection_image.py,
  detection_video.py, visualisation.py).

Flux logique :
1) Acquisition image/vidéo → 2) Détection YOLO → 3) Crop + prétraitement →
4) OCR PaddleOCR → 5) Normalisation + règles métier → 6) Scoring → 7) Affichage/export.

5. DÉTECTION DES PLAQUES (YOLOv8)
Le projet s’appuie sur YOLOv8 (Ultralytics) avec un poids par défaut best.pt. La
configuration expose un seuil de confiance initial de 0,25, ajustable via UI. Ce
choix est cohérent avec les contraintes temps réel : YOLOv8 combine rapidité,
précision et facilité d’intégration. Les métriques d’entraînement disponibles
indiquent des précisions élevées et des mAP stables dès les premières époques
(ex. precision ~0,95 et mAP@0.5 ~0,90 sur les premières itérations), ce qui
suggère un apprentissage correct sur le dataset d’entraînement. La page de
visualisation exploite ces résultats pour tracer les courbes de performance.

Compromis : le seuil YOLO relativement bas favorise le rappel, tandis que la
sélection top-k côté image limite les faux positifs. En vidéo, l’annotation peut
être sous-échantillonnée (mode rapide), réduisant la charge de calcul sans
perdre la stabilité visuelle grâce au tracking.

6. PIPELINE OCR (PADDLEOCR, PRÉTRAITEMENT, NORMALISATION, SCORING)
Le pipeline OCR suit un principe de simplicité robuste :
- Prétraitement minimal (redimensionnement à largeur minimale) pour préserver la
  structure des caractères et éviter des artefacts d’accentuation.
- Passage à PaddleOCR, avec un choix explicite de langue (anglais ou arabe).
- Extraction des candidats OCR avec scores et boîtes associées.
- Normalisation alphanumérique (suppression d’espaces, homogénéisation) et
  correction ciblée pour les plaques normales.
- Scoring final ajusté selon la conformité à un format mauritanien connu.

Ce design réduit la complexité de prétraitement tout en augmentant la fiabilité
sémantique via la validation métier. Il permet également d’absorber certaines
confusions OCR (lettres/chiffres) grâce à des règles adaptatives.

7. RÈGLES MÉTIER SPÉCIFIQUES AUX PLAQUES MAURITANIENNES
La connaissance locale est un facteur clé du projet. Les séries suivantes sont
prises en compte via regex : série normale, diplomatique (CD, CMD, CC), ONU,
ASNA, TT, IT, IF, Zone Franche, Services Gouvernementaux, Conseil Constitutionnel,
services parlementaires, tricycles, format spécial. Pour les plaques normales,
les deux derniers chiffres permettent de déduire la région administrative
(Nouakchott, Trarza, Adrar, etc.). Cette couche métier transforme un OCR brut
en une information contextualisée exploitable par les services.

8. GESTION DES ERREURS, BRUIT VISUEL ET FAUX POSITIFS
La robustesse face au bruit réel est traitée à plusieurs niveaux :
- seuils configurables pour la détection et l’OCR,
- bonus/malus de score selon la conformité au format,
- rafraîchissement OCR en vidéo pour corriger les erreurs transitoires,
- agrégation par track pour stabiliser les résultats.

Cette approche hybride (vision + règles métier) limite les erreurs typiques de
l’OCR sur des images floues ou partiellement occultées. La décision finale est
donc moins sensible aux variations instantanées.

9. APPLICATION STREAMLIT (UX, FONCTIONNALITÉS, CHOIX TECHNIQUES)
L’interface Streamlit est structurée en quatre pages : accueil, visualisation,
détection sur image, détection sur vidéo/webcam. Les fonctionnalités clés sont :
- réglage des seuils YOLO et OCR,
- sélection de langue OCR,
- mode rapide pour la vidéo,
- export CSV des résultats,
- export vidéo annotée,
- visualisations des distributions et métriques d’entraînement.

Ces choix UX facilitent l’évaluation rapide du système par un jury ou un acteur
industriel, tout en servant de base à une exploitation opérationnelle.

10. RÉSULTATS EXPÉRIMENTAUX ET ANALYSES QUALITATIVES
Les résultats de détection montrent une convergence rapide des métriques
(precision, recall, mAP). En pratique, l’application restitue :
- une image annotée avec les plaques détectées,
- un texte OCR normalisé avec score,
- une classification de série et une estimation régionale.

En vidéo, le tracking léger stabilise les sorties et permet une synthèse par
plaques uniques. Ce comportement est crucial pour des scénarios routiers où une
même plaque est vue sur plusieurs frames.

11. LIMITES DU SYSTÈME ACTUEL
- Prétraitement limité : certaines plaques très bruitées ou obliques restent
  difficiles à lire sans correction géométrique.
- Tracking simple : l’association IoU reste fragile aux occlusions prolongées.
- Langues OCR limitées : les polices ou styles non standard peuvent dégrader
  la reconnaissance.
- Évaluation terrain : absence d’un protocole complet de tests en conditions
  routières variées (jour/nuit, météo, vitesse).

12. PERSPECTIVES D’AMÉLIORATION
Court terme :
- ajouter un mode de prétraitement avancé optionnel (contraste adaptatif,
  correction perspective), déclenché selon la qualité d’image.
- enrichir les règles de normalisation avec des corrections basées sur la série.

Moyen terme :
- remplacer l’IoU par un tracking multi-objet plus robuste (Kalman + association
  Hongroise) pour stabiliser l’identité des plaques.
- introduire un calibrage dynamique des seuils selon l’éclairage et le flou.

Long terme :
- intégration à une plateforme nationale (multi-caméras, base de données,
  traçabilité légale).
- supervision continue du modèle (réentraînement, validation périodique).

13. CONCLUSION
Le projet combine efficacement détection, OCR et intelligence métier. La
séparation des modules, l’intégration Streamlit et la prise en compte des règles
mauritaniennes positionnent cette solution comme un socle sérieux pour un système
industriel. L’architecture actuelle permet une montée en charge progressive,
avec une marge d’amélioration claire vers un usage national.

14. ANNEXES TECHNIQUES
- Paramètres clés : seuils YOLO/OCR, rafraîchissement OCR, tracking IoU.
- Modules principaux : src/ocr_pipeline.py, src/video_pipeline.py, src/patterns.py.
- Outils de validation : visualisation du dataset et courbes de performance.

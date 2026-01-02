# Mauritanian-licence-plate-detection

Pipeline Streamlit pour la détection YOLOv8 et l'OCR (PaddleOCR) des plaques mauritaniennes.

## Git & branches
```bash
git checkout -b upgrade-paddleocr-latest
# travail…
git add .
git commit -m "Upgrade PaddleOCR and refactor OCR/vidéo pipelines"
```

## Installation
### Environnement Python
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### PaddlePaddle / PaddleOCR
- CPU x86_64 : `pip install paddlepaddle==2.6.1` avant PaddleOCR si l'installation par défaut échoue.
- GPU (CUDA 11+) : `pip install paddlepaddle-gpu==2.6.1` (vérifier la compatibilité CUDA/cuDNN).
- Mac M1/M2 : installer via conda-forge (`conda install -c conda-forge paddlepaddle`) puis `pip install paddleocr`.
- En cas d'erreur MKL/MKLDNN, désactiver via variable `OMP_NUM_THREADS=1` ou `export FLAGS_use_mkldnn=false`.

## Lancer l'application
```bash
streamlit run streamlit_app.py
```

- Poids YOLO par défaut : `best.pt` (placé à la racine).
- Pages : accueil, visualisation, détection image, détection vidéo/webcam.
- Export CSV disponible pour image et vidéo.

## Vérifier l'OCR rapidement
```bash
python scripts/check_ocr.py
```
Génère un test synthétique en mémoire pour valider l'import PaddleOCR et le pipeline OCR.

## Structure ajoutée (src/)
- `src/config.py` : seuils par défaut et configuration vidéo.
- `src/patterns.py` : regex des séries et mapping des régions.
- `src/models.py` : chargement YOLO + PaddleOCR (cache Streamlit).
- `src/ocr_pipeline.py` : prétraitement, normalisation, validation regex, scoring.
- `src/video_pipeline.py` : tracking IoU, rafraîchissement OCR, agrégation par track.
- `src/utils.py` : IoU, redimensionnement, dessin des boîtes, helpers.

## Options principales
- Image : sliders `min_conf_yolo`, `min_score_ocr`, prétraitement agressif, OCR top-k.
- Vidéo : mode complet ou rapide (annoter 1 frame sur N), tracking léger, OCR rafraîchi toutes les N frames, export vidéo + CSV.

## Troubleshooting
- Si PaddleOCR télécharge des modèles au premier run, laisser terminer avant d'utiliser l'app.
- Si vous changez de langue OCR, Streamlit recalcule et recharge automatiquement le modèle (cache par langue).
- Pour des performances vidéo fluides, utilisez le mode rapide (moins d'OCR) et limitez la résolution des vidéos d'entrée.

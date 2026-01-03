"""Model loading helpers with Streamlit caching."""

from __future__ import annotations
import streamlit as st
import tempfile
import cv2

from src.config import YOLO_WEIGHTS_DEFAULT


@st.cache_resource(show_spinner=False)
def load_yolo(weights_path: str = YOLO_WEIGHTS_DEFAULT):
    from ultralytics import YOLO
    return YOLO(weights_path)


@st.cache_resource(show_spinner=False)
def load_ocr(lang: str):
    from paddleocr import PaddleOCR

    lang_code = "en" if lang.lower().startswith("en") else "ar"
    return PaddleOCR(
        lang=lang_code,
        use_textline_orientation=True
    )


def run_paddle_ocr(ocr, image):
    """
    PaddleOCR 3.x SAFE MODE:
    always use temp file (same behavior as original working code)
    """
    if image is None or image.size == 0:
        return []

    _, tmp_path = tempfile.mkstemp(suffix=".png")
    cv2.imwrite(tmp_path, image)

    return ocr.predict(tmp_path)
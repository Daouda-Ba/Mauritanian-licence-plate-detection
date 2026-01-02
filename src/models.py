"""Model loading helpers with Streamlit caching."""

from __future__ import annotations

import streamlit as st

from src.config import YOLO_WEIGHTS_DEFAULT


@st.cache_resource(show_spinner=False)
def load_yolo(weights_path: str = YOLO_WEIGHTS_DEFAULT):
    from ultralytics import YOLO

    return YOLO(weights_path)


@st.cache_resource(show_spinner=False)
def load_ocr(lang: str):
    """
    Load PaddleOCR for the selected language.

    Returns a callable OCR object. Language is expected to be "English" or "Arabic".
    """
    from paddleocr import PaddleOCR

    lang_code = "en" if lang.lower().startswith("en") else "ar"
    # Using cls for rotated plates; enable mkldnn where available.
    ocr = PaddleOCR(
        use_angle_cls=True,
        lang=lang_code,
        show_log=False,
    )
    return ocr


def run_paddle_ocr(ocr, image):
    """
    Execute OCR with compatibility across PaddleOCR versions.

    The preferred signature is `ocr.ocr(image, cls=True)` (numpy BGR).
    """
    if hasattr(ocr, "ocr"):
        try:
            return ocr.ocr(image, cls=True)
        except TypeError:
            # Older versions without cls argument already configured via constructor.
            return ocr.ocr(image)
    if hasattr(ocr, "predict"):
        return ocr.predict(image)
    raise AttributeError("OCR object does not expose 'ocr' or 'predict'")


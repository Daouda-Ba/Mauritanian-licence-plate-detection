"""Quick import and OCR sanity check."""

import cv2
import numpy as np

from src.models import load_ocr
from src.ocr_pipeline import ocr_plate


def main():
    print("Loading PaddleOCR (English)...")
    ocr = load_ocr("English")
    dummy = np.full((220, 420, 3), 255, dtype=np.uint8)
    cv2.putText(dummy, "1234 AB 12", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 3)

    bbox = (0, 0, dummy.shape[1], dummy.shape[0])
    result = ocr_plate(dummy, bbox, ocr, language="English", aggressive_preprocess=False)
    print("OCR result:", result)


if __name__ == "__main__":
    main()

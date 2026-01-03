"""OCR pipeline with preprocessing, normalization, validation and scoring."""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.config import MIN_PLATE_WIDTH, PAD_PIXELS
from src.models import run_paddle_ocr
from src.patterns import classify_plate
from src.utils import clamp_bbox, resize_min_width


# -------------------------------------------------------------------
# Preprocessing (MINIMAL – PaddleOCR-friendly)
# -------------------------------------------------------------------

def preprocess_plate(crop: np.ndarray, aggressive: bool = False) -> np.ndarray:
    if crop is None or crop.size == 0:
        return crop
    return resize_min_width(crop, MIN_PLATE_WIDTH)


# -------------------------------------------------------------------
# Mauritanian-specific business rule
# -------------------------------------------------------------------

def fix_mauritanian_normal_plate(text: str) -> str:
    """
    Fix OCR artefacts for Mauritanian normal plates.
    Example: 0392BA008 -> 0392BA00
    """
    match = re.match(r"^(\d{4})([A-Z]{2})(\d{2})\d?$", text)
    if match:
        return f"{match.group(1)}{match.group(2)}{match.group(3)}"
    return text


# -------------------------------------------------------------------
# Normalization
# -------------------------------------------------------------------

def normalize_text(raw_text: str, language: str) -> str:
    if not raw_text:
        return ""

    text = raw_text.strip().upper()
    text = re.sub(r"[\s\-_]", "", text)
    text = re.sub(r"[^\w\d]", "", text)

    if language.lower().startswith("ar"):
        return normalize_arabic(text)

    digits_count = sum(ch.isdigit() for ch in text)
    letters_count = sum(ch.isalpha() for ch in text)

    letters_to_digits = {"O": "0", "I": "1", "L": "1", "S": "5", "Z": "2", "B": "8"}
    digits_to_letters = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B"}

    normalized = []
    for ch in text:
        if ch in letters_to_digits and digits_count >= letters_count + 1:
            normalized.append(letters_to_digits[ch])
        elif ch in digits_to_letters and letters_count > digits_count + 1:
            normalized.append(digits_to_letters[ch])
        else:
            normalized.append(ch)

    return "".join(normalized)


def normalize_arabic(text: str) -> str:
    replacements = {
        "ي": "ی",
        "ئ": "ی",
        "ك": "ک",
        "ؤ": "و",
        "ة": "ه",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


# -------------------------------------------------------------------
# Scoring
# -------------------------------------------------------------------

def score_candidate(confidence: float, matches_pattern: bool) -> float:
    base = confidence
    if matches_pattern:
        return min(1.0, base + 0.15)
    return max(0.0, base - 0.1)


# -------------------------------------------------------------------
# PaddleOCR 3.x output parsing
# -------------------------------------------------------------------

def extract_candidates(ocr_output: list, language: str):
    if not ocr_output:
        return [], None

    page = ocr_output[0]
    texts = page.get("rec_texts", [])
    scores = page.get("rec_scores", [])
    polys = page.get("dt_polys", [])

    candidates = []
    for i, raw_text in enumerate(texts):
        conf = float(scores[i]) if i < len(scores) else 0.0
        #norm = normalize_text(raw_text, language)
        norm = fix_mauritanian_normal_plate(raw_text)
        bbox = polys[i] if i < len(polys) else None

        candidates.append(
            {
                "raw": raw_text,
                "norm": norm,
                "conf": conf,
                "bbox": bbox,
            }
        )

    if not candidates:
        return [], None

    best = max(candidates, key=lambda c: c["conf"])
    return candidates, best


# -------------------------------------------------------------------
# Main OCR pipeline
# -------------------------------------------------------------------

def ocr_plate(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    ocr,
    language: str,
    aggressive_preprocess: bool = False,
) -> Dict:

    x1, y1, x2, y2 = map(int, bbox)
    h, w = frame.shape[:2]
    x1c, y1c, x2c, y2c = clamp_bbox(x1, y1, x2, y2, w, h, pad=PAD_PIXELS)

    crop = frame[y1c:y2c, x1c:x2c]
    if crop.size == 0:
        return {
            "raw": "",
            "norm": "",
            "score": 0.0,
            "serie": "Inconnue",
            "region": "Inconnue",
        }

    crop = preprocess_plate(crop, aggressive_preprocess)

    try:
        ocr_output = run_paddle_ocr(ocr, crop)
    except Exception:
        return {
            "raw": "",
            "norm": "",
            "score": 0.0,
            "serie": "Inconnue",
            "region": "Inconnue",
        }

    candidates, best = extract_candidates(ocr_output, language)
    if not best:
        return {
            "raw": "",
            "norm": "",
            "score": 0.0,
            "serie": "Inconnue",
            "region": "Inconnue",
        }

    serie, region = classify_plate(best["norm"])
    final_score = score_candidate(best["conf"], serie != "Inconnue")

    return {
        "raw": best["raw"],
        "norm": best["norm"],
        "score": final_score,
        "serie": serie,
        "region": region,
    }
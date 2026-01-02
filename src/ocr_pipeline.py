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


def preprocess_plate(crop: np.ndarray, aggressive: bool = False) -> np.ndarray:
    """Apply light preprocessing to improve OCR robustness."""
    if crop is None or crop.size == 0:
        return crop

    plate = crop.copy()
    plate = resize_min_width(plate, MIN_PLATE_WIDTH)
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

    # Light denoising
    gray = cv2.fastNlMeansDenoising(gray, h=10)

    # Adaptive threshold to boost contrast
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 10)

    if aggressive:
        # Sharpen edges
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        thresh = cv2.filter2D(thresh, -1, kernel)
        # Morphological cleanup
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))

    warped = _try_perspective_correction(thresh)
    if warped is not None:
        thresh = warped

    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)


def _try_perspective_correction(gray: np.ndarray) -> Optional[np.ndarray]:
    """Best-effort four-point perspective correction."""
    try:
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) != 4:
            return None
        pts = approx.reshape(4, 2).astype("float32")
        rect = _order_points(pts)
        (tl, tr, br, bl) = rect
        width_a = np.linalg.norm(br - bl)
        width_b = np.linalg.norm(tr - tl)
        max_width = int(max(width_a, width_b))
        height_a = np.linalg.norm(tr - br)
        height_b = np.linalg.norm(tl - bl)
        max_height = int(max(height_a, height_b))
        dst = np.array(
            [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
            dtype="float32",
        )
        matrix = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(gray, matrix, (max_width, max_height))
        return warped
    except Exception:
        return None


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Order points for perspective transform."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def normalize_text(raw_text: str, language: str) -> str:
    """Normalize OCR text with confusion mapping and cleanup."""
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
    """Minimal Arabic normalization for commonly confused characters."""
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


def score_candidate(confidence: float, matches_pattern: bool) -> float:
    """Apply bonus/penalty around OCR confidence."""
    base = confidence
    if matches_pattern:
        return min(1.0, base + 0.15)
    return max(0.0, base - 0.1)


def extract_candidates(ocr_output: List, language: str) -> Tuple[List[Dict], Optional[Dict]]:
    """Parse PaddleOCR output into candidate structures."""
    candidates: List[Dict] = []
    for line in ocr_output or []:
        for det in line:
            try:
                box, (raw_text, conf) = det
            except (ValueError, TypeError):
                continue
            norm_text = normalize_text(raw_text, language)
            candidates.append(
                {"raw": raw_text, "norm": norm_text, "conf": float(conf), "bbox": box},
            )

    if not candidates:
        return [], None

    best = max(candidates, key=lambda c: c["conf"])
    return candidates, best


def ocr_plate(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    ocr,
    language: str,
    aggressive_preprocess: bool = False,
) -> Dict:
    """Run full OCR pipeline on a detected plate crop."""
    x1, y1, x2, y2 = map(int, bbox)
    h, w = frame.shape[:2]
    x1c, y1c, x2c, y2c = clamp_bbox(x1, y1, x2, y2, w, h, pad=PAD_PIXELS)
    crop = frame[y1c:y2c, x1c:x2c]
    if crop.size == 0 or x2c <= x1c or y2c <= y1c:
        return {"raw": "", "norm": "", "score": 0.0, "serie": "Inconnue", "region": "Inconnue", "boxes": [], "all_candidates": []}

    preprocessed = preprocess_plate(crop, aggressive=aggressive_preprocess)

    try:
        ocr_output = run_paddle_ocr(ocr, preprocessed)
    except Exception:
        return {"raw": "", "norm": "", "score": 0.0, "serie": "Inconnue", "region": "Inconnue", "boxes": [], "all_candidates": []}

    candidates, best = extract_candidates(ocr_output, language)
    if not best:
        return {"raw": "", "norm": "", "score": 0.0, "serie": "Inconnue", "region": "Inconnue", "boxes": [], "all_candidates": candidates}

    serie, region = classify_plate(best["norm"])
    final_score = score_candidate(best["conf"], serie != "Inconnue")

    return {
        "raw": best["raw"],
        "norm": best["norm"],
        "score": final_score,
        "serie": serie,
        "region": region,
        "boxes": [best.get("bbox")] if best.get("bbox") is not None else [],
        "all_candidates": candidates,
    }


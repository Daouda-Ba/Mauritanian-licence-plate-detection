"""Shared utilities for detection and video pipelines."""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


def clamp_bbox(x1: int, y1: int, x2: int, y2: int, width: int, height: int, pad: int = 0) -> Tuple[int, int, int, int]:
    """Clamp bounding box within image bounds with optional padding."""
    x1p = max(x1 - pad, 0)
    y1p = max(y1 - pad, 0)
    x2p = min(x2 + pad, width)
    y2p = min(y2 + pad, height)
    return x1p, y1p, x2p, y2p


def resize_min_width(img: np.ndarray, min_width: int) -> np.ndarray:
    """Upscale image to a minimum width while keeping aspect ratio."""
    if img.shape[1] >= min_width:
        return img
    scale = min_width / img.shape[1]
    new_size = (min_width, int(img.shape[0] * scale))
    return cv2.resize(img, new_size, interpolation=cv2.INTER_CUBIC)


def iou(box_a: Iterable[float], box_b: Iterable[float]) -> float:
    """Compute Intersection over Union between two boxes (x1, y1, x2, y2)."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area
    if union == 0:
        return 0.0
    return inter_area / union


def draw_labeled_box(frame: np.ndarray, bbox: Tuple[int, int, int, int], label: str, color: Tuple[int, int, int] = (0, 255, 0)) -> None:
    """Draw a bounding box with a label on the frame."""
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, max(y1 - 10, 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def best_candidate_from_history(history: Dict[str, List[float]]) -> Tuple[Optional[str], float]:
    """
    Return the candidate with the highest mean score.

    history example: {"ABC123": [0.8, 0.9], "DEF456": [0.7]}
    """
    best_text = None
    best_score = 0.0
    for text, scores in history.items():
        if not scores:
            continue
        mean_score = float(np.mean(scores))
        if mean_score > best_score:
            best_score = mean_score
            best_text = text
    return best_text, best_score


def merge_candidate_history(history: Counter, candidate: str, score: float) -> Counter:
    """Update histogram Counter with a weighted count based on score."""
    history[candidate] += score
    return history


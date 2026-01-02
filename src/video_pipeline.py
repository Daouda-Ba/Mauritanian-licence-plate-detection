"""Video/webcam processing with lightweight tracking and OCR throttling."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.ocr_pipeline import ocr_plate
from src.utils import best_candidate_from_history, draw_labeled_box, iou, merge_candidate_history


@dataclass
class TrackState:
    track_id: int
    bbox: Tuple[int, int, int, int]
    last_seen: int
    last_ocr_frame: Optional[int] = None
    candidate_scores: Dict[str, List[float]] = field(default_factory=dict)
    best_text: str = ""
    best_score: float = 0.0
    best_serie: str = "Inconnue"
    best_region: str = "Inconnue"
    history_counter: Counter = field(default_factory=Counter)


class VideoProcessor:
    def __init__(self, yolo_model, ocr_model, config, language: str = "English"):
        self.yolo = yolo_model
        self.ocr = ocr_model
        self.config = config
        self.language = language
        self.tracks: Dict[int, TrackState] = {}
        self._next_track_id = 1
        self.last_annotated_frame: Optional[np.ndarray] = None

    def process_frame(self, frame: np.ndarray, frame_idx: int, annotate: bool = True) -> Tuple[np.ndarray, List[Dict]]:
        """
        Process a frame. When annotate=False, reuse the last annotated frame for speed (Mode B).
        """
        if not annotate and self.last_annotated_frame is not None:
            # Duplicate last annotated frame to keep video fluent
            return self.last_annotated_frame.copy(), []

        annotated = frame.copy()
        results = self.yolo.predict(source=annotated, save=False, conf=self.config.min_conf_yolo, verbose=False)
        detections = self._extract_detections(results[0])

        assignments = self._match_tracks(detections)

        frame_records: List[Dict] = []
        for det_idx, det in enumerate(detections):
            bbox = det["bbox"]
            track_id = assignments.get(det_idx)
            if track_id is None:
                track_id = self._register_track(bbox, frame_idx)

            track = self.tracks[track_id]
            track.bbox = bbox
            track.last_seen = frame_idx

            if self._should_run_ocr(track, frame_idx):
                ocr_result = ocr_plate(
                    annotated,
                    bbox,
                    self.ocr,
                    det["language"],
                    aggressive_preprocess=self.config.aggressive_preprocess,
                )
                track.last_ocr_frame = frame_idx
                self._update_track(track, ocr_result)

            label = f"ID {track_id}"
            if track.best_text:
                label += f" {track.best_text} ({track.best_score:.2f})"
            draw_labeled_box(annotated, bbox, label)

            frame_records.append(
                {
                    "track_id": track_id,
                    "text": track.best_text,
                    "score": track.best_score,
                    "serie": track.best_serie,
                    "region": track.best_region,
                    "bbox": bbox,
                    "det_conf": det["conf"],
                }
            )

        self.last_annotated_frame = annotated.copy()
        return annotated, frame_records

    def _extract_detections(self, result) -> List[Dict]:
        detections: List[Dict] = []
        boxes = result.boxes.xyxy.cpu().numpy() if hasattr(result, "boxes") else []
        confs = result.boxes.conf.cpu().numpy() if hasattr(result, "boxes") else []

        for bbox, conf in zip(boxes, confs):
            x1, y1, x2, y2 = map(int, bbox[:4])
            detections.append({"bbox": (x1, y1, x2, y2), "conf": float(conf), "language": self.language})
        return detections

    def _match_tracks(self, detections: List[Dict]) -> Dict[int, int]:
        assignments: Dict[int, int] = {}
        used_tracks = set()
        for det_idx, det in enumerate(detections):
            best_iou = 0.0
            best_track_id = None
            for track_id, track in self.tracks.items():
                if track_id in used_tracks:
                    continue
                iou_score = iou(det["bbox"], track.bbox)
                if iou_score > self.config.track_iou_threshold and iou_score > best_iou:
                    best_iou = iou_score
                    best_track_id = track_id
            if best_track_id is not None:
                assignments[det_idx] = best_track_id
                used_tracks.add(best_track_id)
        return assignments

    def _register_track(self, bbox: Tuple[int, int, int, int], frame_idx: int) -> int:
        track_id = self._next_track_id
        self._next_track_id += 1
        self.tracks[track_id] = TrackState(track_id=track_id, bbox=bbox, last_seen=frame_idx)
        return track_id

    def _should_run_ocr(self, track: TrackState, frame_idx: int) -> bool:
        if track.last_ocr_frame is None:
            return True
        if track.best_score < self.config.min_score_ocr:
            return True
        if self.config.refresh_low_score and (frame_idx - track.last_ocr_frame) >= self.config.ocr_frame_gap:
            return True
        return False

    def _update_track(self, track: TrackState, ocr_result: Dict) -> None:
        if not ocr_result:
            return
        text = ocr_result.get("norm", "") or ""
        score = float(ocr_result.get("score", 0.0))
        track.candidate_scores.setdefault(text, []).append(score)
        best_text, best_score = best_candidate_from_history(track.candidate_scores)
        track.best_text = best_text or track.best_text
        track.best_score = best_score if best_text else track.best_score
        track.best_serie = ocr_result.get("serie", track.best_serie)
        track.best_region = ocr_result.get("region", track.best_region)
        track.history_counter = merge_candidate_history(track.history_counter, text, score)

    def summary(self) -> List[Dict]:
        """Return aggregated results per track sorted by score."""
        records = []
        for track in self.tracks.values():
            records.append(
                {
                    "track_id": track.track_id,
                    "text": track.best_text,
                    "score": round(track.best_score, 3),
                    "serie": track.best_serie,
                    "region": track.best_region,
                    "votes": round(sum(track.history_counter.values()), 2) if track.history_counter else 0,
                }
            )
        records.sort(key=lambda r: r["score"], reverse=True)
        return records

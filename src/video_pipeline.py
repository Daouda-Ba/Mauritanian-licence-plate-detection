# 

"""Video/webcam processing with lightweight tracking and OCR throttling."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.ocr_pipeline import ocr_plate
from src.utils import best_candidate_from_history, draw_labeled_box, iou, merge_candidate_history
from src.rl.q_agent import QAgent
from src.rl.state import make_state, plate_area_ratio, blur_score, aspect_ratio
from src.rl.reward import reward


# =========================
# RL FEATURE FUNCTIONS
# =========================

# def plate_area_ratio(bbox, frame_shape):
#     x1, y1, x2, y2 = bbox
#     H, W = frame_shape[:2]
#     area_plate = max(0, x2 - x1) * max(0, y2 - y1)
#     area_img = max(1, H * W)
#     return area_plate / area_img


# def blur_score(frame, bbox):
#     x1, y1, x2, y2 = bbox
#     crop = frame[y1:y2, x1:x2]
#     if crop is None or crop.size == 0:
#         return 0.0
#     gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
#     return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# def aspect_ratio(bbox):
#     x1, y1, x2, y2 = bbox
#     w = max(1, x2 - x1)
#     h = max(1, y2 - y1)
#     return w / h


# =========================
# TRACK STATE
# =========================

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
    waited_frames: int = 0
    last_state: Optional[tuple] = None
    last_action: Optional[int] = None


# =========================
# VIDEO PROCESSOR
# =========================

class VideoProcessor:
    def __init__(self, yolo_model, ocr_model, config, language: str = "English"):
        self.yolo = yolo_model
        self.ocr = ocr_model
        self.config = config
        self.language = language
        self.tracks: Dict[int, TrackState] = {}
        self._next_track_id = 1
        self.last_annotated_frame: Optional[np.ndarray] = None
        self.stats = {"ocr_calls": 0, "wait_actions": 0}

        # RL agent
        self.agent = QAgent(alpha=0.2, gamma=0.95, eps=0.2)

    def _extract_detections(self, result) -> List[Dict]:
        detections: List[Dict] = []

        if result is None or not hasattr(result, "boxes") or result.boxes is None:
            return detections

        boxes_xyxy = result.boxes.xyxy
        confs = result.boxes.conf

        if boxes_xyxy is None or confs is None:
            return detections

        boxes = boxes_xyxy.cpu().numpy()
        confs = confs.cpu().numpy()

        for bbox, conf in zip(boxes, confs):
            x1, y1, x2, y2 = map(int, bbox[:4])
            detections.append({
                "bbox": (x1, y1, x2, y2),
                "conf": float(conf),
                "language": self.language
            })

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

        self.tracks[track_id] = TrackState(
            track_id=track_id,
            bbox=bbox,
            last_seen=frame_idx
        )
        return track_id

    def _update_track(self, track: TrackState, ocr_result: Dict) -> None:
        if not ocr_result:
            return

        text = ocr_result.get("norm", "") or ""
        score = float(ocr_result.get("score", 0.0))

        if not text:
            return

        track.candidate_scores.setdefault(text, []).append(score)

        best_text, best_score = best_candidate_from_history(track.candidate_scores)

        if best_text:
            track.best_text = best_text
            track.best_score = best_score
            track.best_serie = ocr_result.get("serie", track.best_serie)
            track.best_region = ocr_result.get("region", track.best_region)

        track.history_counter = merge_candidate_history(
            track.history_counter,
            text,
            score
        )
        
    
    
    def process_frame(self, frame: np.ndarray, frame_idx: int, annotate: bool = True):
        if not annotate and self.last_annotated_frame is not None:
            return self.last_annotated_frame.copy(), []

        annotated = frame.copy()
        results = self.yolo.predict(
            source=annotated,
            save=False,
            conf=self.config.min_conf_yolo,
            verbose=False
        )

        result0 = results[0] if isinstance(results, (list, tuple)) else results
        detections = self._extract_detections(result0)
        # detections = self._extract_detections(results[0])
        assignments = self._match_tracks(detections)

        frame_records = []

        for det_idx, det in enumerate(detections):
            bbox = det["bbox"]
            track_id = assignments.get(det_idx)

            if track_id is None:
                track_id = self._register_track(bbox, frame_idx)

            track = self.tracks[track_id]
            track.bbox = bbox
            track.last_seen = frame_idx

            # =========================
            # RL STATE
            # =========================
            det_conf = det["conf"]
            ar = aspect_ratio(bbox)
            area = plate_area_ratio(bbox, annotated.shape)
            blur = blur_score(annotated, bbox)

            s = make_state(det_conf, area, blur, ar)
            a = self.agent.act(s)

            # =========================
            # RL STATS (VISIBLE RL)
            # =========================
            if a == 1:
                self.stats["ocr_calls"] += 1
            else:
                self.stats["wait_actions"] += 1

            ocr_result = None

            if a == 1:
                ocr_result = ocr_plate(
                    annotated,
                    bbox,
                    self.ocr,
                    det["language"],
                    aggressive_preprocess=self.config.aggressive_preprocess,
                )
                track.last_ocr_frame = frame_idx
                self._update_track(track, ocr_result)
                track.waited_frames = 0
            else:
                track.waited_frames += 1

            r = reward(a, ocr_result, s, track.waited_frames)
            self.agent.update(s, a, r, s)

            track.last_state = s
            track.last_action = a

            decision = "OCR" if a == 1 else "WAIT"
            label = ""
            if track.best_text:
                label = f"{track.best_text} ({track.best_score:.2f}) [{decision}]"
            else:
                label = f"[{decision}]"

                
            draw_labeled_box(annotated, bbox, label)

            frame_records.append({
                "track_id": track_id,
                "text": track.best_text,
                "score": track.best_score,
                "serie": track.best_serie,
                "region": track.best_region,
                "bbox": bbox,
                "det_conf": det_conf,
            })

        self.last_annotated_frame = annotated.copy()
        return annotated, frame_records
    
    def summary(self) -> List[Dict]:
        """Return aggregated results per track sorted by score."""
        records = []

        for track in self.tracks.values():
            if not track.best_text:
                continue  # ignorer les plaques jamais lues

            records.append(
                {
                    "track_id": track.track_id,
                    "text": track.best_text,
                    "score": round(track.best_score, 3),
                    "serie": track.best_serie,
                    "region": track.best_region,
                    "votes": round(sum(track.history_counter.values()), 2)
                    if track.history_counter else 0,
                }
            )

        records.sort(key=lambda r: r["score"], reverse=True)
        return records

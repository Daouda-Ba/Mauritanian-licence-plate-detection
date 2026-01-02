"""Central configuration values for detection and OCR pipelines."""

from dataclasses import dataclass


YOLO_WEIGHTS_DEFAULT = "best.pt"
MIN_CONF_YOLO_DEFAULT = 0.25
MIN_SCORE_OCR_DEFAULT = 0.5
OCR_FRAME_GAP_DEFAULT = 5
TOP_K_DEFAULT = 3
TRACK_IOU_THRESHOLD = 0.3
AGGRESSIVE_PREPROCESS_DEFAULT = False
PAD_PIXELS = 8
MIN_PLATE_WIDTH = 220


@dataclass
class VideoConfig:
    """Runtime configuration for the video/webcam pipeline."""

    min_conf_yolo: float = MIN_CONF_YOLO_DEFAULT
    min_score_ocr: float = MIN_SCORE_OCR_DEFAULT
    ocr_frame_gap: int = OCR_FRAME_GAP_DEFAULT
    aggressive_preprocess: bool = AGGRESSIVE_PREPROCESS_DEFAULT
    track_iou_threshold: float = TRACK_IOU_THRESHOLD
    annotate_every_n: int = 1  # Mode A: annotate every frame; Mode B: >1
    refresh_low_score: bool = True


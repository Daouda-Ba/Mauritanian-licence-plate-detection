import cv2

def bin3(x, t1, t2):
    if x < t1:
        return 0
    if x < t2:
        return 1
    return 2

def plate_area_ratio(bbox, frame_shape):
    x1, y1, x2, y2 = bbox
    H, W = frame_shape[:2]
    area_plate = max(0, x2 - x1) * max(0, y2 - y1)
    area_img = (H * W) if H > 0 and W > 0 else 1
    return area_plate / area_img

def blur_score(frame, bbox):
    x1, y1, x2, y2 = bbox
    crop = frame[y1:y2, x1:x2]
    if crop is None or crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def aspect_ratio(bbox):
    x1, y1, x2, y2 = bbox
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    return w / h

def make_state(det_conf, area_ratio, blur, ar):
    c = bin3(det_conf, 0.35, 0.60)
    a = bin3(area_ratio, 0.01, 0.03)
    b = bin3(blur, 60.0, 150.0)
    r = bin3(ar, 2.2, 3.5)
    return (c, a, b, r)

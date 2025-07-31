import cv2
import numpy as np
from PIL.Image import Image


def aadi_opencv_week5(am_img, pm_img, MIN_AREA=150, ANGLE_THRESHOLD=8):
    if isinstance(am_img, Image):
        am_img = cv2.cvtColor(np.array(am_img), cv2.COLOR_RGB2BGR)
    if isinstance(pm_img, Image):
        pm_img = cv2.cvtColor(np.array(pm_img), cv2.COLOR_RGB2BGR)

    def preprocess(region):
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8)).apply(blur)
        closed = cv2.morphologyEx(clahe, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2)
        return cv2.resize(closed, (1024, 1024))

    def color_mask(hsv, color):
        if color == "red":
            bounds = [
                (np.array([0, 120, 70]), np.array([10, 255, 255])),
                (np.array([160, 120, 70]), np.array([180, 255, 255]))
            ]
        elif color == "green":
            bounds = [(np.array([35, 70, 50]), np.array([85, 255, 255]))]
        else:
            raise ValueError("Unsupported color")

        mask = sum((cv2.inRange(hsv, low, high) for low, high in bounds), np.zeros_like(hsv[:, :, 0]))
        return cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)

    def extract_combined_region(image, hsv, color):
        mask = color_mask(hsv, color)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = image.shape[:2]
        mid_x = w // 2
        combined = {"left": np.zeros_like(image), "right": np.zeros_like(image)}

        for cnt in contours:
            if cv2.contourArea(cnt) < MIN_AREA:
                continue
            x, _, _, _ = cv2.boundingRect(cnt)
            mask_single = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask_single, [cnt], -1, 255, -1)
            region = cv2.bitwise_and(image, image, mask=mask_single)
            side = "left" if x < mid_x else "right"
            combined[side] = cv2.bitwise_or(combined[side], region)

        return {side: preprocess(combined[side]) for side in ["left", "right"]}

    def compute_features(img):
        orb = cv2.ORB_create(nfeatures=5000, scaleFactor=1.2, nlevels=8, edgeThreshold=15,
                             patchSize=31, WTA_K=2, scoreType=cv2.ORB_FAST_SCORE)
        return orb.detectAndCompute(img, None)

    def angle_between(p1a, p1b, p2a, p2b):
        v1, v2 = np.array(p1b) - p1a, np.array(p2b) - p2a
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norm == 0:
            return 180
        cosine = np.clip(np.dot(v1, v2) / norm, -1.0, 1.0)
        return np.degrees(np.arccos(cosine))

    def match_score(kp1, des1, kp2, des2):
        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            return None
        matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(des1, des2)
        if len(matches) < 2:
            return None

        filtered = [
            m1 for m1, m2 in zip(matches[:-1], matches[1:])
            if angle_between(kp1[m1.queryIdx].pt, kp1[m2.queryIdx].pt,
                             kp2[m1.trainIdx].pt, kp2[m2.trainIdx].pt) <= ANGLE_THRESHOLD
        ]

        if len(filtered) < 4:
            return None

        avg_dist = np.mean([m.distance for m in filtered])
        return avg_dist / (len(filtered) + 1e-6)

    # --- Main processing ---
    hsv_am = cv2.cvtColor(am_img, cv2.COLOR_BGR2HSV)
    hsv_pm = cv2.cvtColor(pm_img, cv2.COLOR_BGR2HSV)

    am_feats = {}
    pm_feats = {}

    for color in ["red", "green"]:
        am_regions = extract_combined_region(am_img, hsv_am, color)
        pm_regions = extract_combined_region(pm_img, hsv_pm, color)

        for side in ["left", "right"]:
            key = f"{color}-{side}"
            am_feats[key] = (am_regions[side], *compute_features(am_regions[side]))
            pm_feats[key] = (pm_regions[side], *compute_features(pm_regions[side]))

    total_score = 0
    valid_parts = 0

    for key in ["red-left", "red-right", "green-left", "green-right"]:
        pm_img_r, pm_kp, pm_des = pm_feats[key]
        am_img_r, am_kp, am_des = am_feats[key]
        score = match_score(pm_kp, pm_des, am_kp, am_des)
        if score is not None:
            similarity = 1.0 / (1.0 + score)
            total_score += similarity * 100
            valid_parts += 1

    return (total_score / valid_parts) if valid_parts else 0.0

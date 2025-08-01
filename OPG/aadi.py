import cv2
import numpy as np
from PIL.Image import Image


def aadi_opencv_week5(am_img, pm_img, MIN_AREA=150, ANGLE_THRESHOLD=8):
    if isinstance(am_img, Image):
        am_img = cv2.cvtColor(np.array(am_img), cv2.COLOR_RGB2BGR)
    if isinstance(pm_img, Image):
        pm_img = cv2.cvtColor(np.array(pm_img), cv2.COLOR_RGB2BGR)

    def preprocess_region(region):
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8)).apply(blur)
        closed = cv2.morphologyEx(clahe, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2)
        return cv2.resize(closed, (1024, 1024))

    def extract_combined_region(image, hsv, color_name, min_area):
        if color_name == "red":
            lower = [np.array([0, 120, 70]), np.array([160, 120, 70])]
            upper = [np.array([10, 255, 255]), np.array([180, 255, 255])]
        elif color_name == "green":
            lower = [np.array([35, 70, 50])]
            upper = [np.array([85, 255, 255])]
        else:
            raise ValueError("Unsupported color")

        mask = cv2.inRange(hsv, lower[0], upper[0])
        for i in range(1, len(lower)):
            mask |= cv2.inRange(hsv, lower[i], upper[i])
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = image.shape[:2]
        mid_x = w // 2

        combined_left = np.zeros_like(image)
        combined_right = np.zeros_like(image)

        for cnt in contours:
            if cv2.contourArea(cnt) >= min_area:
                x, _, _, _ = cv2.boundingRect(cnt)
                mask_single = np.zeros(image.shape[:2], dtype=np.uint8)
                cv2.drawContours(mask_single, [cnt], -1, 255, -1)
                region = cv2.bitwise_and(image, image, mask=mask_single)
                if x < mid_x:
                    combined_left = cv2.bitwise_or(combined_left, region)
                else:
                    combined_right = cv2.bitwise_or(combined_right, region)

        return {
            "left": preprocess_region(combined_left),
            "right": preprocess_region(combined_right)
        }

    def compute_features(image):
        orb = cv2.ORB_create(
            nfeatures=5000,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=15,
            patchSize=31,
            WTA_K=2,
            scoreType=cv2.ORB_FAST_SCORE
        )
        kp, des = orb.detectAndCompute(image, None)
        return kp, des

    def angle_between_vectors(p1a, p1b, p2a, p2b):
        v1 = np.array(p1b) - np.array(p1a)
        v2 = np.array(p2b) - np.array(p2a)
        norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norm_product == 0:
            return 180
        cosine_angle = np.clip(np.dot(v1, v2) / norm_product, -1.0, 1.0)
        return np.degrees(np.arccos(cosine_angle))

    def angle_match_score(kp1, des1, kp2, des2, angle_thresh=ANGLE_THRESHOLD):
        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            return None
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        if len(matches) < 2:
            return None

        filtered_matches = []
        for i in range(len(matches) - 1):
            m1, m2 = matches[i], matches[i + 1]
            p1a, p1b = kp1[m1.queryIdx].pt, kp1[m2.queryIdx].pt
            p2a, p2b = kp2[m1.trainIdx].pt, kp2[m2.trainIdx].pt
            angle = angle_between_vectors(p1a, p1b, p2a, p2b)
            if angle <= angle_thresh:
                filtered_matches.append(m1)

        if len(filtered_matches) < 4:
            return None

        avg_dist = sum(m.distance for m in filtered_matches) / len(filtered_matches)
        norm_score = avg_dist / (len(filtered_matches) + 1e-6)
        return norm_score, filtered_matches, None

    # --- Extract and compare regions ---
    hsv_am = cv2.cvtColor(am_img, cv2.COLOR_BGR2HSV)
    hsv_pm = cv2.cvtColor(pm_img, cv2.COLOR_BGR2HSV)

    am_regions = {}
    pm_regions = {}

    for color in ["red", "green"]:
        am_extracted = extract_combined_region(am_img, hsv_am, color, MIN_AREA)
        pm_extracted = extract_combined_region(pm_img, hsv_pm, color, MIN_AREA)

        for side in ["left", "right"]:
            key = f"{color}-{side}"
            am_img_proc = am_extracted[side]
            pm_img_proc = pm_extracted[side]
            am_regions[key] = (am_img_proc, *compute_features(am_img_proc))
            pm_regions[key] = (pm_img_proc, *compute_features(pm_img_proc))

    total_score = 0
    valid_parts = 0
    for region_key in ["red-left", "red-right", "green-left", "green-right"]:
        pm_img_r, pm_kp, pm_des = pm_regions[region_key]
        am_img_r, am_kp, am_des = am_regions[region_key]
        result = angle_match_score(pm_kp, pm_des, am_kp, am_des)
        if result is not None:
            score, _, _ = result
            similarity = 1.0 / (1.0 + score)  # Normalized similarity in (0, 1]
            total_score += similarity * 100   # Convert to percentage
            valid_parts += 1


    return (total_score / valid_parts) if valid_parts > 0 else 0.0
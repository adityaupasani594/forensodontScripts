import os
import cv2
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity


def resize_image(img, max_dim=800):
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        return cv2.resize(img, (int(w * scale), int(h * scale)))
    return img


def preprocess_chain(image_path):
    if not isinstance(image_path, str):
        raise ValueError(f"Invalid path type: {type(image_path)} - value: {image_path}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File not found: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"cv2.imread() failed to load image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    blurred = cv2.GaussianBlur(clahe, (5, 5), 0)
    return cv2.fastNlMeansDenoising(blurred, None, h=10, templateWindowSize=7, searchWindowSize=21)


def extract_toothprint_features(img):
    features = [img.shape[1] / img.shape[0]]  # aspect ratio
    moments = cv2.moments(img)
    angle = 0
    if moments['mu02'] != 0:
        angle = 0.5 * np.arctan2(2 * moments['mu11'], moments['mu20'] - moments['mu02'])
    features.append(angle)
    features.extend(cv2.HuMoments(moments).flatten())
    return np.array(features)


def sarvankar(am_input, pm_input):
    def prepare_input(img_input):
        if isinstance(img_input, Image.Image):
            return cv2.cvtColor(np.array(img_input), cv2.COLOR_RGB2BGR)
        if isinstance(img_input, str):
            return preprocess_chain(img_input)
        if isinstance(img_input, np.ndarray):
            return img_input
        raise ValueError(f"Unsupported input type: {type(img_input)}")

    am_img = resize_image(prepare_input(am_input))
    pm_img = resize_image(prepare_input(pm_input))

    if am_img is None or pm_img is None:
        return 0.0

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(am_img, None)
    kp2, des2 = sift.detectAndCompute(pm_img, None)

    if des1 is None or des2 is None:
        return 0.0

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = [
        m for m_n in matches if len(m_n) == 2 for m, n in [m_n] if m.distance < 0.75 * n.distance
    ]

    global_score = len(good_matches) / len(matches) * 100 if matches else 0.0

    # --- Patch Refinement (not used in score, but processed) ---
    patch_size = 64
    for match in good_matches[:10]:
        x1, y1 = map(int, kp1[match.queryIdx].pt)
        x2, y2 = map(int, kp2[match.trainIdx].pt)

        roi1 = am_img[max(0, y1 - patch_size // 2): y1 + patch_size // 2,
                      max(0, x1 - patch_size // 2): x1 + patch_size // 2]
        roi2 = pm_img[max(0, y2 - patch_size // 2): y2 + patch_size // 2,
                      max(0, x2 - patch_size // 2): x2 + patch_size // 2]

        if roi1.shape[:2] != (patch_size, patch_size) or roi2.shape[:2] != (patch_size, patch_size):
            continue

        des_roi1 = sift.detectAndCompute(roi1, None)[1]
        des_roi2 = sift.detectAndCompute(roi2, None)[1]

        if des_roi1 is None or des_roi2 is None:
            continue

        matches_roi = bf.knnMatch(des_roi1, des_roi2, k=2)
        _ = [m for m_n in matches_roi if len(m_n) == 2 for m, n in [m_n] if m.distance < 0.75 * n.distance]

        vec1 = extract_toothprint_features(roi1)
        vec2 = extract_toothprint_features(roi2)
        _ = cosine_similarity([vec1], [vec2])[0][0]

    # --- Normalize ---
    normalized_score = np.clip((global_score - 0) / (15 - 0) * 100, 0.0, 100.0) if global_score else 0.0
    return normalized_score

import os
import cv2
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity


def resize_image(img, max_dim=800):
    if max(img.shape[:2]) > max_dim:
        scale = max_dim / max(img.shape[:2])
        return cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))
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
    denoised = cv2.fastNlMeansDenoising(blurred, None, h=10, templateWindowSize=7, searchWindowSize=21)
    return denoised


def extract_toothprint_features(img):
    features = []
    h, w = img.shape
    features.append(w / h)  # aspect ratio

    moments = cv2.moments(img)
    angle = 0
    if moments['mu02'] != 0:
        angle = 0.5 * np.arctan2(2 * moments['mu11'], moments['mu20'] - moments['mu02'])
    features.append(angle)

    hu = cv2.HuMoments(moments).flatten()
    features.extend(hu.tolist())

    return np.array(features)


def sarvankar(am_input, pm_input):
    # === Convert PIL.Image to np.ndarray ===
    if isinstance(am_input, Image.Image):
        am_input = cv2.cvtColor(np.array(am_input), cv2.COLOR_RGB2BGR)
    if isinstance(pm_input, Image.Image):
        pm_input = cv2.cvtColor(np.array(pm_input), cv2.COLOR_RGB2BGR)

    # === Load and preprocess ===
    if isinstance(am_input, str):
        am_img = preprocess_chain(am_input)
    elif isinstance(am_input, np.ndarray):
        am_img = am_input
    else:
        raise ValueError(f"Unsupported input type for AM image: {type(am_input)}")

    if isinstance(pm_input, str):
        pm_img = preprocess_chain(pm_input)
    elif isinstance(pm_input, np.ndarray):
        pm_img = pm_input
    else:
        raise ValueError(f"Unsupported input type for PM image: {type(pm_input)}")

    if am_img is None or pm_img is None:
        return 0.0

    am_img = resize_image(am_img)
    pm_img = resize_image(pm_img)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(am_img, None)
    kp2, des2 = sift.detectAndCompute(pm_img, None)

    if des1 is None or des2 is None:
        return 0.0

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = [m for m_n in matches if len(m_n) == 2 for m, n in [m_n] if m.distance < 0.75 * n.distance]

    global_score = len(good) / len(matches) * 100 if matches else 0.0

    # === Patch Refinement ===
    patch_size = 64
    for match in good[:10]:  # Evaluate first 10 good matches
        pt1 = tuple(map(int, kp1[match.queryIdx].pt))
        pt2 = tuple(map(int, kp2[match.trainIdx].pt))
        x1, y1 = pt1
        x2, y2 = pt2

        roi1 = am_img[max(0, y1 - patch_size // 2): y1 + patch_size // 2,
                      max(0, x1 - patch_size // 2): x1 + patch_size // 2]
        roi2 = pm_img[max(0, y2 - patch_size // 2): y2 + patch_size // 2,
                      max(0, x2 - patch_size // 2): x2 + patch_size // 2]

        if roi1.shape != (patch_size, patch_size) or roi2.shape != (patch_size, patch_size):
            continue

        kp_roi1, des_roi1 = sift.detectAndCompute(roi1, None)
        kp_roi2, des_roi2 = sift.detectAndCompute(roi2, None)

        if des_roi1 is None or des_roi2 is None:
            continue

        matches_roi = bf.knnMatch(des_roi1, des_roi2, k=2)
        good_roi = [m for m_n in matches_roi if len(m_n) == 2 for m, n in [m_n] if m.distance < 0.75 * n.distance]
        _ = len(good_roi) / len(matches_roi) * 100 if matches_roi else 0

        vec1 = extract_toothprint_features(roi1)
        vec2 = extract_toothprint_features(roi2)
        _ = cosine_similarity([vec1], [vec2])[0][0]

    # === Normalize the global_score using given range ===
    min_score = 0
    max_score = 15
    if max_score - min_score == 0:
        normalized_score = 0.0
    else:
        normalized_score = (global_score - min_score) / (max_score - min_score) * 100
        normalized_score = max(0.0, min(100.0, normalized_score))  # Clamp to 0–100

    return normalized_score
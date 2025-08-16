import cv2
import os
import pickle
import numpy as np

# ===============================================================
# Soham Feature Extraction + Matching (with pickle caching)
# ===============================================================

def extract_features(image_path):
    """Extracts SIFT keypoints and descriptors from an image."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"[ERROR] Could not read image: {image_path}")
        return None

    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)
    return (keypoints, descriptors)


def save_features(image_path, feature_dir="cache_soham"):
    """Extracts and caches features as pickle file."""
    os.makedirs(feature_dir, exist_ok=True)
    base = os.path.basename(image_path)
    pickle_path = os.path.join(feature_dir, base + "_soham.pkl")

    if os.path.exists(pickle_path):
        with open(pickle_path, "rb") as f:
            return pickle.load(f)

    features = extract_features(image_path)
    with open(pickle_path, "wb") as f:
        pickle.dump(features, f)

    return features


def load_features(image_path, feature_dir="cache_soham"):
    """Loads cached features if they exist, otherwise extracts them."""
    os.makedirs(feature_dir, exist_ok=True)
    base = os.path.basename(image_path)
    pickle_path = os.path.join(feature_dir, base + "_soham.pkl")

    if os.path.exists(pickle_path):
        with open(pickle_path, "rb") as f:
            return pickle.load(f)
    else:
        return save_features(image_path, feature_dir)


def match_images(img1_path, img2_path):
    """Matches two images using SIFT + FLANN."""
    kp1, des1 = load_features(img1_path)
    kp2, des2 = load_features(img2_path)

    if des1 is None or des2 is None:
        print("[ERROR] One of the descriptors is None.")
        return 0

    # FLANN based matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    return len(good_matches)



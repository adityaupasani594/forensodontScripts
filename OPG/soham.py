import os
import pickle
import cv2
import numpy as np
from numpy.linalg import norm
from skimage.metrics import structural_similarity as ssim


def convert_to_bytes(img):
    """Convert OpenCV image to byte array for consistency."""
    return cv2.imencode('.jpg', img)[1].tobytes()


# ========= CACHING ========= #

def get_soham_cache_path(path):
    """Return path for ResNet cache file for given image path."""
    name = os.path.basename(path).replace(" ", "_")
    os.makedirs("soham_cache", exist_ok=True)
    return os.path.join("soham_cache", f"{name}_soham.pkl")


def extract_resnet_features_cached(img_bytes):
    """
    Dummy ResNet feature extractor.
    Replace this with actual ResNet feature extraction logic.
    """
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    # Placeholder: just normalize the array to mimic feature vector
    return arr[:512] / (np.linalg.norm(arr[:512]) + 1e-8)


def load_or_generate_soham_resnet_cache(img_path):
    """Load cached ResNet features, or compute and save them if not cached."""
    cache_path = get_soham_cache_path(img_path)
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            print(f"[CACHE] Loaded Soham ResNet cache from {cache_path}")
            return pickle.load(f)

    print(f"[CACHE] No cache found. Creating new Soham ResNet cache at {cache_path}")
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Failed to load image at {img_path}")
    img_bytes = convert_to_bytes(img)
    features = extract_resnet_features_cached(img_bytes)

    with open(cache_path, "wb") as f:
        pickle.dump(features, f)

    return features


# ========= FEATURE FUNCTIONS ========= #

def compute_sift_similarity(img1, img2):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    if des1 is None or des2 is None:
        return 0
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    return len(good) / (len(matches) + 1e-8)


def compute_brisk_similarity(img1, img2):
    brisk = cv2.BRISK_create()
    kp1, des1 = brisk.detectAndCompute(img1, None)
    kp2, des2 = brisk.detectAndCompute(img2, None)
    if des1 is None or des2 is None:
        return 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.match(des1, des2)
    return len(matches) / (len(kp1) + len(kp2) + 1e-8)


def compute_ssim_similarity(img1, img2):
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(img1_gray, img2_gray, full=True)
    return score


def compute_resnet_similarity(img_path1, img_path2):
    feat1 = load_or_generate_soham_resnet_cache(img_path1)
    feat2 = load_or_generate_soham_resnet_cache(img_path2)
    return float(np.dot(feat1, feat2) / (1e-8 + norm(feat1) * norm(feat2)))


# ========= MAIN PIPELINE ========= #

def soham_opencv_week5(am_img_path, pm_img_path):
    """
    Compare AM vs PM using SIFT, BRISK, SSIM, ResNet (cached).
    Returns a dict of scores.
    """
    am_img = cv2.imread(am_img_path)
    pm_img = cv2.imread(pm_img_path)
    if am_img is None or pm_img is None:
        raise ValueError("Could not load one of the input images")

    results = {
        'SIFT': compute_sift_similarity(am_img, pm_img),
        'BRISK': compute_brisk_similarity(am_img, pm_img),
        'SSIM': compute_ssim_similarity(am_img, pm_img),
        'RESNET': compute_resnet_similarity(am_img_path, pm_img_path)
    }

    return results

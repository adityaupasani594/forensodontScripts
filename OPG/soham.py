import cv2
import numpy as np
from PIL import Image
from functools import lru_cache
from skimage.metrics import structural_similarity as ssim
from numpy.linalg import norm
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image as keras_image

# Load ResNet model once
_resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

@lru_cache(maxsize=128)
def extract_color_mask_cached(img_bytes, color='green'):
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if color == 'green':
        lower = np.array([36, 50, 50])
        upper = np.array([89, 255, 255])
        return cv2.inRange(hsv, lower, upper)
    else:
        lower1 = np.array([0, 70, 50])
        upper1 = np.array([10, 255, 255])
        lower2 = np.array([170, 70, 50])
        upper2 = np.array([180, 255, 255])
        return cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1), cv2.inRange(hsv, lower2, upper2))

def apply_mask(img, mask):
    return cv2.bitwise_and(img, img, mask=mask)

def compute_ssim(img1, img2):
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(img1_gray, img2_gray, full=True)
    return score

def compute_feature_similarity(img1, img2, method='SIFT'):
    if method == 'SIFT':
        try:
            detector = cv2.SIFT_create(nfeatures=500)
        except:
            return 0
    else:
        detector = cv2.BRISK_create()

    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)
    if des1 is None or des2 is None:
        return 0
    matcher = cv2.BFMatcher(cv2.NORM_L2 if method == 'SIFT' else cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    good = [m for m in matches if m.distance < 60]
    return len(good) / max(len(kp1), len(kp2)) if max(len(kp1), len(kp2)) > 0 else 0

@lru_cache(maxsize=128)
def extract_resnet_features_cached(img_bytes):
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    x = keras_image.img_to_array(img_resized)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return _resnet_model.predict(x, verbose=0).flatten()

def compute_resnet_similarity_bytes(img_bytes1, img_bytes2):
    f1 = extract_resnet_features_cached(img_bytes1)
    f2 = extract_resnet_features_cached(img_bytes2)
    return float(np.dot(f1, f2) / (norm(f1) * norm(f2) + 1e-8))

def safe_metric(func, *args):
    try:
        return func(*args)
    except:
        return 0.0

def convert_to_bytes(img):
    return cv2.imencode('.jpg', img)[1].tobytes()

def soham_opencv_week5(am_img, pm_img):
    if isinstance(am_img, Image.Image):
        am_img = cv2.cvtColor(np.array(am_img), cv2.COLOR_RGB2BGR)
    if isinstance(pm_img, Image.Image):
        pm_img = cv2.cvtColor(np.array(pm_img), cv2.COLOR_RGB2BGR)

    am_bytes = convert_to_bytes(am_img)
    pm_bytes = convert_to_bytes(pm_img)

    # Masks
    am_green_mask = extract_color_mask_cached(am_bytes, 'green')
    pm_green_mask = extract_color_mask_cached(pm_bytes, 'green')
    am_red_mask = extract_color_mask_cached(am_bytes, 'red')
    pm_red_mask = extract_color_mask_cached(pm_bytes, 'red')

    # Apply masks
    am_green = apply_mask(am_img, am_green_mask)
    pm_green = apply_mask(pm_img, pm_green_mask)
    am_red = apply_mask(am_img, am_red_mask)
    pm_red = apply_mask(pm_img, pm_red_mask)

    # Metrics
    green_SIFT = safe_metric(compute_feature_similarity, am_green, pm_green, 'SIFT')
    green_BRISK = safe_metric(compute_feature_similarity, am_green, pm_green, 'BRISK')
    green_SSIM = safe_metric(compute_ssim, am_green, pm_green)
    green_RESNET = safe_metric(compute_resnet_similarity_bytes, am_bytes, pm_bytes)

    red_SIFT = safe_metric(compute_feature_similarity, am_red, pm_red, 'SIFT')
    red_BRISK = safe_metric(compute_feature_similarity, am_red, pm_red, 'BRISK')
    red_SSIM = safe_metric(compute_ssim, am_red, pm_red)
    red_RESNET = safe_metric(compute_resnet_similarity_bytes, am_bytes, pm_bytes)

    # Weighted sum fusion
    w = [0, 0.5, 0, 1, 0.5, 0.2]
    green_score = w[1]*green_SIFT + w[3]*green_BRISK + w[4]*green_SSIM + w[5]*green_RESNET
    red_score = w[1]*red_SIFT + w[3]*red_BRISK + w[4]*red_SSIM + w[5]*red_RESNET

    return (green_score + red_score) * 100

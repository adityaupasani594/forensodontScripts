import cv2
import numpy as np
from PIL import Image
from functools import lru_cache
from skimage.metrics import structural_similarity as ssim
from numpy.linalg import norm
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image as keras_image

# Load ResNet model once globally
_resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')


@lru_cache(maxsize=128)
def extract_color_mask_cached(img_bytes, color='green'):
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if color == 'green':
        lower, upper = np.array([36, 50, 50]), np.array([89, 255, 255])
        return cv2.inRange(hsv, lower, upper)
    # Red (two ranges)
    lower1, upper1 = np.array([0, 70, 50]), np.array([10, 255, 255])
    lower2, upper2 = np.array([170, 70, 50]), np.array([180, 255, 255])
    return cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1), cv2.inRange(hsv, lower2, upper2))


def apply_mask(img, mask):
    return cv2.bitwise_and(img, img, mask=mask)


def compute_ssim(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(gray1, gray2, full=True)
    return score


def compute_feature_similarity(img1, img2, method='SIFT'):
    try:
        detector = cv2.SIFT_create(nfeatures=500) if method == 'SIFT' else cv2.BRISK_create()
    except Exception:
        return 0.0

    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return 0.0

    matcher = cv2.BFMatcher(cv2.NORM_L2 if method == 'SIFT' else cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    good_matches = [m for m in matches if m.distance < 60]

    return len(good_matches) / max(len(kp1), len(kp2)) if max(len(kp1), len(kp2)) > 0 else 0.0


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

    # Extract masks and apply them
    masks = {
        'green': (extract_color_mask_cached(am_bytes, 'green'), extract_color_mask_cached(pm_bytes, 'green')),
        'red': (extract_color_mask_cached(am_bytes, 'red'), extract_color_mask_cached(pm_bytes, 'red'))
    }

    scores = {}
    for color in ['green', 'red']:
        am_masked = apply_mask(am_img, masks[color][0])
        pm_masked = apply_mask(pm_img, masks[color][1])

        scores[color] = {
            'SIFT': safe_metric(compute_feature_similarity, am_masked, pm_masked, 'SIFT'),
            'BRISK': safe_metric(compute_feature_similarity, am_masked, pm_masked, 'BRISK'),
            'SSIM': safe_metric(compute_ssim, am_masked, pm_masked),
            'RESNET': safe_metric(compute_resnet_similarity_bytes, am_bytes, pm_bytes)
        }

    weights = {'SIFT': 0.5, 'BRISK': 1.0, 'SSIM': 0.5, 'RESNET': 0.2}

    def weighted_score(s):
        return sum(weights[k] * s[k] for k in weights)

    final_score = (weighted_score(scores['green']) + weighted_score(scores['red'])) * 100
    return final_score

import os
import random
import pickle
import numpy as np
import torch
import timm
from PIL import Image
from torchvision import transforms

# ========== DETERMINISTIC SETUP ==========
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ========== DEVICE ==========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ========== TRANSFORM FACTORY ==========
def build_transform(size):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


TRANSFORMS = {
    'b1': build_transform(240),
    'b3': build_transform(300),
    'convnext': build_transform(224),
}

# ========== MODEL CACHE ==========
MODELS = {}


def get_model(name):
    if name in MODELS:
        return MODELS[name]
    model = timm.create_model(name, pretrained=True)
    if 'efficientnet_b1' in name or 'efficientnet_b3' in name:
        model.classifier = torch.nn.Identity()
    elif 'convnext' in name:
        model.head = torch.nn.Identity()
    model.to(DEVICE).eval()
    MODELS[name] = model
    return model


# ========== EMBEDDING EXTRACTOR ==========
def get_embedding(model, image, transform):
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = model(image_tensor).cpu().numpy().flatten()
    return emb / (np.linalg.norm(emb) + 1e-8)


# ========== SIMILARITY ==========
def dl_ensemble_scores(am1, am2, am3, pm1, pm2, pm3):
    return (np.dot(am1, pm1) + np.dot(am2, pm2) + np.dot(am3, pm3)) / 3.0 * 100


# ========== CACHE PATH UTIL ==========
def get_cache_path(folder):
    name = os.path.basename(os.path.abspath(folder)).replace(" ", "_")
    return os.path.join(folder, f".cache_{name}_embeddings.pkl")


def get_pm_cache_path(path):
    name = os.path.basename(path).replace(" ", "_")
    os.makedirs("pm_cache", exist_ok=True)
    return os.path.join("pm_cache", f"{name}_embeddings.pkl")


# ========== AM CACHE ==========
def generate_am_embedding_cache(am_folder):
    if not isinstance(am_folder, str):
        raise TypeError(f"Expected str for folder path, got {type(am_folder)}")

    cache_path = get_cache_path(am_folder)
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            cache = pickle.load(f)

    model_b1 = get_model('efficientnet_b1')
    model_b3 = get_model('efficientnet_b3')
    model_conv = get_model('convnext_base')

    image_files = [f for f in sorted(os.listdir(am_folder))
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    new_files = [f for f in image_files if f not in cache]

    for fname in new_files:
        path = os.path.join(am_folder, fname)
        try:
            img = Image.open(path).convert('RGB')
            emb_b1 = get_embedding(model_b1, img, TRANSFORMS['b1'])
            emb_b3 = get_embedding(model_b3, img, TRANSFORMS['b3'])
            emb_conv = get_embedding(model_conv, img, TRANSFORMS['convnext'])
            cache[fname] = [emb_b1, emb_b3, emb_conv]
            print(f"[INFO] Embedded and cached: {fname}")
        except Exception as e:
            print(f"[WARNING] Skipping {path}: {e}")

    with open(cache_path, 'wb') as f:
        pickle.dump(cache, f)

    return cache_path


def load_or_generate_am_cache(am_folder):
    generate_am_embedding_cache(am_folder)
    with open(get_cache_path(am_folder), 'rb') as f:
        return pickle.load(f)


# ========== PM CACHE ==========
def load_or_generate_pm_embedding(pm_image_path):
    cache_path = get_pm_cache_path(pm_image_path)
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    img = Image.open(pm_image_path).convert("RGB")
    emb_b1 = get_embedding(get_model('efficientnet_b1'), img, TRANSFORMS['b1'])
    emb_b3 = get_embedding(get_model('efficientnet_b3'), img, TRANSFORMS['b3'])
    emb_conv = get_embedding(get_model('convnext_base'), img, TRANSFORMS['convnext'])

    with open(cache_path, 'wb') as f:
        pickle.dump([emb_b1, emb_b3, emb_conv], f)

    return [emb_b1, emb_b3, emb_conv]


# ========== MATCHING ==========
def compare_with_am(pm_image, am_input, topk=5):
    if isinstance(pm_image, str):
        pm_embs = load_or_generate_pm_embedding(pm_image)
    else:
        raise TypeError("PM image must be a path to a file")

    results = []

    if isinstance(am_input, str):
        am_cache = load_or_generate_am_cache(am_input)
        for fname, am_embs in am_cache.items():
            try:
                score = dl_ensemble_scores(*am_embs, *pm_embs)
                results.append((fname, score))
            except Exception as e:
                print(f"[ERROR] Failed to compare with {fname}: {e}")

    elif isinstance(am_input, dict):
        model_b1 = get_model('efficientnet_b1')
        model_b3 = get_model('efficientnet_b3')
        model_conv = get_model('convnext_base')

        for name, img in am_input.items():
            try:
                img = img.convert("RGB")
                emb_b1 = get_embedding(model_b1, img, TRANSFORMS['b1'])
                emb_b3 = get_embedding(model_b3, img, TRANSFORMS['b3'])
                emb_conv = get_embedding(model_conv, img, TRANSFORMS['convnext'])
                score = dl_ensemble_scores(emb_b1, emb_b3, emb_conv, *pm_embs)
                results.append((name, score))
            except Exception as e:
                print(f"[ERROR] Skipping {name}: {e}")
    else:
        raise TypeError("am_input must be a folder path or a dict of PIL.Images")

    results.sort(key=lambda x: -x[1])
    print("Vedant done")
    return results[:topk]

import os
import concurrent.futures as cf
import shutil
import threading
import time
import pickle
import numpy as np
import tempfile

import requests
from flask import Flask, request, send_file, jsonify
from dotenv import load_dotenv
from supabase import create_client
from PIL import Image
from io import BytesIO

from OPG.aadi import aadi_opencv_week5
from OPG.sarvankar import sarvankar
from OPG.soham import soham_opencv_week5
from OPG.vedant import compare_with_am
from marking import mark_opg_pipeline

# ===== CPU tuning =====
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
try:
    import torch

    torch.set_num_threads(max(1, os.cpu_count() // 2))
    torch.set_num_interop_threads(1)
except Exception:
    pass

# ===== Flask + Supabase setup =====
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# --- OPG bucket ---
SUPABASE_BUCKET = os.getenv("BUCKET_NAME")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
app = Flask(__name__)

# ===== Parallel executor =====
EXECUTOR = cf.ProcessPoolExecutor(max_workers=4)

# ===== Cache files =====
IMAGE_CACHE_FILE = "am_cache.pkl"

# ===== Global stores =====
AM_IMAGES = {}  # filename -> np.ndarray
AM_LOCK = threading.Lock()


# ====== Helper functions ======
def pil_to_numpy(img: Image.Image):
    return np.array(img)


def get_image_url(bucket: str, filename: str) -> str:
    try:
        return supabase.storage.from_(bucket).get_public_url(filename)
    except Exception:
        return None


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def ensure_pil(img):
    if isinstance(img, np.ndarray):
        return Image.fromarray(img)
    return img


# -------------------- ROUTE: Add AM Image --------------------
@app.route("/add_am", methods=["POST"])
def add_am_image():
    if "file" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No image selected"}), 400

    try:
        # Convert to numpy
        img = Image.open(file.stream).convert("RGB")
        img_np = np.array(img)

        # Upload to Supabase bucket
        file.seek(0)  # reset stream pointer
        supabase.storage.from_(SUPABASE_BUCKET).upload(file.filename, file, {"upsert": True})

        # Update in-memory cache
        with AM_LOCK:
            AM_IMAGES[file.filename] = img_np
            save_pickle(AM_IMAGES, IMAGE_CACHE_FILE)

        return jsonify({
            "message": f"Image {file.filename} added successfully",
            "url": get_image_url(SUPABASE_BUCKET, file.filename),
            "total_cached": len(AM_IMAGES)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def fetch_am_images_from_supabase():
    """Download and return OPG AM images from Supabase as numpy arrays."""
    fresh = {}
    response = supabase.storage.from_(SUPABASE_BUCKET).list()
    for item in response:
        filename = item["name"]
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        try:
            blob = supabase.storage.from_(SUPABASE_BUCKET).download(filename)
            img = Image.open(BytesIO(blob)).convert("RGB")
            fresh[filename] = pil_to_numpy(img)
        except Exception as e:
            print(f"[WARN] Skip {filename}: {e}")
    return fresh


# -------------------- ROUTES --------------------
@app.route("/mark", methods=["POST"])
def mark_opg():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    with tempfile.TemporaryDirectory() as tmpdir:
        image_path = os.path.join(tmpdir, file.filename)
        file.save(image_path)

        try:
            marked_path = mark_opg_pipeline(image_path, tmpdir)

            final_dir = os.path.join(app.root_path, "static", "marked")
            os.makedirs(final_dir, exist_ok=True)
            final_path = os.path.join(final_dir, os.path.basename(marked_path))
            shutil.copy(marked_path, final_path)

            return send_file(final_path, mimetype="image/png")
        except Exception as e:
            return jsonify({"error": str(e)}), 500


def load_am_images():
    """Load cached AM images or fetch + cache from OPG Supabase bucket."""
    global AM_IMAGES
    try:
        if os.path.exists(IMAGE_CACHE_FILE):
            AM_IMAGES = load_pickle(IMAGE_CACHE_FILE)
            print(f"[INFO] Loaded {len(AM_IMAGES)} OPG AM images from cache.")
        else:
            fresh = fetch_am_images_from_supabase()
            AM_IMAGES = fresh
            save_pickle(fresh, IMAGE_CACHE_FILE)
            print(f"[INFO] Downloaded and cached {len(fresh)} OPG AM images.")
    except Exception as e:
        print(f"[ERROR] Failed to load OPG AM images: {e}")


def background_refresh(interval=600):
    """Refresh OPG AM images every `interval` seconds."""
    while True:
        time.sleep(interval)
        try:
            fresh = fetch_am_images_from_supabase()
            with AM_LOCK:
                AM_IMAGES.update(fresh)
            save_pickle(AM_IMAGES, IMAGE_CACHE_FILE)
            print("[INFO] Background refresh completed.")
        except Exception as e:
            print(f"[ERROR] Background refresh failed: {e}")


# ===== Initial load =====
load_am_images()
threading.Thread(target=background_refresh, args=(600,), daemon=True).start()


# ===== Worker functions =====
def run_soham(pm_image, am_images, topk=5):
    scores = []
    for name, am_img in am_images.items():
        try:
            score = soham_opencv_week5(pm_image, am_img)
            scores.append((name, score))
        except Exception as e:
            print(f"[WARN] soham failed on {name}: {e}")
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:topk]


def run_dl(pm_image, am_images, topk=5):
    pm_pil = ensure_pil(pm_image)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
        pm_pil.save(tmp_path)

    try:
        am_dict = {name: ensure_pil(am) for name, am in am_images.items()}
        return compare_with_am(tmp_path, am_dict, topk=topk)
    except Exception as e:
        print(f"[ERROR] DL compare failed: {e}")
        return []
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


def run_sarvankar(pm_image, am_images, topk=5):
    scores = []
    for name, am_img in am_images.items():
        try:
            score = sarvankar(pm_image, am_img)
            scores.append((name, score))
        except Exception as e:
            print(f"[WARN] sarvankar failed on {name}: {e}")
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:topk]


def run_aadi(pm_image, am_images, topk=5):
    scores = []
    for name, am_img in am_images.items():
        try:
            score = aadi_opencv_week5(pm_image, am_img)
            scores.append((name, score))
        except Exception as e:
            print(f"[WARN] aadi failed on {name}: {e}")
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:topk]


# -------------------- ROUTE: OPG Match --------------------
@app.route("/match", methods=["POST"])
def match_pm_image():
    if 'file' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files['file']
    if file.filename == "":
        return jsonify({"error": "No image selected"}), 400

    try:
        pm_image = np.array(Image.open(file.stream).convert("RGB"))
        with AM_LOCK:
            if not AM_IMAGES:
                return jsonify({"error": "No AM images found"}), 500
            am_images = dict(AM_IMAGES)

        futures = {
            "soham_opencv": EXECUTOR.submit(run_soham, pm_image, am_images, 5),
            "dl_ensemble": EXECUTOR.submit(run_dl, pm_image, am_images, 5),
            "sarvankar": EXECUTOR.submit(run_sarvankar, pm_image, am_images, 5),
            "aadi": EXECUTOR.submit(run_aadi, pm_image, am_images, 5),
        }

        algo_results, image_counts = {}, {}
        for k, fut in futures.items():
            try:
                results = fut.result()
            except Exception as e:
                print(f"[ERROR] {k} failed: {e}")
                results = []
            algo_results[k] = [{"image": n, "score": float(s), "url": get_image_url(SUPABASE_BUCKET, n)} for n, s in
                               results]
            for n, _ in results:
                image_counts[n] = image_counts.get(n, 0) + 1

        top1s = {k: (v[0] if v else {"image": "None", "score": 0.0, "url": None})
                 for k, v in algo_results.items()}
        common_images = [{"image": img, "url": get_image_url(SUPABASE_BUCKET, img)} for img, c in image_counts.items()
                         if c > 1]
        best_match = max(top1s.values(), key=lambda x: x["score"]) if top1s else {"image": "None", "score": 0.0,
                                                                                  "url": None}

        return jsonify({
            "top5_per_algorithm": algo_results,
            "top1_per_algorithm": top1s,
            "common_images_in_top5": common_images,
            "best_match_overall": best_match
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print(app.url_map)
    app.run(host="0.0.0.0", port=5000, threaded=False)

import os
import concurrent.futures as cf
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from supabase import create_client
from PIL import Image
from io import BytesIO
from werkzeug.utils import secure_filename

from OPG.aadi import aadi_opencv_week5
from OPG.sarvankar import sarvankar
from OPG.soham import soham_opencv_week5
from OPG.vedant import compare_with_am

# ===== Load environment variables =====
load_dotenv()

# Debug print to confirm .env loaded correctly
print("DEBUG -- URL:", os.getenv("SUPABASE_URL"))
print("DEBUG -- KEY:", os.getenv("SUPABASE_KEY")[:10] if os.getenv("SUPABASE_KEY") else None, "...")
print("DEBUG -- BUCKET:", os.getenv("BUCKET_NAME"))

# ===== CPU-only tuning (optional but recommended) =====
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
try:
    import torch
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    torch.set_num_interop_threads(1)
except Exception:
    pass

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_BUCKET = os.getenv("BUCKET_NAME")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
app = Flask(__name__)

TEMP_DIR = "temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)

# Thread pool for algorithm-level parallelism (CPU-only)
EXECUTOR = cf.ThreadPoolExecutor(max_workers=4)  # one per algorithm


def get_am_images_from_bucket():
    response = supabase.storage.from_(SUPABASE_BUCKET).list()
    images = {}
    for item in response:
        filename = item["name"]
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        try:
            blob = supabase.storage.from_(SUPABASE_BUCKET).download(filename)
            images[filename] = Image.open(BytesIO(blob)).convert("RGB")
        except Exception as e:
            print(f"[WARN] Skip {filename}: {e}")
    return images


@app.route("/match", methods=["POST"])
def match_pm_image():
    if 'file' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files['file']
    if file.filename == "":
        return jsonify({"error": "No image selected"}), 400

    filename = secure_filename(file.filename)
    temp_path = os.path.join(TEMP_DIR, filename)
    file.save(temp_path)

    try:
        pm_image = Image.open(temp_path).convert("RGB")
        am_images = get_am_images_from_bucket()
        if not am_images:
            return jsonify({"error": "No AM images found in Supabase bucket"}), 500

        # --- define the four tasks ---
        def run_soham():
            pairs = ((n, soham_opencv_week5(img, pm_image)) for n, img in am_images.items())
            return max(pairs, key=lambda x: x[1])

        def run_dl():
            out = compare_with_am(temp_path, am_images, topk=1)
            return out[0] if out else ("None", 0.0)

        def run_sarvankar():
            pairs = ((n, sarvankar(img, pm_image)) for n, img in am_images.items())
            return max(pairs, key=lambda x: x[1])

        def run_aadi():
            pairs = ((n, aadi_opencv_week5(img, pm_image)) for n, img in am_images.items())
            return max(pairs, key=lambda x: x[1])

        # --- submit in parallel ---
        futures = {
            "soham_opencv": EXECUTOR.submit(run_soham),
            "dl_ensemble":  EXECUTOR.submit(run_dl),
            "sarvankar":    EXECUTOR.submit(run_sarvankar),
            "aadi":         EXECUTOR.submit(run_aadi),
        }

        # --- gather results ---
        results = {}
        for k, fut in futures.items():
            try:
                name, score = fut.result()
            except Exception as e:
                print(f"[ERROR] {k} failed: {e}")
                name, score = "None", 0.0
            results[k] = {"image": name, "score": float(score)}

        return jsonify({"best_matches": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


if __name__ == "__main__":
    # For production, run behind gunicorn/uvicorn with multiple workers
    app.run(host="0.0.0.0", port=5000)

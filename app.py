import os
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

# Load environment variables
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_BUCKET = os.getenv("BUCKET_NAME")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

app = Flask(__name__)

TEMP_DIR = "temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)


def get_am_images_from_bucket():
    response = supabase.storage.from_(SUPABASE_BUCKET).list()
    images = {}

    for item in response:
        filename = item['name']
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        try:
            image_data = supabase.storage.from_(SUPABASE_BUCKET).download(filename)
            image = Image.open(BytesIO(image_data)).convert("RGB")
            images[filename] = image
        except Exception as e:
            print(f"Failed to download or open {filename}: {e}")
            continue
    return images


# === API Endpoint ===
@app.route("/match", methods=["POST"])
def match_pm_image():
    if 'file' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['file']
    if file.filename == "":
        return jsonify({"error": "No image selected"}), 400

    try:
        # Save uploaded image to disk
        filename = secure_filename(file.filename)
        temp_path = os.path.join(TEMP_DIR, filename)
        file.save(temp_path)

        # Load PM image (for PIL-based methods)
        pm_image = Image.open(temp_path).convert("RGB")

        # Get AM images
        am_images = get_am_images_from_bucket()
        if not am_images:
            return jsonify({"error": "No AM images found in Supabase bucket"}), 500

        # --- Algorithm 1: soham_opencv ---
        opencv_results = [(am_name, soham_opencv_week5(am_img, pm_image)) for am_name, am_img in am_images.items()]
        best_opencv_match = max(opencv_results, key=lambda x: x[1])

        # --- Algorithm 2: DL Ensemble (needs file path) ---
        dl_results = compare_with_am(temp_path, am_images, topk=1)
        best_dl_match = dl_results[0] if dl_results else ("None", 0)

        # --- Algorithm 3: sarvankar_function ---
        sarvankar_results = [(am_name, sarvankar(am_img, pm_image)) for am_name, am_img in am_images.items()]
        best_sarvankar_match = max(sarvankar_results, key=lambda x: x[1])

        # --- Algorithm 4: aadi_function ---
        aadi_results = [(am_name, aadi_opencv_week5(am_img, pm_image)) for am_name, am_img in am_images.items()]
        best_aadi_match = max(aadi_results, key=lambda x: x[1])

        return jsonify({
            "best_matches": {
                "soham_opencv": {"image": best_opencv_match[0], "score": best_opencv_match[1]},
                "dl_ensemble": {"image": best_dl_match[0], "score": best_dl_match[1]},
                "sarvankar": {"image": best_sarvankar_match[0], "score": best_sarvankar_match[1]},
                "aadi": {"image": best_aadi_match[0], "score": best_aadi_match[1]},
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

import concurrent.futures as cf
import os
import uuid
from collections import Counter
from io import BytesIO

from PIL import Image, UnidentifiedImageError
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from supabase import create_client
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

EXECUTOR = cf.ThreadPoolExecutor(max_workers=4)


def get_am_images_from_bucket():
    response = supabase.storage.from_(SUPABASE_BUCKET).list()
    images = {}
    for item in response:
        filename = item.get("name")
        if not filename or not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        try:
            blob = supabase.storage.from_(SUPABASE_BUCKET).download(filename)
            images[filename] = Image.open(BytesIO(blob)).convert("RGB")
        except Exception as e:
            print(f"[WARN] Skip {filename}: {e}")
    return images


def _run_algo_safe(func, am_images, pm_image, topk=5):
    try:
        results = [(am_name, func(am_img, pm_image)) for am_name, am_img in am_images.items()]
        return sorted(results, key=lambda x: x[1], reverse=True)[:topk]
    except Exception as e:
        return [("error", str(e))]


def _run_dl_ensemble_safe(temp_path, am_images, topk=5):
    try:
        results = compare_with_am(temp_path, am_images, topk=topk)
        if not results:
            return [("None", 0)] * topk
        return results[:topk]
    except Exception as e:
        return [("error", str(e))]


@app.route("/match", methods=["POST"])
def match_pm_image():
    temp_path = None
    try:
        if "file" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No image selected"}), 400

        filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
        temp_path = os.path.join(TEMP_DIR, filename)
        file.save(temp_path)

        try:
            pm_image = Image.open(temp_path).convert("RGB")
        except UnidentifiedImageError:
            return jsonify({"error": "Uploaded file is not a valid image"}), 400

        am_images = get_am_images_from_bucket()
        if not am_images:
            return jsonify({"error": "No AM images found in Supabase bucket"}), 500

        with cf.ThreadPoolExecutor(max_workers=4) as executor:
            future_map = {
                "soham_opencv": executor.submit(_run_algo_safe, soham_opencv_week5, am_images, pm_image, 5),
                "dl_ensemble": executor.submit(_run_dl_ensemble_safe, temp_path, am_images, 5),
                "sarvankar": executor.submit(_run_algo_safe, sarvankar, am_images, pm_image, 5),
                "aadi": executor.submit(_run_algo_safe, aadi_opencv_week5, am_images, pm_image, 5),
            }

            algo_results = {}
            for name, fut in future_map.items():
                try:
                    algo_results[name] = fut.result()
                except Exception as e:
                    algo_results[name] = [("error", str(e))]

        # Common matches across algorithms' top-5s
        match_counter = Counter()
        for match_list in algo_results.values():
            match_counter.update([name for name, _ in match_list])
        common_matches = [name for name, count in match_counter.items() if count >= 2]

        # Top-1 map (image names only), ignoring algorithms that errored
        top1s = {
            algo: matches[0][0]
            for algo, matches in algo_results.items()
            if matches and matches[0][0] != "error"
        }

        # Final match selection:
        # 1) If any common candidates exist, choose the one with the best average rank across algos where it appears.
        # 2) Else, majority vote among top-1s.
        final_match = None
        if common_matches:
            avg_ranks = {}
            for cid in common_matches:
                ranks = []
                for matches in algo_results.values():
                    for idx, (name, _score) in enumerate(matches):
                        if name == cid:
                            ranks.append(idx + 1)  # rank is 1-based
                            break
                avg_ranks[cid] = (sum(ranks) / len(ranks)) if ranks else float("inf")
            final_match = min(avg_ranks, key=avg_ranks.get)
        else:
            if top1s:
                vote_counts = Counter(top1s.values())
                final_match = vote_counts.most_common(1)[0][0]

        response = {
            "top5_matches": {
                algo: [{"image": name, "score": score} for name, score in matches]
                for algo, matches in algo_results.items()
            },
            "top1_matches": {
                algo: (
                    {"image": matches[0][0], "score": matches[0][1]}
                    if matches else {"image": "None", "score": 0}
                )
                for algo, matches in algo_results.items()
            },
            "common_matches_in_2_or_more_algos": common_matches,
            "final_match": final_match,
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                print(f"[WARN] Failed to remove temp file {temp_path}: {e}")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

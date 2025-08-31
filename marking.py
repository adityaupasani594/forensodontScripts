import os
import cv2
import json
import urllib.request
import numpy as np
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor

# ============================
# 1. Load YOLO model
# ============================
yolo_model = YOLO("runs/detect/tooth-detector-v2/weights/best.pt")  # update if path differs


# ============================
# 2. YOLO: Bounding box detector
# ============================
def draw_and_save_boxes(image_path, output_folder, json_folder):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"❌ Could not read: {image_path}")

    results = yolo_model(image_path)[0].boxes
    bboxes = []

    for box in results:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        bboxes.append([x1, y1, x2, y2])

    os.makedirs(output_folder, exist_ok=True)
    filename = os.path.basename(image_path)
    save_path = os.path.join(output_folder, filename)
    cv2.imwrite(save_path, img)

    os.makedirs(json_folder, exist_ok=True)
    json_name = os.path.splitext(filename)[0] + ".json"
    json_path = os.path.join(json_folder, json_name)
    with open(json_path, "w") as f:
        json.dump(bboxes, f)

    return json_path


# ============================
# 3. SAM: Segmentation
# ============================
checkpoint_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
checkpoint_path = "sam_vit_b.pth"
if not os.path.exists(checkpoint_path):
    print("📥 Downloading SAM checkpoint...")
    urllib.request.urlretrieve(checkpoint_url, checkpoint_path)

sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
sam.to("cpu")
predictor = SamPredictor(sam)


def generate_mask(image_path, json_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"❌ Could not read image: {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"⚠️ Bounding box JSON not found: {json_path}")

    with open(json_path, "r") as f:
        boxes = np.array(json.load(f))

    predictor.set_image(image_rgb)
    jaw_mask = np.zeros(image.shape[:2], dtype=np.uint8)

    for box in boxes:
        masks, _, _ = predictor.predict(box=box, multimask_output=False)
        mask = masks[0].astype(np.uint8)
        jaw_mask = np.maximum(jaw_mask, mask)

    output_path = os.path.join(
        output_folder, os.path.splitext(os.path.basename(image_path))[0] + "_mask.png"
    )
    cv2.imwrite(output_path, jaw_mask * 255)
    return output_path


# ============================
# 4. Overlay methods
# ============================
def draw_morphological_boundaries(img, mask):
    if len(mask.shape) == 3:
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        mask_gray = mask

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    gradient = cv2.morphologyEx(mask_gray, cv2.MORPH_GRADIENT, kernel)

    result = img.copy()
    result[gradient > 0] = [0, 255, 0]  # Green edges
    return result


def overlay_mask(am_img_path, sam_mask_path, save_folder):
    img = cv2.imread(am_img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"❌ Could not load AM image: {am_img_path}")

    mask = cv2.imread(sam_mask_path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise ValueError(f"❌ Could not load SAM mask: {sam_mask_path}")

    if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

    if mask.ndim == 3:
        if mask.shape[2] == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        elif mask.shape[2] == 4:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGRA2GRAY)
        else:
            mask = mask[:, :, 0]

    morph_img = draw_morphological_boundaries(img, mask)
    combined = cv2.resize(morph_img, (img.shape[1], img.shape[0]))

    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, os.path.basename(am_img_path))
    cv2.imwrite(save_path, combined)
    return save_path


# ============================
# 5. Main pipeline
# ============================
def mark_opg_pipeline(image_path, workdir):
    bbox_folder = os.path.join(workdir, "bboxes")
    mask_folder = os.path.join(workdir, "masks")
    output_folder = os.path.join(workdir, "marked")

    # Step 1: YOLO bboxes
    json_path = draw_and_save_boxes(image_path, output_folder=bbox_folder, json_folder=bbox_folder)

    # Step 2: SAM segmentation
    mask_path = generate_mask(image_path, json_path, output_folder=mask_folder)

    # Step 3: Overlay
    marked_path = overlay_mask(image_path, mask_path, save_folder=output_folder)

    return marked_path

from supabase import create_client

# ====== CONFIGURATION ======
from dotenv import load_dotenv
import os

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
BUCKET_NAME = os.getenv("BUCKET_NAME")
FOLDER_PATH = ""

# ====== INIT SUPABASE CLIENT ======
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


# ====== LIST IMAGES ======
def list_images(bucket: str, folder: str = ""):
    result = supabase.storage.from_(bucket).list(path=folder)
    image_files = [file["name"] for file in result if file["name"].lower().endswith((".jpg", ".jpeg", ".png"))]
    return image_files


# ====== RUN LISTING ======
if __name__ == "__main__":
    images = list_images(BUCKET_NAME, FOLDER_PATH)
    print("Found image files:")
    for img in images:
        print(f"- {img}")

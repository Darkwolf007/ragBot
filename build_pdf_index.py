import os, json, faiss, torch, fitz, pytesseract, numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
PDF_PATH = "document.pdf"
OUT_DIR = "output"
os.makedirs(f"{OUT_DIR}/images", exist_ok=True)

TEXT_MODEL_NAME = "all-MiniLM-L6-v2"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# --------------------------------------------------
# SANITY CHECKS
# --------------------------------------------------
if not os.path.exists(PDF_PATH):
    raise FileNotFoundError(f"❌ PDF not found: {PDF_PATH}")

torch_version = tuple(map(int, torch.__version__.split("+")[0].split(".")))
if torch_version < (2, 6, 0):
    raise RuntimeError(f"❌ Torch {torch.__version__} detected. Please upgrade to >=2.6.0")

# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------
print("Loading models ...")
text_model = SentenceTransformer(TEXT_MODEL_NAME)
clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)

# --------------------------------------------------
# EXTRACT CONTENT
# --------------------------------------------------
doc = fitz.open(PDF_PATH)
records = []

print(f"Reading {PDF_PATH} ...")
for i, page in enumerate(doc):
    # ---- Text extraction ----
    text = page.get_text("text")
    if text.strip():
        records.append({"type": "text", "page": i + 1, "content": text})

    # ---- Image extraction ----
    for j, img in enumerate(page.get_images(full=True)):
        xref = img[0]
        pix = fitz.Pixmap(doc, xref)
        image_path = f"{OUT_DIR}/images/page_{i + 1}_{j}.png"
        pix.save(image_path)

        try:
            ocr_text = pytesseract.image_to_string(Image.open(image_path))
        except Exception as e:
            print(f"⚠️ OCR failed on {image_path}: {e}")
            ocr_text = ""

        records.append({
            "type": "image",
            "page": i + 1,
            "path": image_path,
            "ocr_text": ocr_text
        })

print(f"✅ Extracted {len(records)} items")

# --------------------------------------------------
# EMBEDDINGS
# --------------------------------------------------
text_embeddings, metadata = [], []
print("Generating embeddings ...")

def safe_rgb(image_path):
    """Safely open an image and ensure it's valid RGB."""
    try:
        image = Image.open(image_path)
        arr = np.array(image)
        # skip if malformed or very small (<10px)
        if arr.ndim < 2 or min(arr.shape[0], arr.shape[1]) < 10:
            raise ValueError("Tiny or invalid image")
        if image.mode != "RGB":
            image = image.convert("RGB")
        # Fix channel-flipped weird arrays (1,H,3) or (3,H,3)
        if arr.ndim == 3 and arr.shape[0] in (1, 3):
            image = Image.fromarray(arr.transpose(1, 2, 0))
        return image
    except Exception as e:
        print(f"⚠️ Skipping bad image {image_path}: {e}")
        return None

for rec in records:
    try:
        if rec["type"] == "text":
            emb = text_model.encode(rec["content"], normalize_embeddings=True)
            text_embeddings.append(emb)
            metadata.append(rec)

        elif rec["type"] == "image":
            image = safe_rgb(rec["path"])
            if image is None:
                continue

            inputs = clip_processor(images=image, return_tensors="pt")
            with torch.no_grad():
                emb_img = clip_model.get_image_features(**inputs)
            emb_img = emb_img.squeeze().cpu().numpy()
            emb_img = emb_img / np.linalg.norm(emb_img)

            emb_ocr = text_model.encode(rec["ocr_text"], normalize_embeddings=True)

            # --- fuse or concatenate safely ---
            if emb_img.shape[0] != emb_ocr.shape[0]:
                emb_final = np.concatenate([emb_img, emb_ocr])
            else:
                emb_final = (emb_img + emb_ocr) / 2.0
            emb_final = emb_final / np.linalg.norm(emb_final)

            text_embeddings.append(emb_final)
            metadata.append(rec)

    except Exception as e:
        print(f"⚠️ Skipped {rec.get('path', 'text')} due to error: {e}")
        continue

# --------------------------------------------------
# UNIFY EMBEDDING DIMENSIONS
# --------------------------------------------------
dims = [e.shape[0] for e in text_embeddings]
max_dim = max(dims)
for i, emb in enumerate(text_embeddings):
    if emb.shape[0] < max_dim:
        pad = np.zeros(max_dim - emb.shape[0])
        text_embeddings[i] = np.concatenate([emb, pad])

# --------------------------------------------------
# VECTOR INDEX
# --------------------------------------------------
if not text_embeddings:
    raise RuntimeError("❌ No embeddings generated. Check your PDF or model setup.")

emb_matrix = np.vstack(text_embeddings).astype("float32")
index = faiss.IndexFlatIP(emb_matrix.shape[1])
index.add(emb_matrix)

faiss.write_index(index, f"{OUT_DIR}/pdf_index.faiss")
with open(f"{OUT_DIR}/metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print(f"✅ Saved index to {OUT_DIR}/pdf_index.faiss and metadata.json")
print(f"Total vectors: {len(metadata)} | Unified Dim: {emb_matrix.shape[1]}")

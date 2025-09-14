import os
import re
import io
import zipfile
import shutil
import hashlib
import requests
from pathlib import Path
from typing import Optional
import streamlit as st
import io, os, zipfile, tempfile, pickle
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]  # repo root (parent of streamlit_app/)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data_pipeline import ImagePreprocessor
from src.utils import safe_extract_zip

# Optional imports for robust prediction calls
try:
    from PIL import Image
except Exception:
    Image = None

# Try to reuse user's existing predict functions if present
_predict_image_func = None
_predict_zip_func = None
try:
    from src.predict import predict_image as _predict_image_func  # type: ignore
except Exception:
    pass
try:
    from src.predict import predict_zip as _predict_zip_func  # type: ignore
except Exception:
    pass

# ======================
# Config / Defaults
# ======================
_DEF_CHUNK = 1 << 20  # 1 MiB
MODEL_LOCAL_PATH = os.getenv("MODEL_LOCAL_PATH", "artifacts/model.pkl")  # adjust default if needed
GDRIVE_FILE_ID_OR_URL = os.getenv("GDRIVE_FILE_ID", "1O5TnYOzuZT2_nsG3EnWqAQDrWq4Pk5hH")  # file ID or full URL
MODEL_SHA256 = (os.getenv("MODEL_SHA256", "") or "").lower()  # optional integrity check

# ======================
# Helpers: hashing, I/O
# ======================
def sha256_file(path: str, chunk_size: int = _DEF_CHUNK) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

def stream_to_file(resp: requests.Response, dst_path: str, chunk_size: int = _DEF_CHUNK):
    resp.raise_for_status()
    Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
    tmp = f"{dst_path}.part"
    with open(tmp, "wb") as f:
        for chunk in resp.iter_content(chunk_size):
            if chunk:
                f.write(chunk)
    os.replace(tmp, dst_path)

# ======================
# Google Drive (public)
# ======================
def _gdrive_confirm_token(resp: requests.Response) -> Optional[str]:
    for k, v in resp.cookies.items():
        if k.startswith("download_warning"):
            return v
    if "confirm=" in resp.url:
        return resp.url.split("confirm=")[1].split("&")[0]
    return None

def extract_file_id(maybe_id_or_url: str) -> str:
    s = maybe_id_or_url.strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{10,}", s):
        return s
    m = re.search(r"/file/d/([A-Za-z0-9_-]{10,})", s)
    if m:
        return m.group(1)
    m = re.search(r"[?&]id=([A-Za-z0-9_-]{10,})", s)
    if m:
        return m.group(1)
    raise ValueError("Could not extract Google Drive file ID. Provide a file ID or a valid Drive URL.")

def download_gdrive_public_cached(file_id_or_url: str, dst_path: str, expected_sha256: Optional[str] = None, timeout: int = 90) -> str:
    # Cache hit?
    if os.path.exists(dst_path) and os.path.getsize(dst_path) > 0:
        if expected_sha256:
            try:
                if sha256_file(dst_path) == expected_sha256.lower():
                    return dst_path
            except Exception:
                pass
        else:
            return dst_path

    file_id = extract_file_id(file_id_or_url)
    sess = requests.Session()
    base = "https://drive.google.com/uc"
    params = {"id": file_id, "export": "download"}
    r = sess.get(base, params=params, stream=True, timeout=timeout)
    token = _gdrive_confirm_token(r)
    if token:
        params["confirm"] = token
        r = sess.get(base, params=params, stream=True, timeout=timeout)

    # Basic HTML/Quota/Permission detection
    ctype = r.headers.get("Content-Type", "").lower()
    if "text/html" in ctype:
        head = r.text[:1024].lower()
        if any(x in head for x in ("quota", "access denied", "sign in", "account", "permission")):
            raise RuntimeError("Google Drive public link error: quota or permission issue. Make a copy to get a new file ID or adjust sharing.")

    stream_to_file(r, dst_path)

    if expected_sha256 and sha256_file(dst_path) != expected_sha256.lower():
        raise RuntimeError("SHA-256 mismatch after download.")
    return dst_path

# ======================
# Uploaders (same UX)
# ======================
ALLOWED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

def save_uploaded_file(uploaded_file, dst_dir: str) -> Path:
    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)
    safe_name = Path(uploaded_file.name).name  # flatten any subpath
    out_path = dst / safe_name
    with open(out_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return out_path

def safe_extract_zip(zip_uploaded_file, dst_dir: str):
    """Return list of extracted image Paths (skips __MACOSX, hidden, non-images)."""
    dst = Path(dst_dir); dst.mkdir(parents=True, exist_ok=True)
    tmp_zip = dst / "upload.tmp.zip"
    with open(tmp_zip, "wb") as f:
        f.write(zip_uploaded_file.getbuffer())

    extracted = []
    with zipfile.ZipFile(tmp_zip, "r") as zf:
        for info in zf.infolist():
            name = info.filename
            if info.is_dir():
                continue
            base = Path(name).name
            if base.startswith(".") or name.startswith("__MACOSX/") or base.startswith("._"):
                continue
            ext = Path(base).suffix.lower()
            if ext not in ALLOWED_EXTS:
                continue
            out_path = dst / base  # flatten to avoid path traversal
            with zf.open(info) as src, open(out_path, "wb") as dstf:
                shutil.copyfileobj(src, dstf, length=_DEF_CHUNK)
            extracted.append(out_path)
    try:
        tmp_zip.unlink(missing_ok=True)
    except Exception:
        pass
    return extracted

# ======================
# Prediction bridges
# ======================

@st.cache_resource
def load_artifacts(cfg_path: str):
    cfg = load_config(cfg_path)
    pre = ImagePreprocessor.load(cfg.paths['pipeline_pkl'])
    with open(cfg.paths['model_pkl'], 'rb') as f:
        wrapper = pickle.load(f)
    return cfg, pre, wrapper

def preprocess_and_predict_file(pre, wrapper, file_bytes, filename):
    tmpdir = tempfile.mkdtemp()
    tmppath = os.path.join(tmpdir, filename)
    with open(tmppath, 'wb') as f:
        f.write(file_bytes)
    arr = pre.preprocess_path(tmppath)
    x = np.expand_dims(arr, axis=0)
    idxs, probs = wrapper.predict(x)
    i = int(idxs[0]); conf = float(np.max(probs[0]))
    return i, conf

def call_predict_image(artifact_or_path, image_path: str):
    """Try to call user's predict_image with a PATH; fallback to simple behavior."""
    if _predict_image_func is not None:
        try:
            # Preferred: pass artifact/path directly
            return _predict_image_func(artifact_or_path, image_path)
        except TypeError:
            # Try passing a file-like
            with open(image_path, "rb") as f:
                return _predict_image_func(artifact_or_path, f)
        except Exception as e:
            raise e
    # Fallback stub
    return ("unknown", 0.0, None)

def call_predict_batch(artifact_or_path, image_paths):
    results = []
    if _predict_zip_func is not None:
        # Some users implement predict_zip(artifact, zip_or_folder)
        # We'll create a zip dynamically if needed, but simpler: loop per-image
        pass
    for p in image_paths:
        try:
            label, score, _ = call_predict_image(artifact_or_path, str(p))
            results.append({"file": p.name, "path": str(p), "label": label, "score": float(score)})
        except Exception as e:
            results.append({"file": p.name, "path": str(p), "label": "ERROR", "score": None, "error": str(e)})
    return results

# ======================
# Streamlit App
# ======================
st.set_page_config(page_title="Model App (Drive public cache + Uploaders)", page_icon="🧪")
st.title("🧪 Model App — Google Drive (public cache) + Uploaders")

# Controls
col1, col2, col3 = st.columns([1,1,2])
with col1:
    force_refresh = st.button("Force refresh model cache")
with col2:
    show_debug = st.checkbox("Show debug", value=False)
with col3:
    gdrive_input = st.text_input("Drive File ID or URL", value=GDRIVE_FILE_ID_OR_URL, help="Public file ID or link; set env GDRIVE_FILE_ID to auto-fill.")

if force_refresh and Path(MODEL_LOCAL_PATH).exists():
    try:
        Path(MODEL_LOCAL_PATH).unlink(missing_ok=True)
        st.info("Model cache cleared.")
    except Exception as e:
        st.warning(f"Could not clear cache: {e}")

# Ensure model available (download once if ID/URL supplied)
artifact_path = None
if gdrive_input:
    with st.spinner("Downloading model from Google Drive (public)…"):
        try:
            Path(MODEL_LOCAL_PATH).parent.mkdir(parents=True, exist_ok=True)
            artifact_path = download_gdrive_public_cached(gdrive_input, MODEL_LOCAL_PATH, expected_sha256=(MODEL_SHA256 or None))
            st.success(f"Model ready: {artifact_path}")
            if show_debug:
                st.caption(f"SHA-256: {sha256_file(artifact_path)}  •  Size: {Path(artifact_path).stat().st_size:,} bytes")
        except Exception as e:
            st.error(f"Model download failed: {e}")
            st.stop()
else:
    st.info("No Google Drive file ID/URL provided. You can still upload a model file below to use.")

# Allow manual model upload if user prefers
st.subheader("Upload model file (optional)")
upl_model = st.file_uploader("Model artifact (will overwrite local cache path)", type=["pkl","bin","pt","h5","keras","onnx","zip"], key="model_uploader")
if st.button("Save uploaded model as cache"):
    if not upl_model:
        st.warning("Please choose a model file to upload.")
    else:
        with st.spinner("Saving model artifact…"):
            Path(MODEL_LOCAL_PATH).parent.mkdir(parents=True, exist_ok=True)
            with open(MODEL_LOCAL_PATH, "wb") as f:
                f.write(upl_model.getbuffer())
            artifact_path = MODEL_LOCAL_PATH
            st.success(f"Saved model to {artifact_path}")

# Uploaders (same UX names/keys as common patterns in your earlier app)
st.markdown("---")
st.subheader("Single Image Prediction")
"""single_file = st.file_uploader("Upload an image", type=["png","jpg","jpeg","bmp","tif","tiff","webp"], key="single")
if st.button("Predict Single", type="primary"):
    if not single_file:
        st.warning("Please upload an image first.")
    elif not (artifact_path or Path(MODEL_LOCAL_PATH).exists()):
        st.error("Model file is not ready. Provide a Drive link or upload a model file.")
    else:
        # Ensure we have the path
        model_path = artifact_path or MODEL_LOCAL_PATH
        # Save uploaded image to disk and pass the PATH to predictor
        with st.spinner("Saving & running prediction…"):
            img_out = Path("uploads/single"); img_out.mkdir(parents=True, exist_ok=True)
            img_path = img_out / Path(single_file.name).name
            with open(img_path, "wb") as f:
                f.write(single_file.getbuffer())
            try:
                label, score, vis = call_predict_image(model_path, str(img_path))
                st.success(f"Prediction: {label} (score: {score:.4f})")
                if vis is not None:
                    st.image(vis, caption=f"{label} ({score:.3f})")
                else:
                    # If no visualization, still preview input
                    try:
                        if Image is not None:
                            st.image(str(img_path), caption="Input image")
                    except Exception:
                        pass
            except Exception as e:
                st.error(f"Prediction failed: {e}")

st.markdown("---")
st.subheader("Batch ZIP Predictions")
zip_file = st.file_uploader("Upload a .zip with images", type=["zip"], key="zip")
if st.button("Predict ZIP", type="primary"):
    if not zip_file:
        st.warning("Please upload a ZIP file.")
    elif not (artifact_path or Path(MODEL_LOCAL_PATH).exists()):
        st.error("Model file is not ready. Provide a Drive link or upload a model file.")
    else:
        with st.spinner("Extracting & running predictions…"):
            # Extract ZIP safely
            batch_dir = Path("uploads/batch"); batch_dir.mkdir(parents=True, exist_ok=True)
            tmp_zip = batch_dir / "batch.zip"
            with open(tmp_zip, "wb") as f:
                f.write(zip_file.getbuffer())
            image_paths = []
            try:
                with zipfile.ZipFile(tmp_zip, "r") as zf:
                    for info in zf.infolist():
                        if info.is_dir():
                            continue
                        name = info.filename
                        base = Path(name).name
                        # Skip __MACOSX, AppleDouble, dotfiles
                        if base.startswith(".") or name.startswith("__MACOSX/") or base.startswith("._"):
                            continue
                        ext = Path(base).suffix.lower()
                        if ext not in {".png",".jpg",".jpeg",".bmp",".tif",".tiff",".webp"}:
                            continue
                        out_path = batch_dir / base
                        with zf.open(info) as src, open(out_path, "wb") as dstf:
                            shutil.copyfileobj(src, dstf, length=_DEF_CHUNK)
                        image_paths.append(out_path)
            finally:
                try:
                    tmp_zip.unlink(missing_ok=True)
                except Exception:
                    pass

            if not image_paths:
                st.warning("No valid images found in the ZIP.")
            else:
                model_path = artifact_path or MODEL_LOCAL_PATH
                rows = []
                for p in image_paths:
                    try:
                        label, score, _ = call_predict_image(model_path, str(p))
                        rows.append((p.name, str(p), label, float(score)))
                    except Exception as e:
                        rows.append((p.name, str(p), "ERROR", None))
                import pandas as pd
                df = pd.DataFrame(rows, columns=["file","path","label","score"])
                st.dataframe(df, use_container_width=True)
                csv_bytes = df.to_csv(index=False).encode("utf-8")
                st.download_button("Download predictions CSV", data=csv_bytes, file_name="predictions.csv", mime="text/csv")"""

cfg_path = st.text_input('Config path', value='configs/config_local.yaml')
if not os.path.exists(cfg_path):
    st.error('Config file not found.')
    st.stop()
cfg, pre, wrapper = load_artifacts(cfg_path)
class_names = wrapper.class_names

tab1, tab2 = st.tabs(['Single Image Prediction', 'ZIP/Bulk Prediction'])

with tab1:
    st.subheader('Single Image Prediction')
    file = st.file_uploader('Upload an image', type=['png','jpg','jpeg','bmp','tiff','tif','webp'])
    if st.button('Predict Single', type='primary') and file:
        try:
            i, conf = preprocess_and_predict_file(pre, wrapper, file.getbuffer(), file.name)
            st.success(f'Prediction: {class_names[i]} (confidence {conf:.3f})')
        except Exception as e:
            st.error(f'Prediction failed: {e}')

with tab2:
    st.subheader('ZIP/Bulk Prediction')
    zip_file = st.file_uploader('Upload a ZIP of images', type=['zip'])
    if st.button('Predict ZIP', type='primary') and zip_file:
        try:
            with st.spinner("Processing CSV..."):
                td = tempfile.mkdtemp()
                extract_dir = os.path.join(td, 'images')
                os.makedirs(extract_dir, exist_ok=True)
                safe_extract_zip(io.BytesIO(zip_file.getbuffer()), extract_dir, max_files=cfg.streamlit.get('max_zip_extract_files', 3500))
                imgs = [os.path.join(extract_dir, f) for f in os.listdir(extract_dir) if os.path.splitext(f)[1].lower() in ['.png','.jpg','.jpeg','.bmp','.tiff','.tif','.webp']]
                if not imgs:
                    st.warning('No images found in ZIP.')
                else:
                    from src.run_pipeline import predict_files
                    out_csv = os.path.join(td, 'predictions.csv')
                    predict_files(cfg, imgs, out_csv)
                    df = pd.read_csv(out_csv)
                    st.download_button('Download CSV', data=df.to_csv(index=False), file_name='predictions.csv', mime='text/csv')
                    st.dataframe(df, use_container_width=True, height=520)
            st.success("CSV processed successfully ✅")
        except Exception as e:
            st.error(f'Batch prediction failed: {e}')

st.markdown("---")
st.caption("Tip: Provide a public Google Drive File ID/URL via the text input or env var GDRIVE_FILE_ID. The model will download once, then be reused from local cache. If Drive quota is hit, make a copy of the file to get a new File ID.")

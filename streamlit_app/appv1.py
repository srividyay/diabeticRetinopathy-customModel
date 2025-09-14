import sys
from pathlib import Path
import streamlit as st
import numpy as np
import pandas as pd
import io, os, zipfile, tempfile, pickle

ROOT = Path(__file__).resolve().parents[1]  # repo root (parent of streamlit_app/)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data_pipeline import ImagePreprocessor
from src.utils import safe_extract_zip

#pip install gdown
import gdown 
import os

def download_models():
    """Download models if not present"""
    model_urls = {
        'model.keras': 'https://drive.google.com/uc?id=1S33KAvMN1unUBDVB4oXBbB4Q9gWu8Phr'
        # etc.
    }

    if not os.path.exists('artifacts'):
        os.makedirs('artifacts')

    for filename, url in model_urls.items():
        filepath = f'artifacts/{filename}'
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            gdown.download(url, filepath, quiet=False)

st.set_page_config(page_title='Image Severity Classifier', layout='wide')

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

def app():
    # Call this before loading models
    download_models()
    st.title('Image Severity Classifier')
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

if __name__ == '__main__':
    app()
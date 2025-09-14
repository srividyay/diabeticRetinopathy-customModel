import os, io, gc, pickle
import numpy as np
import pandas as pd
from pathlib import Path

from .config import load_config
from .data_pipeline import ImagePreprocessor
from .logging_utils import setup_logger
from .utils import write_csv

def list_images(folder: str):
    exts = ('.png','.jpg','.jpeg','.bmp','.tiff','.tif','.webp')
    return [str(p) for p in Path(folder).glob('*') if p.suffix.lower() in exts]

def predict_files(cfg, files, out_csv):
    logger = setup_logger('run_pipeline', cfg.paths['artifacts_dir'])
    pre = ImagePreprocessor.load(cfg.paths['pipeline_pkl'])
    with open(cfg.paths['model_pkl'], 'rb') as f:
        wrapper = pickle.load(f)
    class_names = wrapper.class_names
    rows = []
    batch, names = [], []
    for fp in files:
        try:
            arr = pre.preprocess_path(fp)
            batch.append(arr); names.append(os.path.basename(fp))
        except Exception as e:
            logger.error(f'Failed to preprocess {fp}: {e}')
    if not batch:
        write_csv([], out_csv, ['Image','Predicted Class','Severity Score','Confidence']); return out_csv

    x = np.stack(batch, axis=0)
    idxs, probs = wrapper.predict(x)
    for name, i, prob in zip(names, idxs, probs):
        conf = float(np.max(prob))
        sev_map = cfg.severity.get('mapping', {})
        sev = sev_map.get(str(int(i)), round(conf*100,2))
        rows.append((name, str(class_names[int(i)]), sev, conf))
    write_csv(rows, out_csv, ['Image','Predicted Class','Severity Score','Confidence'])
    return out_csv

def run_bulk(config_path: str, input_folder: str, out_csv: str):
    cfg = load_config(config_path)
    files = list_images(input_folder)
    return predict_files(cfg, files, out_csv)

def run_single(config_path: str, input_path: str, out_csv: str):
    cfg = load_config(config_path)
    return predict_files(cfg, [input_path], out_csv)

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--mode', choices=['bulk','single'], required=True)
    ap.add_argument('--input', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    if args.mode == 'bulk':
        run_bulk(args.config, args.input, args.out)
    else:
        run_single(args.config, args.input, args.out)

import os, pickle
from dataclasses import dataclass
from typing import Tuple
import numpy as np
from PIL import Image
import cv2

@dataclass
class ImagePreprocessor:
    image_size: Tuple[int, int] = (512, 512)
    channels: int = 3
    normalize: bool = True
    contrast_clip_limit: float = 2.0
    histogram_equalize: bool = False

    def _open_image(self, path: str) -> Image.Image:
        return Image.open(path).convert('RGB')

    def preprocess_array(self, arr: np.ndarray) -> np.ndarray:
        arr = cv2.resize(arr, self.image_size, interpolation=cv2.INTER_AREA)
        if self.contrast_clip_limit and self.contrast_clip_limit > 0:
            lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=self.contrast_clip_limit, tileGridSize=(8,8))
            cl = clahe.apply(l)
            lab = cv2.merge((cl, a, b))
            arr = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        if self.histogram_equalize:
            img_yuv = cv2.cvtColor(arr, cv2.COLOR_RGB2YUV)
            img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
            arr = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
        if self.normalize:
            arr = arr.astype('float32') / 255.0
        return arr

    def preprocess_path(self, path: str) -> np.ndarray:
        #if not os.path.exists(path + '.png'):
        if not os.path.exists(path):
           print(f"[WARN] Skipping missing file: {path}")
           return
        img = self._open_image(path)
        
        arr = np.array(img)
        return self.preprocess_array(arr)

    def save_processed(self, src_path, dst_dir, class_name):
        from pathlib import Path
        src_path = str(src_path)  # ensure string
        dst_dir = Path(dst_dir)
        dst_class_dir = dst_dir / str(class_name)
        dst_class_dir.mkdir(parents=True, exist_ok=True)

        try:
            arr = self.preprocess_path(src_path)   # ⬅️ use actual file path
            if arr is None:
                return None
        except FileNotFoundError:
            # log and skip gracefully
            print(f"[WARN] File not found in data_pipeline -> : {src_path}")
            return None
        
        arr_u8 = (arr * 255).astype("uint8") if type(arr) != "uint8" else arr
        out_path = dst_class_dir / (Path(src_path).stem + ".jpg")
        Image.fromarray(arr_u8).save(out_path)
        return str(out_path)

    def save(self, pkl_path: str):
        with open(pkl_path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(pkl_path: str) -> 'ImagePreprocessor':
        with open(pkl_path, 'rb') as f:
            return pickle.load(f)

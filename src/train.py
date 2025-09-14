import os
print("ENV CHECK:", {k: os.environ.get(k) for k in [
    "MPLBACKEND","MPLCONFIGDIR","TF_CPP_MIN_LOG_LEVEL",
    "OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","OBJC_DISABLE_INITIALIZE_FORK_SAFETY",
    "VECLIB_MAXIMUM_THREADS","MKL_NUM_THREADS"
]})

import logging
import os, gc, pickle, random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from .config import load_config
from .gpu_memory import configure_gpu_memory
from .data_pipeline import ImagePreprocessor
from .model import simpleModel, ModelWrapper
from .plots import plot_history, plot_confusion_by_dataset
from .logging_utils import setup_logger

def build_dataset(image_paths, labels, pre: ImagePreprocessor, batch_size: int, shuffle=True):
    def gen():
        for p, y in zip(image_paths, labels):
            arr = pre.preprocess_path(p)
            yield arr, y
    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(pre.image_size[0], pre.image_size[1], 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        )
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=len(image_paths))
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def save_sample_previews(df, pre, raw_dir, out_dir, k=3):
    os.makedirs(out_dir, exist_ok=True)
    samples = df.sample(min(k, len(df)), random_state=0)
    for i, row in samples.iterrows():
        raw_path = os.path.join(raw_dir, row['id_code'], '.png')
        if not os.path.exists(raw_path):
            print(f"[WARN] File not found : {raw_path}")
            continue
        # raw
        try:
            img = plt.imread(raw_path + '.png')
            plt.figure()
            plt.imshow(img)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f'sample_raw_{i}.png'))
            plt.close()
            # processed
            arr = pre.preprocess_path(raw_path)
            arr8 = (arr*255).astype('uint8')
            plt.figure()
            plt.imshow(arr8)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f'sample_processed_{i}.png'))
            plt.close()
        except FileNotFoundError:
            # log and skip gracefully
            print(f"[WARN] File not found in data_pipeline -> save_sample_previews: {raw_path}")
            return None

def main(config_path: str, raw_override: str=None, csv_override: str=None):
    cfg = load_config(config_path)
    logger = setup_logger('train', cfg.paths['outputs_dir'])
    configure_gpu_memory(cfg.gpu.get('enable_memory_growth', True))

    raw_dir = Path(raw_override or cfg.paths['raw_images_dir'])
    csv_path = Path(csv_override or cfg.paths['csv_labels_path'])
    proc_dir = Path(cfg.paths['processed_images_dir'])
    artifacts_dir = Path(cfg.paths['artifacts_dir'])
    plots_dir = Path(cfg.paths['plots_dir'])

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    df['filepath'] = df['id_code'].apply(lambda x: str(raw_dir / x))#.apply(lambda x: str(x + '.jpg'))
    class_names = sorted(df['diagnosis'].unique())
    class_to_idx = {c:i for i,c in enumerate(class_names)}
    df['label'] = df['diagnosis'].map(class_to_idx)
    CLASS_NAME_MAP = {
        0: "No DR",
        1: "Mild DR",
        2: "Moderate DR",
        3: "Severe DR",
        4: "Proliferative DR"
    }   
    #df["label"] = df["diagnosis"].map(CLASS_NAME_MAP)
    # Class distribution plot
    dist = df['diagnosis'].value_counts().reindex(class_names).fillna(0)
    plt.figure()
    dist.plot(kind='bar')
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(plots_dir / 'class_distribution.png'); plt.close()

    # Preprocessor
    pre = ImagePreprocessor(
        image_size=tuple(cfg.data['image_size']),
        channels=cfg.data['channels'],
        normalize=cfg.data['normalize'],
        contrast_clip_limit=cfg.data['contrast_clip_limit'],
        histogram_equalize=cfg.data.get('histogram_equalize', False),
    )
    pre.save(cfg.paths['pipeline_pkl'])
    logger.info(f"Saved preprocessor to {cfg.paths['pipeline_pkl']}")

    # Save processed images into class folders
    for _, r in df.iterrows():
        file_path = Path(r["filepath"] + '.jpg')
        if not os.path.exists(file_path):
            print(f"[WARN] Skipping missing file -> train -> before calling save_processed : {r['filepath']}")
            continue
        
        pre.save_processed(file_path, cfg.paths["processed_images_dir"], r["label"])

    print("="*50)

    # Save sample previews before/after
    save_sample_previews(df, pre, str(raw_dir), str(plots_dir))

    # Split
    train_df, val_df = train_test_split(df, test_size=cfg.data['valid_split'], stratify=df['label'], random_state=cfg.random_seed)
    train_df, test_df = train_test_split(df, test_size=cfg.data['valid_split'], stratify=df['label'], random_state=cfg.random_seed)

    train_df['filepath'] = train_df['filepath'].apply(lambda x: str(x + '.jpg'))
    val_df['filepath'] = val_df['filepath'].apply(lambda x: str(x + '.jpg'))
    test_df['filepath'] = val_df['filepath'].apply(lambda x: str(x + '.jpg'))
    
    # Datasets
    train_ds = build_dataset(train_df['filepath'].tolist(), train_df['label'].tolist(), pre, batch_size=cfg.data['batch_size'], shuffle=True)
    val_ds   = build_dataset(val_df['filepath'].tolist(),   val_df['label'].tolist(),   pre, batch_size=cfg.data['batch_size'], shuffle=False)
    test_ds = build_dataset(test_df['filepath'].tolist(),   test_df['label'].tolist(),   pre, batch_size=cfg.data['batch_size'], shuffle=False)

    # -------------------------------------------------------------------------------------------------------------------------------------------- #

    # ---------------------------------
    # Your existing dataset & augmenter
    # ---------------------------------

    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.1, 0.1)
    ], name="DataAugmentation")

    AUTOTUNE = tf.data.AUTOTUNE
    class_names = train_ds.class_names
    num_classes = len(class_names)

    # -------------------------------------------------------
    # 1) Count class frequencies in the (batched) train_ds
    # -------------------------------------------------------
    counts = np.zeros(num_classes, dtype=np.int64)

    for _, y in train_ds.unbatch():
        idx = int(tf.squeeze(y).numpy())
        counts[idx] += 1

    print("Per-class counts:", dict(zip(class_names, counts)))

    # -----------------------------------------------------------------
    # 2) Choose minority classes (configurable). Example: below 80% of
    #    the majority-class count (you can tighten/loosen this ratio).
    # -----------------------------------------------------------------
    MINORITY_RATIO = 0.80
    max_count = counts.max()
    minority_ids = np.where(counts < MINORITY_RATIO * max_count)[0]
    print("Minority class indices:", minority_ids, "->",
        [class_names[i] for i in minority_ids])

    # If none are minority, nothing to augment
    minority_ids_tf = tf.constant(minority_ids, dtype=tf.int32)

    # -----------------------------------------------------------------
    # 3) Apply augmentation ONLY to minority classes in batched pipeline
    # -----------------------------------------------------------------
    def augment_minority_batch(x, y):
        """x: (B,H,W,C), y: (B,) or (B,num_classes) depending on label_mode"""
        # Resolve class id per item in the batch
        y_idx = tf.cast(tf.squeeze(y), tf.int32)                     # (B,)

        def _apply():
            # mask: True where class is minority
            mask = tf.reduce_any(
                tf.equal(tf.expand_dims(y_idx, -1), minority_ids_tf), axis=-1
            )  # (B,)

            x_aug = data_augmentation(x, training=True)

            # Broadcast mask to (B,1,1,1) and select per item
            mask4d = tf.reshape(mask, (-1, 1, 1, 1))
            x_out = tf.where(mask4d, x_aug, x)
            return x_out, y

        # If no minority classes, skip augmentation
        return tf.cond(tf.size(minority_ids_tf) > 0, _apply, lambda: (x, y))

    # Build the final training pipeline
    train_ds = train_ds.map(augment_minority_batch, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.prefetch(AUTOTUNE)

    # -------------------------------------------------------------------------------------------------------------------------------------------- #
    # Model (provided simpleModel with 5 outputs)
    #Saving tested original simpleModel inputs
    #model = simpleModel(input_shape=(cfg.data['image_size'][0], cfg.data['image_size'][1], cfg.data['channels']), num_classes=len(class_names))
    #New simpleModel with augmentation enabled
    model = simpleModel(input_shape=(cfg.data['image_size'][0], cfg.data['image_size'][1], cfg.data['channels']), num_classes=len(class_names))
    opt = keras.optimizers.Adam(learning_rate=cfg.training['base_learning_rate'])
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    callbacks=[
        keras.callbacks.EarlyStopping(patience=cfg.training['patience'], restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(cfg.paths['model_path'], save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(patience=max(1, cfg.training['patience']-1)),
    ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=cfg.training['epochs'], verbose=2, callbacks=callbacks)
    plot_history(history, plots_dir)
    
    # Confusion matrix on val
    x_val = np.stack([pre.preprocess_path(p) for p in val_df['filepath']], axis=0)
    probs = model.predict(x_val, verbose=0)
    y_pred = np.argmax(probs, axis=1)
    plot_confusion_by_dataset(val_df['label'].to_numpy(), y_pred, class_names, plots_dir, "validation")

    # Evaluate the model
    # Calculate test accuracy
    test_accuracy = model.evaluate(test_ds, verbose=1)
    print(f"Test Accuracy: {test_accuracy:.3f} ({test_accuracy*100:.1f}%)")
    x_test = np.stack([pre.preprocess_path(p) for p in test_df['filepath']], axis=0)
    probs = model.predict(x_test, verbose=0)
    y_pred_test = np.argmax(probs, axis=1)
    plot_confusion_by_dataset(test_df['label'].to_numpy(), y_pred_test, class_names, plots_dir, "test")

    # Save wrapper pkl
    wrapper = ModelWrapper(cfg.paths['model_path'], class_names)
    with open(cfg.paths['model_pkl'], 'wb') as f:
        pickle.dump(wrapper, f)
    logger.info(f"Saved model wrapper to {cfg.paths['model_pkl']}")

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--raw', default=None)  # optional override
    ap.add_argument('--csv', default=None)  # optional override
    args = ap.parse_args()
    main(args.config, args.raw, args.csv)

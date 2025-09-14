import tensorflow as tf

def configure_gpu_memory(enable_growth: bool = True):
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus and enable_growth:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print(f"[GPU] Skipping memory growth config: {e}")

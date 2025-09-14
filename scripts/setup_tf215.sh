#!/usr/bin/env bash
set -euo pipefail

PY310=${PY310:-python3.10}
VENV_DIR=${VENV_DIR:-.venv}

echo ">> Creating venv with ${PY310}"
${PY310} -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

echo ">> Upgrading pip/setuptools/wheel"
python -m pip install --upgrade pip setuptools wheel

ARCH=$(uname -m)
if [[ "${ARCH}" == "arm64" ]]; then
  echo ">> Apple Silicon detected: installing tensorflow-macos + tensorflow-metal"
  python -m pip uninstall -y keras || true
  python -m pip install "tensorflow-macos==2.15.*" "tensorflow-metal==1.*" "numpy<2" h5py
else
  echo ">> Intel/Other: installing tensorflow"
  python -m pip uninstall -y keras || true
  python -m pip install "tensorflow==2.15.*" "numpy<2" h5py
fi

echo ">> Verifying import"
python - <<'PY'
import numpy as np
import tensorflow as tf
from tensorflow import keras
print("TF:", tf.__version__)
print("Keras (tf.keras):", keras.__version__)
print("NumPy:", np.__version__)
PY

echo ">> Done. Activate with: source ${VENV_DIR}/bin/activate"
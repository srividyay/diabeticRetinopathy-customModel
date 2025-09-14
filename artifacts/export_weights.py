from tensorflow import keras

# Load the full model (if you only have model.keras)
model = keras.models.load_model("model.keras")

# Save only the weights (much lighter than full SavedModel)
model.save_weights("model_weights.h5")   # HDF5 format
# Or TensorFlow checkpoint format
# model.save_weights("model_weights")}
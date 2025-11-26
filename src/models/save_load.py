def save_model(model, file_path, model_name):
    model.save(f"{file_path}/{model_name}.h5")


def load_model(file_path, model_name):
    from tensorflow.keras.models import load_model as keras_load_model

    return keras_load_model(f"{file_path}/{model_name}.h5")

import numpy as np


def data_loader(model_name):
    x_train = np.load(f"artifacts/processed data/{model_name}/x_train.npy")
    y_train = np.load(f"artifacts/processed data/{model_name}/y_train.npy")
    x_test = np.load(f"artifacts/processed data/{model_name}/x_test.npy")
    y_test = np.load(f"artifacts/processed data/{model_name}/y_test.npy")

    return x_train, y_train, x_test, y_test

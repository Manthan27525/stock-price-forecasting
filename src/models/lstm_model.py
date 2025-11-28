from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
import keras_tuner as kt


def build_lstm_model(hp):
    model = Sequential()
    for i in range(hp.Int("num_layers", 1, 10)):
        model.add(
            LSTM(
                units=hp.Int(f"units_{i}", min_value=32, max_value=256, step=32),
                return_sequences=(i < hp.Int("num_layers", 1, 3) - 1),
            )
        )
        model.add(
            Dropout(
                rate=hp.Float(f"dropout_{i}", min_value=0.0, max_value=0.5, step=0.1)
            )
        )

    model.add(Dense(units=1))
    model.compile(
        optimizer=hp.Choice("optimizer", ["adam", "rmsprop"]), loss="mean_squared_error"
    )

    return model


def tuned_lstm_model(x_train, y_train, x_test, y_test):
    tuner = kt.RandomSearch(
        build_lstm_model,
        objective="val_accuracy",
        max_trials=5,
        executions_per_trial=3,
        directory="lstm_tuning",
        project_name="stock_price_forecasting",
    )

    tuner.search(x_train, y_train, epochs=50, validation_data=(x_test, y_test))
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    model = build_lstm_model(best_hps)

    return model


def simple_lstm_model(x_train, y_train, x_test, y_test):
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    model = Sequential()
    model.add(Input(shape=(x_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(50, return_sequences=True))  # <-- FIXED
    model.add(Dropout(0.2))

    model.add(LSTM(50, return_sequences=False))  # Last LSTM → False
    model.add(Dropout(0.2))

    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mean_squared_error")

    return model

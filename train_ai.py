from keras import losses
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from keras.utils import plot_model

from generate_training_data import generate_training_data

if __name__ == '__main__':
    train_data_size = 100
    train_data = np.array([generate_training_data(1, 1) for _ in range(train_data_size)])
    data_labels = [0] * train_data_size

    model = Sequential()

    # Input layer dimension is 44. Each option is 22 elements and there are two options.
    model.add(Dense(units=44, activation='relu', input_dim=44))

    model.add(Dense(units=22, activation='relu'))

    # Output layer dimension is 1 because it outputs 0 for first option and 1 for second option.
    model.add(Dense(units=1, activation='softmax'))

    plot_model(model, to_file="model.png", show_shapes=True, show_layer_names=True)

    model.compile(loss=losses.mean_squared_error,
                  optimizer='sgd',
                  metrics=['accuracy'])

    model.fit(train_data, data_labels, epochs=5, batch_size=32)

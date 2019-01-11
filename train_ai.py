from keras import losses, optimizers
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import plot_model

from generate_training_data import generate_training_data

if __name__ == '__main__':
    train_data_size = 100
    max_num_people_per_option = 1
    (train_data, train_labels) = generate_training_data(max_num_people_per_option, train_data_size)

    model = Sequential()

    # Input layer dimension is 44. Each option is 22 elements and there are two options.
    model.add(Dense(units=44, activation='relu', input_dim=44))

    model.add(Dense(units=22, activation='relu'))

    # Output layer dimension is 2 (class 1 is first_option and class 2 is second_option)
    model.add(Dense(units=2, activation='softmax'))

    plot_model(model, to_file="model.png", show_shapes=True, show_layer_names=True)

    model.compile(loss=losses.categorical_crossentropy,
                  optimizer='sgd',
                  metrics=['accuracy'])

    model.fit(train_data, train_labels, epochs=5, batch_size=32)

import jsonpickle as jsonpickle
from keras import losses, metrics
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import plot_model
import matplotlib.pyplot as plt

from generate_training_data import generate_training_data, TrainMetadata


def write_data_to_file(metadata: TrainMetadata, type: str):
    (data, labels) = generate_training_data(metadata.max_num_people_per_option,
                                            metadata.train_data_size)

    data_file = open(type + "_data", "w")
    data_file.write(jsonpickle.encode(data))
    data_file.close()

    labels_file = open(type + "_labels", "w")
    labels_file.write(jsonpickle.encode(labels))
    labels_file.close()

    metadata_file = open(type + "_metadata", "w")
    metadata_file.write(jsonpickle.encode(metadata))
    metadata_file.close()


def read_data_from_file(type: str):
    data_file = open(type + "_data", "r")
    data = jsonpickle.decode(data_file.read())
    data_file.close()

    labels_file = open(type + "_labels", "r")
    labels = jsonpickle.decode(labels_file.read())
    labels_file.close()

    metadata_file = open(type + "_metadata", "r")
    metadata = jsonpickle.decode(metadata_file.read())
    metadata_file.close()

    return data, labels, metadata


def write_training_data_to_file(train_metadata: TrainMetadata):
    return write_data_to_file(train_metadata, "train")


def read_training_data_from_file():
    return read_data_from_file("train")


def write_test_data_to_file(test_metadata: TrainMetadata):
    return write_data_to_file(test_metadata, "test")


def read_test_data_from_file():
    return read_data_from_file("test")


if __name__ == '__main__':
    # write_training_data_to_file(TrainMetadata(100000, 10))
    # write_test_data_to_file(TrainMetadata(100000, 10))

    (train_data, train_labels, train_metadata) = read_training_data_from_file()
    (test_data, test_labels, test_metadata) = read_test_data_from_file()

    model = Sequential()

    input_dim = 44 * train_metadata.max_num_people_per_option
    output_dim = 2

    # Input layer dimension is 44 * max_num_people_per_option because each option is 22 elements
    # and there are two options, and each option is padded to the max number of people.
    model.add(Dense(units=44 * train_metadata.max_num_people_per_option, activation='relu',
                    input_dim=input_dim))

    model.add(Dense(units=round((input_dim + output_dim) / 2), activation='relu'))

    # Output layer dimension is 2 (class 1 is first_option and class 2 is second_option).
    model.add(Dense(units=output_dim, activation='softmax'))

    plot_model(model, to_file="model.png", show_shapes=True, show_layer_names=True)

    model.compile(loss=losses.categorical_crossentropy,
                  optimizer='sgd',
                  metrics=[metrics.categorical_accuracy])

    model.fit(train_data, train_labels, epochs=1, batch_size=32)

    (loss, accuracy) = model.evaluate(test_data, test_labels, batch_size=32)
    print("Loss:")
    print(loss)
    print("Accuracy:")
    print(accuracy)

    print("Predictions:")
    print(model.predict(test_data))
    print("Expected:")
    print(test_labels)

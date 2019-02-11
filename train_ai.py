from keras import losses, metrics
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import plot_model
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # required

from generate_data_pgmpy import DilemmaGenerator
from manage_data import TrainMetadata, write_data_to_file, read_data_from_file, \
    preprocess_data_before_saving


def plot_data():
    x = np.array([100, 80, 50])
    y = np.array([90, 85, 80])
    X, Y = np.meshgrid(x, y)
    Z = np.array(
        [
            [.74124, .8818, .99996],
            [.99992, .99998, 1],
            [1, 1, 1]
        ]
    )

    contours = plt.contour(X, Y, Z)
    plt.clabel(contours, inline=True, inline_spacing=5)
    plt.colorbar()

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
    ax.view_init(30, 30)

    plt.show()


def generate_training_data_in_memory(metadata: TrainMetadata, generators):
    data, labels = preprocess_data_before_saving(metadata, generators)
    return data, labels, metadata


def train_and_test_iteration(filename, train_data, train_labels, train_metadata, test_data,
                             test_labels, test_metadata):
    model = Sequential()

    # 22 elements per option, 3 options, each option padded to max number of people
    output_dim = 2
    input_dim = 22 * output_dim * train_metadata.max_num_people_per_option

    model.add(Dense(units=input_dim, activation='relu',
                    input_dim=input_dim))

    model.add(Dense(units=round((input_dim + output_dim) / 2), activation='relu'))

    # Output layer dimension is 2 (class 1 is first_option and class 2 is second_option).
    model.add(Dense(units=output_dim, activation='softmax'))

    plot_model(model, to_file="model.png", show_shapes=True, show_layer_names=True)

    model.compile(loss=losses.categorical_crossentropy,
                  optimizer='sgd',
                  metrics=[metrics.categorical_accuracy])

    model.fit(train_data, train_labels, epochs=5, batch_size=32)

    (loss, accuracy) = model.evaluate(test_data, test_labels, batch_size=32)
    print("Loss:")
    print(loss)
    print("Accuracy:")
    print(accuracy)

    print("Predictions:")
    print(model.predict(test_data))
    print("Expected:")
    print(test_labels)

    with open(filename, "a") as f:
        f.write("%s,%s\n" % (loss, accuracy))


def train_and_test(filename, option_cpd, jaywalking_cpd, test_data, test_labels, test_metadata):
    generators = [
        DilemmaGenerator(
            option_vals=[
                [option_cpd[0], option_cpd[1]]
            ],
            jaywalking_vals=[
                [jaywalking_cpd[0], jaywalking_cpd[1]],
                [jaywalking_cpd[1], jaywalking_cpd[0]]
            ]
        ),
        DilemmaGenerator(
            option_vals=[
                [option_cpd[1], option_cpd[0]]
            ],
            jaywalking_vals=[
                [jaywalking_cpd[1], jaywalking_cpd[0]],
                [jaywalking_cpd[0], jaywalking_cpd[1]]
            ]
        )
    ]

    train_data, train_labels, train_metadata = generate_training_data_in_memory(
        TrainMetadata(50000, 10), generators)

    with open(filename, "a") as f:
        f.write("%s,%s,%s,%s\n" % (option_cpd[0], option_cpd[1], jaywalking_cpd[0],
                                   jaywalking_cpd[1]))

    for _ in range(5):
        train_and_test_iteration(filename, train_data, train_labels, train_metadata,
                                 test_data, test_labels, test_metadata)


if __name__ == '__main__':
    test_data_filename = "test option 40-60 jaywalking 100-0 0-100"
    results_filename = "dense results for " + test_data_filename

    test_data, test_labels, test_metadata = read_data_from_file(test_data_filename)

    for first_option_probability in np.linspace(0, 1, 10):
        option_cpd = [first_option_probability, 1 - first_option_probability]
        for jaywalking_probability in np.linspace(0, 1, 10):
            jaywalking_cpd = [jaywalking_probability, 1 - jaywalking_probability]
            train_and_test(results_filename, option_cpd, jaywalking_cpd, test_data, test_labels,
                           test_metadata)

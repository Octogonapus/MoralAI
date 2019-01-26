from keras import losses, metrics
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import plot_model

from generate_data_pgmpy import DilemmaGenerator
from manage_data import TrainMetadata, write_data_to_file, read_data_from_file

if __name__ == '__main__':
    # write_data_to_file(TrainMetadata(50000, 10), [
    #     DilemmaGenerator(
    #         option_vals=[
    #             [0.8, 0.2]
    #         ],
    #         jaywalking_vals=[
    #             [0.5, 0.5],
    #             [0.5, 0.5]
    #         ]
    #     ),
    #     # [0.8, 0.2],
    #     # [0.2, 0.8]
    #     DilemmaGenerator(
    #         option_vals=[
    #             [0.2, 0.8]
    #         ],
    #         jaywalking_vals=[
    #             [0.5, 0.5],
    #             [0.5, 0.5]
    #         ]
    #     )
    #     # [0.2, 0.8],
    #     # [0.8, 0.2]
    # ], "train 80-20 50-50 50-50 and 20-80 50-50 50-50")

    # write_data_to_file(TrainMetadata(50000, 10), [
    #     DilemmaGenerator(
    #         option_vals=[
    #             [0.4, 0.6]
    #         ],
    #         jaywalking_vals=[
    #             [1, 0],
    #             [0, 1]
    #         ]
    #     )
    # ], "test option 40-60 jaywalking 100-0 0-100")

    (train_data, train_labels, train_metadata) = read_data_from_file(
        "train 80-20 50-50 50-50 and 20-80 50-50 50-50")
    (test_data, test_labels, test_metadata) = read_data_from_file(
        "test option 40-60 jaywalking 100-0 0-100")

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

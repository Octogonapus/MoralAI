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
    #             [0.8, 0.1, 0.1]
    #         ],
    #         jaywalking_vals=[
    #             [0.5, 0.5, 0.5],
    #             [0.5, 0.5, 0.5]
    #         ]
    #     ),
    #     DilemmaGenerator(
    #         option_vals=[
    #             [0.1, 0.1, 0.8]
    #         ],
    #         jaywalking_vals=[
    #             [0.5, 0.5, 0.5],
    #             [0.5, 0.5, 0.5]
    #         ]
    #     )
    # ], "train 80-10-10 50-50 50-50 50-50 and 10-10-80 50-50 50-50 50-50")

    # write_data_to_file(TrainMetadata(50000, 10), [
    #     DilemmaGenerator(
    #         option_vals=[
    #             [0.4, 0.3, 0.3]
    #         ],
    #         jaywalking_vals=[
    #             [0, 1, 1],
    #             [1, 0, 0]
    #         ]
    #     )
    # ], "test 40-30-30 0-100 100-0 100-0")

    (train_data, train_labels, train_metadata) = read_data_from_file(
        "train 80-10-10 50-50 50-50 50-50 and 10-10-80 50-50 50-50 50-50")
    (test_data, test_labels, test_metadata) = read_data_from_file(
        "test 40-30-30 0-100 100-0 100-0")

    model = Sequential()

    # 22 elements per option, 3 options, each option padded to max number of people
    output_dim = 3
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

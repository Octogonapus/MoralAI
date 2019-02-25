# This file is part of MoralAI.
#
# MoralAI is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# MoralAI is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with MoralAI.  If not, see <https://www.gnu.org/licenses/>.

import json

from keras import losses, metrics
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import plot_model
import numpy as np

from model import create_dilemma_from_export
from generate_data_pgmpy import DilemmaGenerator
from manage_data import TrainMetadata, read_data_from_file, preprocess_data_before_saving


def generate_training_data_in_memory(metadata: TrainMetadata, generators):
    data, labels = preprocess_data_before_saving(metadata, generators)
    return data, labels, metadata


def train_and_test_iteration(train_data, train_labels, train_metadata, test_data, test_labels,
                             test_metadata):
    model = Sequential()

    # 22 elements per option, 3 options, each option padded to max number of people
    output_dim = 2
    input_dim = 22 * output_dim * train_metadata.max_num_people_per_option

    model.add(Dense(units=input_dim, activation='relu', input_dim=input_dim))

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

    predictions = model.predict(test_data)
    print("Predictions:")
    print(predictions)
    print("Expected:")
    print(test_labels)

    num_jaywalkers_when_wrong = 0  # number of jaywalkers when the ai classified incorrectly
    num_jaywalkers = 0  # number of jaywalkers total

    for i in range(len(test_labels)):
        prediction_label = predictions[i].tolist()
        test_label = test_labels[i].tolist()

        prediction_index = prediction_label.index(max(prediction_label))
        test_index = test_label.index(max(test_label))

        dilemma = create_dilemma_from_export(test_data[i].tolist(), 2,
                                             test_metadata.max_num_people_per_option)

        for option in dilemma.options:
            for person in option:
                if person.jaywalking:
                    num_jaywalkers += 1

        if prediction_index != test_index:
            for person in dilemma.options[prediction_index]:
                if person.jaywalking:
                    num_jaywalkers_when_wrong += 1

    return loss, accuracy, num_jaywalkers, num_jaywalkers_when_wrong


def train_and_test(option_cpd, jaywalking_cpd, test_data, test_labels, test_metadata):
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
        TrainMetadata(50000, 10),
        generators
    )

    test_results_json = []
    for _ in range(5):
        loss, accuracy, num_jaywalkers, num_jaywalkers_when_wrong = train_and_test_iteration(
            train_data, train_labels, train_metadata, test_data, test_labels, test_metadata
        )

        test_results_json.append({
            "loss": loss,
            "accuracy": accuracy,
            "num_jaywalkers": num_jaywalkers,
            "prob_jaywalking_when_wrong": num_jaywalkers_when_wrong / num_jaywalkers
        })

    return test_results_json


if __name__ == '__main__':
    # generators = [
    #     DilemmaGenerator(
    #         option_vals=[
    #             [0.4, 0.6]
    #         ],
    #         jaywalking_vals=[
    #             [0, 1],
    #             [1, 0]
    #         ]
    #     ),
    #     DilemmaGenerator(
    #         option_vals=[
    #             [0.6, 0.4]
    #         ],
    #         jaywalking_vals=[
    #             [1, 0],
    #             [0, 1]
    #         ]
    #     )
    # ]
    #
    # write_data_to_file(TrainMetadata(50000, 10), generators,
    #                    "test 40-60 0-100 100-0")

    test_data_filename = "test 40-60 0-100 100-0"
    results_filename = "dense results for " + test_data_filename

    test_data, test_labels, test_metadata = read_data_from_file(test_data_filename)

    # 2 options
    num_people = len(test_labels) * 2 * test_metadata.max_num_people_per_option

    option_level_results = []
    for first_option_probability in np.linspace(0, 1, 10):
        option_cpd = [first_option_probability, 1 - first_option_probability]

        jaywalking_level_results = []
        for jaywalking_probability in np.append(np.linspace(0, 3 / 10, 5),
                                                np.linspace(7 / 10, 1, 5)):
            jaywalking_cpd = [jaywalking_probability, 1 - jaywalking_probability]
            jaywalking_level_results.append({
                "jaywalking_cpd": jaywalking_cpd,
                "results": train_and_test(option_cpd, jaywalking_cpd, test_data, test_labels,
                                          test_metadata)
            })

        option_level_results.append({
            "option_cpd": option_cpd,
            "jaywalking_level_results": jaywalking_level_results
        })

    with open(results_filename, "a") as f:
        f.write(json.dumps({
            "num_people": num_people,
            "option_level_results": option_level_results
        }))

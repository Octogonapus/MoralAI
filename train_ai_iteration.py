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

from keras import Sequential, losses, metrics
from keras.layers import Dense
from keras.utils import plot_model

from generate_data_pgmpy import DilemmaGenerator
from manage_data import TrainMetadata, preprocess_data_before_saving, read_data_from_file
from model import create_dilemma_from_export


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


def train_and_test(test_data_filename, option_cpd, jaywalking_cpd):
    test_data, test_labels, test_metadata = read_data_from_file(test_data_filename)

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

    with open("results from iteration for " + test_data_filename, "a") as f:
        f.write(json.dumps(
            test_results_json
        ))

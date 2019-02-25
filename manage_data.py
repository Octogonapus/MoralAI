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

import jsonpickle
import numpy as np

from generate_training_data import generate_training_data


class TrainMetadata:

    def __init__(self, train_data_size: int, max_num_people_per_option: int):
        self.train_data_size = train_data_size
        self.max_num_people_per_option = max_num_people_per_option


def preprocess_data_before_saving(metadata: TrainMetadata, generators):
    data_tuples = [
        generate_training_data(generator,
                               metadata.max_num_people_per_option,
                               metadata.train_data_size)
        for generator in generators
    ]

    data = []
    for item in [x[0] for x in data_tuples]:
        [data.append(x) for x in item]

    labels = []
    for item in [x[1] for x in data_tuples]:
        [labels.append(x) for x in item]

    return np.array(data), np.array(labels)


def write_data_to_file(metadata: TrainMetadata, generators, name: str):
    data, labels = preprocess_data_before_saving(metadata, generators)

    data_file = open(name + "_data", "w")
    data_file.write(jsonpickle.encode(data))
    data_file.close()

    labels_file = open(name + "_labels", "w")
    labels_file.write(jsonpickle.encode(labels))
    labels_file.close()

    metadata_file = open(name + "_metadata", "w")
    metadata_file.write(jsonpickle.encode(metadata))
    metadata_file.close()


def read_data_from_file(name: str):
    data_file = open(name + "_data", "r")
    data = jsonpickle.decode(data_file.read())
    data_file.close()

    labels_file = open(name + "_labels", "r")
    labels = jsonpickle.decode(labels_file.read())
    labels_file.close()

    metadata_file = open(name + "_metadata", "r")
    metadata = jsonpickle.decode(metadata_file.read())
    metadata_file.close()

    return data, labels, metadata

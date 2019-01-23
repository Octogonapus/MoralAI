import itertools

import jsonpickle
import numpy as np

from generate_training_data import generate_training_data


class TrainMetadata:

    def __init__(self, train_data_size: int, max_num_people_per_option: int):
        self.train_data_size = train_data_size
        self.max_num_people_per_option = max_num_people_per_option


def write_data_to_file(metadata: TrainMetadata, generators, name: str):
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

    data_file = open(name + "_data", "w")
    data_file.write(jsonpickle.encode(np.array(data)))
    data_file.close()

    labels_file = open(name + "_labels", "w")
    labels_file.write(jsonpickle.encode(np.array(labels)))
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

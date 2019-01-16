import jsonpickle

from generate_data_pgmpy import DilemmaGenerator
from generate_training_data import generate_training_data


class TrainMetadata:

    def __init__(self, train_data_size: int, max_num_people_per_option: int):
        self.train_data_size = train_data_size
        self.max_num_people_per_option = max_num_people_per_option


def write_data_to_file(metadata: TrainMetadata, generator: DilemmaGenerator, type: str):
    (data, labels) = generate_training_data(generator,
                                            metadata.max_num_people_per_option,
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

import numpy as np

from generate_data_pgmpy import DilemmaGenerator


def generate_modeled_dilemmas(generator, max_num_people: int, num_dilemmas: int):
    dilemmas = [generator.generate_dilemma(max_num_people) for _ in range(num_dilemmas)]
    return list(map(lambda x: (x[0].export_raw(), x[1]), dilemmas))


def generate_training_data(generator, max_num_people: int, num_dilemmas: int):
    data_and_labels = generate_modeled_dilemmas(generator, max_num_people, num_dilemmas)
    data = [np.array(x[0]) for x in data_and_labels]
    labels = [np.array(x[1]) for x in data_and_labels]
    return np.array(data), np.array(labels)

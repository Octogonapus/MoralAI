import random
import numpy as np

from model import Person, Dilemma, Race, LegalSex


def generate_random_person():
    return Person(
        age=random.randint(1, 60),
        race=Race(random.randint(1, 5)),
        legal_sex=LegalSex(random.randint(1, 2)),
        jaywalking=bool(random.getrandbits(1)),
        driving_under_the_influence=bool(random.getrandbits(1))
    )


def generate_empty_person():
    return Person()


def generate_training_data_raw(first_option_size: int, second_option_size: int, max_size: int):
    first_option = [generate_empty_person() for _ in range(first_option_size)]
    second_option = [generate_empty_person() for _ in range(second_option_size)]
    dilemma = Dilemma(first_option, second_option, max_size)
    label = [1, 0] if first_option_size >= second_option_size else [0, 1]
    return dilemma.export_raw(), label


def generate_training_data(max_num_people: int, num_data_points: int):
    data_and_labels = [
        generate_training_data_raw(
            random.randint(0, max_num_people),
            random.randint(0, max_num_people),
            max_num_people
        ) for _ in range(num_data_points)
    ]

    data = [np.array(x[0]) for x in data_and_labels]
    labels = [np.array(x[1]) for x in data_and_labels]
    return np.array(data), np.array(labels)

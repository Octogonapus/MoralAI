import random

from model import Person, Dilemma, Race, LegalSex


def generate_random_person():
    return Person(
        age=random.randint(1, 60),
        race=Race(random.randint(1, 5)),
        legal_sex=LegalSex(random.randint(1, 2)),
        jaywalking=bool(random.getrandbits(1)),
        driving_under_the_influence=bool(random.getrandbits(1))
    )


def generate_training_data(first_option_size: int, second_option_size: int, ):
    first_option = [generate_random_person() for _ in range(first_option_size)]
    second_option = [generate_random_person() for _ in range(second_option_size)]
    dilemma = Dilemma(first_option, second_option)
    return dilemma.export()

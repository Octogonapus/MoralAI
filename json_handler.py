import unittest
from dataclasses import dataclass
from enum import Enum
from typing import List

import jsonpickle as jsonpickle


class Race(Enum):
    white = 1
    black = 2
    asian = 3
    native_american = 4
    other_race = 5


class LegalSex(Enum):
    male = 1
    female = 2


@dataclass
class Person:
    def __init__(self, age: int = None, race: Race = None, legal_sex: LegalSex = None,
                 jaywalking: bool = None, driving_under_the_influence: bool = None):
        self.age = age
        self.race = race
        self.legal_sex = legal_sex
        self.jaywalking = jaywalking
        self.driving_under_the_influence = driving_under_the_influence


@dataclass
class Dilemma:
    def __init__(self, first_option: List[Person], second_option: List[Person]):
        self.firstOption = first_option
        self.secondOption = second_option


def load_dilemmas_from_file(path):
    data_file = open(path, "r")
    result = jsonpickle.decode(data_file.read())
    data_file.close()
    return result


class TestJsonParsing(unittest.TestCase):

    def test_pickling_dilemma(self):
        data = Dilemma(
            [
                Person(
                    1,
                    Race.black,
                    LegalSex.male,
                    False,
                    True
                )
            ],
            [
                Person(
                    2,
                    Race.white,
                    LegalSex.female,
                    True,
                    False
                )
            ]
        )

        self.assertEqual(jsonpickle.decode(jsonpickle.encode(data)), data)


if __name__ == '__main__':
    unittest.main()

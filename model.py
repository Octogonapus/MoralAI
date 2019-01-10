import unittest
from dataclasses import dataclass
from enum import Enum
from functools import reduce
from typing import List

import numpy as np


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

    def export_age(self):
        if self.age is None:
            return [1, 0, 0, 0, 0, 0, 0]
        else:
            if 1 <= self.age <= 10:
                return [0, 1, 0, 0, 0, 0, 0]
            elif 11 <= self.age <= 20:
                return [0, 0, 1, 0, 0, 0, 0]
            elif 21 <= self.age <= 30:
                return [0, 0, 0, 1, 0, 0, 0]
            elif 31 <= self.age <= 40:
                return [0, 0, 0, 0, 1, 0, 0]
            elif 41 <= self.age <= 50:
                return [0, 0, 0, 0, 0, 1, 0]
            elif 51 <= self.age <= 60:
                return [0, 0, 0, 0, 0, 0, 1]

    def export_race(self):
        if self.race is None:
            return [1, 0, 0, 0, 0, 0]
        else:
            if self.race is Race.white:
                return [0, 1, 0, 0, 0, 0]
            elif self.race is Race.black:
                return [0, 0, 1, 0, 0, 0]
            elif self.race is Race.asian:
                return [0, 0, 0, 1, 0, 0]
            elif self.race is Race.native_american:
                return [0, 0, 0, 0, 1, 0]
            elif self.race is Race.other_race:
                return [0, 0, 0, 0, 0, 1]

    def export_legal_sex(self):
        if self.legal_sex is None:
            return [1, 0, 0]
        else:
            if self.legal_sex is LegalSex.male:
                return [0, 1, 0]
            elif self.legal_sex is LegalSex.female:
                return [0, 0, 1]

    def export_jaywalking(self):
        if self.jaywalking is None:
            return [1, 0, 0]
        else:
            if self.jaywalking:
                return [0, 1, 0]
            else:
                return [0, 0, 1]

    def export_driving_under_the_influence(self):
        if self.driving_under_the_influence is None:
            return [1, 0, 0]
        else:
            if self.driving_under_the_influence:
                return [0, 1, 0]
            else:
                return [0, 0, 1]

    def export(self):
        return self.export_age() + self.export_race() + self.export_legal_sex() + \
               self.export_jaywalking() + self.export_driving_under_the_influence()


@dataclass
class Dilemma:
    def __init__(self, first_option: List[Person], second_option: List[Person]):
        self.firstOption = first_option
        self.secondOption = second_option

    def export(self):
        return np.array(
            reduce(lambda a, b: a + b, map(lambda a: a.export(), self.firstOption)) +
            reduce(lambda a, b: a + b, map(lambda a: a.export(), self.secondOption))
        )


class TestPersonExport(unittest.TestCase):

    def testExportWithEmptyPerson(self):
        person = Person()

        self.assertEqual(
            person.export(),
            [
                1, 0, 0, 0, 0, 0, 0,
                1, 0, 0, 0, 0, 0,
                1, 0, 0,
                1, 0, 0,
                1, 0, 0
            ]
        )

    def testExportWithFullPerson(self):
        person = Person(
            age=16,
            race=Race.native_american,
            legal_sex=LegalSex.female,
            jaywalking=True,
            driving_under_the_influence=False
        )

        self.assertEqual(
            person.export(),
            [
                0, 0, 1, 0, 0, 0, 0,
                0, 0, 0, 0, 1, 0,
                0, 0, 1,
                0, 1, 0,
                0, 0, 1
            ]
        )


if __name__ == '__main__':
    unittest.main()

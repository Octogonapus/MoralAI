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

import unittest
from enum import Enum
from functools import reduce
from typing import List


class Race(Enum):
    white = 1
    black = 2
    asian = 3
    native_american = 4
    other_race = 5


class LegalSex(Enum):
    male = 1
    female = 2


age_mapping = {
    None: [1, 0, 0, 0, 0, 0, 0],
    1: [0, 1, 0, 0, 0, 0, 0],
    11: [0, 0, 1, 0, 0, 0, 0],
    21: [0, 0, 0, 1, 0, 0, 0],
    31: [0, 0, 0, 0, 1, 0, 0],
    41: [0, 0, 0, 0, 0, 1, 0],
    51: [0, 0, 0, 0, 0, 0, 1]
}

race_mapping = {
    None: [1, 0, 0, 0, 0, 0],
    Race.white: [0, 1, 0, 0, 0, 0],
    Race.black: [0, 0, 1, 0, 0, 0],
    Race.asian: [0, 0, 0, 1, 0, 0],
    Race.native_american: [0, 0, 0, 0, 1, 0],
    Race.other_race: [0, 0, 0, 0, 0, 1]
}

legal_sex_mapping = {
    None: [1, 0, 0],
    LegalSex.male: [0, 1, 0],
    LegalSex.female: [0, 0, 1]
}

jaywalking_mapping = {
    None: [1, 0, 0],
    True: [0, 1, 0],
    False: [0, 0, 1]
}

driving_under_the_influence_mapping = {
    None: [1, 0, 0],
    True: [0, 1, 0],
    False: [0, 0, 1]
}


def create_person_from_export(export):
    """
    Creates a person from a bit vector (generated by Person.export_raw) representing that person.

    :param export: The bit vector representing the person.
    :return: The person.
    """

    def lookup_from_export(mapping, slice):
        try:
            return list(mapping.keys())[list(mapping.values()).index(slice)]
        except ValueError:
            if max(slice) == 0:
                return None
            raise ValueError("Cannot parse slice: ", slice)

    return Person(
        age=lookup_from_export(age_mapping, export[0:7]),
        race=lookup_from_export(race_mapping, export[7:13]),
        legal_sex=lookup_from_export(legal_sex_mapping, export[13:16]),
        jaywalking=lookup_from_export(jaywalking_mapping, export[16:19]),
        driving_under_the_influence=lookup_from_export(driving_under_the_influence_mapping,
                                                       export[19:22])
    )


class Person:

    def __init__(self, age: int = None, race: Race = None, legal_sex: LegalSex = None,
                 jaywalking: bool = None, driving_under_the_influence: bool = None):
        self.age = age

        if self.age is not None:
            if self.age <= 10:
                self.age = 1
            elif 11 <= self.age <= 20:
                self.age = 11
            elif 21 <= self.age <= 30:
                self.age = 21
            elif 31 <= self.age <= 40:
                self.age = 31
            elif 41 <= self.age <= 50:
                self.age = 41
            else:
                self.age = 51

        self.race = race
        self.legal_sex = legal_sex
        self.jaywalking = jaywalking
        self.driving_under_the_influence = driving_under_the_influence

    def __eq__(self, o: object) -> bool:
        if isinstance(o, Person):
            return self.age == o.age and self.race == o.race and self.legal_sex == o.legal_sex \
                   and self.jaywalking == o.jaywalking and self.driving_under_the_influence == \
                   o.driving_under_the_influence
        return False

    def export_as_list(self):
        """
        Exports this person as a list (bit vector).
        :return: The list of bits representing this person.
        """
        return age_mapping[self.age] + race_mapping[self.race] + legal_sex_mapping[self.legal_sex] + \
               jaywalking_mapping[
                   self.jaywalking] + driving_under_the_influence_mapping[
                   self.driving_under_the_influence]

    @staticmethod
    def export_empty_person_as_list():
        """
        Exports an empty person, meant to be used for padding.
        :return: The list of bits representing an empty person.
        """
        return [0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0,
                0, 0, 0,
                0, 0, 0,
                0, 0, 0]


def create_dilemma_from_export(export, num_options: int, max_size: int):
    """
    Creates a dilemma from a bit vector (generated by Dilemma.export_as_list) representing that
    dilemma.

    :param export: The bit vector representing the dilemma.
    :param num_options: The number of options in the dilemma.
    :param max_size: The maximum number of people in an option.
    :return: The dilemma.
    """

    options = []
    person_export_length = len(Person.export_empty_person_as_list())
    for i in range(num_options):
        option_slice = export[i * max_size * person_export_length:
                              (i + 1) * max_size * person_export_length]

        people_in_option = []
        for j in range(max_size):
            person_slice = option_slice[j * person_export_length:
                                        (j + 1) * person_export_length]
            people_in_option.append(create_person_from_export(person_slice))

        options.append(people_in_option)
    return Dilemma(options, max_size)


class Dilemma:
    def __init__(self, options: List[List[Person]], max_size: int):
        """
        A Dilemma is a list of options of people.

        :param options: The options.
        :param max_size: The maximum number of people in an option.
        """

        self.options = options
        self.max_size = max_size

    def __eq__(self, o: object) -> bool:
        if isinstance(o, Dilemma):
            return self.options == o.options and self.max_size == o.max_size
        return False

    def export_option(self, option: List[Person]):
        """
        Exports all people in the option and pads the rest up to self.max_size.

        :param option: A list of people comprising the option.
        :return: A list of bits representing the people, padded to self.max_size number of people.
        """

        return [
            *([] if len(option) is 0 else reduce(
                lambda a, b: a + b, map(lambda a: a.export_as_list(), option)
            )),
            *(Person.export_empty_person_as_list() * (self.max_size - len(option)))
        ]

    def export_as_list(self):
        return [i for j in self.options for i in self.export_option(j)]


class TestPersonExport(unittest.TestCase):

    def testExportWithEmptyPerson(self):
        person = Person()

        self.assertEqual(
            person.export_as_list(),
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
            person.export_as_list(),
            [
                0, 0, 1, 0, 0, 0, 0,
                0, 0, 0, 0, 1, 0,
                0, 0, 1,
                0, 1, 0,
                0, 0, 1
            ]
        )


class TestPersonFromExport(unittest.TestCase):

    def testEmptyPersonFromExport(self):
        export = [
            1, 0, 0, 0, 0, 0, 0,
            1, 0, 0, 0, 0, 0,
            1, 0, 0,
            1, 0, 0,
            1, 0, 0
        ]

        self.assertEqual(
            create_person_from_export(export),
            Person()
        )

    def testFullPersonFromExport(self):
        export = [
            0, 0, 1, 0, 0, 0, 0,
            0, 0, 0, 0, 1, 0,
            0, 0, 1,
            0, 1, 0,
            0, 0, 1
        ]

        self.assertEqual(
            create_person_from_export(export),
            Person(
                age=11,
                race=Race.native_american,
                legal_sex=LegalSex.female,
                jaywalking=True,
                driving_under_the_influence=False
            )
        )


class TestDilemmaExport(unittest.TestCase):

    def testExportWithEmptyDilemma(self):
        dilemma = Dilemma([[], [], []], 1)

        self.assertEqual(
            [
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0,
                0, 0, 0,
                0, 0, 0,
                0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0,
                0, 0, 0,
                0, 0, 0,
                0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0,
                0, 0, 0,
                0, 0, 0,
                0, 0, 0
            ],
            dilemma.export_as_list()
        )

    def testExportWithEmptyDilemmaWithMaxSize2(self):
        dilemma = Dilemma([[], [], []], 2)

        self.assertEqual(
            [
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0,
                0, 0, 0,
                0, 0, 0,
                0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0,
                0, 0, 0,
                0, 0, 0,
                0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0,
                0, 0, 0,
                0, 0, 0,
                0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0,
                0, 0, 0,
                0, 0, 0,
                0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0,
                0, 0, 0,
                0, 0, 0,
                0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0,
                0, 0, 0,
                0, 0, 0,
                0, 0, 0
            ],
            dilemma.export_as_list()
        )

    def testExportWithDilemmaOfTwoEmptyPeople(self):
        dilemma = Dilemma([[Person()], [Person()], [Person()]], 1)

        raw = dilemma.export_as_list()
        self.assertEqual(
            [
                1, 0, 0, 0, 0, 0, 0,
                1, 0, 0, 0, 0, 0,
                1, 0, 0,
                1, 0, 0,
                1, 0, 0,
                1, 0, 0, 0, 0, 0, 0,
                1, 0, 0, 0, 0, 0,
                1, 0, 0,
                1, 0, 0,
                1, 0, 0,
                1, 0, 0, 0, 0, 0, 0,
                1, 0, 0, 0, 0, 0,
                1, 0, 0,
                1, 0, 0,
                1, 0, 0
            ],
            raw
        )

    def testExportWithDilemmaOfTwoEmptyPeopleWithMaxSize2(self):
        dilemma = Dilemma([[Person()], [Person()], [Person()]], 2)

        raw = dilemma.export_as_list()
        self.assertEqual(
            [
                1, 0, 0, 0, 0, 0, 0,
                1, 0, 0, 0, 0, 0,
                1, 0, 0,
                1, 0, 0,
                1, 0, 0,
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0,
                0, 0, 0,
                0, 0, 0,
                0, 0, 0,
                1, 0, 0, 0, 0, 0, 0,
                1, 0, 0, 0, 0, 0,
                1, 0, 0,
                1, 0, 0,
                1, 0, 0,
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0,
                0, 0, 0,
                0, 0, 0,
                0, 0, 0,
                1, 0, 0, 0, 0, 0, 0,
                1, 0, 0, 0, 0, 0,
                1, 0, 0,
                1, 0, 0,
                1, 0, 0,
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0,
                0, 0, 0,
                0, 0, 0,
                0, 0, 0
            ],
            raw
        )


class TestDilemmaFromExport(unittest.TestCase):

    def testDilemmaWith3OptionsOfEmptyPeopleFromExport(self):
        export = [
            1, 0, 0, 0, 0, 0, 0,
            1, 0, 0, 0, 0, 0,
            1, 0, 0,
            1, 0, 0,
            1, 0, 0,
            1, 0, 0, 0, 0, 0, 0,
            1, 0, 0, 0, 0, 0,
            1, 0, 0,
            1, 0, 0,
            1, 0, 0,
            1, 0, 0, 0, 0, 0, 0,
            1, 0, 0, 0, 0, 0,
            1, 0, 0,
            1, 0, 0,
            1, 0, 0
        ]

        self.assertEqual(
            create_dilemma_from_export(export, 3, 1),
            Dilemma([[Person()], [Person()], [Person()]], 1)
        )


if __name__ == '__main__':
    unittest.main()

import unittest
from dataclasses import dataclass
from enum import Enum
from typing import List

import jsonpickle as jsonpickle

from model import Dilemma, Person, Race, LegalSex


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
        print("test")


if __name__ == '__main__':
    unittest.main()

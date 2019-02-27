#!/usr/bin/env python3
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
import getopt
import sys
from typing import List

from generate_data_pgmpy import DilemmaGenerator
from manage_data import write_data_to_file, TrainMetadata


def generate_data_two_options(filename: str, option_cpd: List[float], jaywalking_cpd: List[float]):
    generators = [
        DilemmaGenerator(
            option_vals=[
                option_cpd
            ],
            jaywalking_vals=[
                jaywalking_cpd,
                jaywalking_cpd[::-1]
            ]
        ),
        DilemmaGenerator(
            option_vals=[
                option_cpd[::-1]
            ],
            jaywalking_vals=[
                jaywalking_cpd[::-1],
                jaywalking_cpd
            ]
        )
    ]

    write_data_to_file(TrainMetadata(50000, 10), generators, filename)


def main(argv):
    try:
        opts, args = getopt.getopt(argv, "o:", ["ocpd=", "jcpd="])
    except getopt.GetoptError:
        print("Usage: train_ai.py -o <test_file_prefix>")
        sys.exit(2)

    test_data_filename = None
    ocpd = None
    jcpd = None
    for opt, arg in opts:
        if opt == "-o":
            test_data_filename = arg
        elif opt == "--ocpd":
            ocpd = float(arg)
        elif opt == "--jcpd":
            jcpd = float(arg)

    if test_data_filename is None:
        print("-o argument required")
        print("Usage: train_ai.py -o <test_file_prefix>")
        sys.exit(2)
    elif ocpd is None:
        print("--ocpd argument required")
        print("Usage: train_ai.py -o <test_file_prefix>")
        sys.exit(2)
    elif jcpd is None:
        print("--jcpd argument required")
        print("Usage: train_ai.py -o <test_file_prefix>")
        sys.exit(2)

    generate_data_two_options(test_data_filename, [ocpd, 1 - ocpd], [jcpd, 1 - jcpd])


if __name__ == '__main__':
    main(sys.argv[1:])

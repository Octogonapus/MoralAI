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
import json
import sys

import numpy as np

from train_ai_iteration import train_and_test


def main(argv):
    try:
        opts, args = getopt.getopt(argv, "i:")
    except getopt.GetoptError:
        print("Usage: train_ai.py -i <test_file_prefix>")
        sys.exit(2)

    test_data_filename = None
    for opt, arg in opts:
        if opt == "-i":
            test_data_filename = arg

    if test_data_filename is None:
        print("-i argument required")
        print("Usage: train_ai.py -i <test_file_prefix>")
        sys.exit(2)

    results_filename = "dense results for " + test_data_filename

    option_level_results = []
    for first_option_probability in np.linspace(0, 1, 10):
        option_cpd = [first_option_probability, 1 - first_option_probability]

        jaywalking_level_results = []
        for jaywalking_probability in np.append(np.linspace(0, 3 / 10, 5),
                                                np.linspace(7 / 10, 1, 5)):
            jaywalking_cpd = [jaywalking_probability, 1 - jaywalking_probability]
            jaywalking_level_results.append({
                "jaywalking_cpd": jaywalking_cpd,
                "results": train_and_test(test_data_filename, option_cpd, jaywalking_cpd)
            })

        option_level_results.append({
            "option_cpd": option_cpd,
            "jaywalking_level_results": jaywalking_level_results
        })

    with open(results_filename, "a") as f:
        f.write(json.dumps({
            "option_level_results": option_level_results
        }))


if __name__ == '__main__':
    main(sys.argv[1:])

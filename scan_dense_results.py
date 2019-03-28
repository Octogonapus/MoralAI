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

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # required


def parse_dense_results(data: str, item: str):
    first_option_probability = []
    jaywalking_probability = []
    z_data = []

    obj = json.loads(data.rstrip('\n'))

    for option_level_result in obj["option_level_results"]:
        first_option_probability.append(option_level_result["option_cpd"][0])

        inner_data = []
        for jaywalking_level_result in option_level_result["jaywalking_level_results"]:
            jaywalking_probability.append(jaywalking_level_result["jaywalking_cpd"][1])

            num_results = len(jaywalking_level_result["results"])
            inner_data.append(
                sum([jaywalking_level_result["results"][i][item] for i in range(num_results)]) /
                num_results
            )

        z_data.append(inner_data)

    return first_option_probability, list(dict.fromkeys(jaywalking_probability)), z_data


def generate_plots(data_tuple, title, filename=""):
    (x_data, y_data, z_data) = data_tuple
    X, Y = np.meshgrid(np.array(x_data), np.array(y_data))
    Z = np.array(z_data)

    fig1 = plt.figure(1, figsize=(12, 6))
    fig1.suptitle(title)

    ax = fig1.add_subplot(1, 2, 1)
    contours = ax.contour(X, Y, Z, 9)
    fig1.colorbar(contours, ax=ax)
    ax.set_xlabel("$P(O_1)$")
    ax.set_ylabel("$P(J \\mid O_1)$")
    ax.set_title(" ")  # So the figure title doesn't overlap

    ax = fig1.add_subplot(1, 2, 2, projection='3d')
    ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
    ax.view_init(25, -35)
    ax.set_xlabel("$P(O_1)$")
    ax.set_ylabel("$P(J \\mid O_1)$")
    ax.set_title(" ")  # So the figure title doesn't overlap

    if filename != "":
        fig1.savefig(filename)


def format_name_for_latex(filename: str) -> str:
    return filename.replace(" ", "_")


def make_figure(test_name: str, data_series: str, title: str, filename_suffix: str):
    plt.rc('text', usetex=True)

    with open("dense results for " + test_name, "r") as f:
        data = f.readlines()
        generate_plots(parse_dense_results(data[0], data_series),
                       title + test_name,
                       format_name_for_latex("figures/" + test_name + filename_suffix))


def main(argv):
    try:
        opts, args = getopt.getopt(argv, "",
                                   ["test-data=", "data-series=", "title=", "file-suffix="])
    except getopt.GetoptError:
        print("Usage: scan_dense_results.py --test-data <test data name> "
              "--data-series <data series name> --title <title> --type <filename suffix>")
        sys.exit(2)

    test_name = None
    data_series = None
    title = None
    filename_suffix = None
    for opt, arg in opts:
        if opt == "--test-data":
            test_name = arg
        elif opt == "--data-series":
            data_series = arg
        elif opt == "--title":
            title = arg
        elif opt == "--file-suffix":
            filename_suffix = arg

    if test_name is None:
        print("--test-data argument required")
        sys.exit(2)
    elif data_series is None:
        print("--data-series argument required")
        sys.exit(2)
    elif title is None:
        print("--title argument required")
        sys.exit(2)
    elif filename_suffix is None:
        print("--file-suffix argument required")
        sys.exit(2)

    make_figure(test_name, data_series, title, filename_suffix)


if __name__ == '__main__':
    main(sys.argv[1:])

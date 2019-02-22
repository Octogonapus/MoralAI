import json

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
            jaywalking_probability.append(jaywalking_level_result["jaywalking_cpd"][0])
            num_results = len(jaywalking_level_result["results"])
            inner_data.append(
                sum([jaywalking_level_result["results"][i][item] for i in range(num_results)]) /
                num_results
            )

        z_data.append(inner_data)

    return first_option_probability, jaywalking_probability, z_data


def generate_plots(data_tuple, title):
    (x_data, y_data, z_data) = data_tuple
    X, Y = np.meshgrid(np.array(x_data), np.array(y_data))
    Z = np.array(z_data)

    fig1 = plt.figure(1, figsize=plt.figaspect(0.5))
    fig1.suptitle(title)

    ax = fig1.add_subplot(1, 2, 1)
    contours = ax.contour(X, Y, Z, 9)
    fig1.colorbar(contours, ax=ax)
    ax.set_xlabel("Probability of being in the first option")
    ax.set_ylabel("Probability of jaywalking in the first option")

    ax = fig1.add_subplot(1, 2, 2, projection='3d')
    ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
    ax.view_init(25, -35)
    ax.set_xlabel("Probability of being in the first option")
    ax.set_ylabel("Probability of jaywalking in the first option")

    # contours = axarr[0].contour(X, Y, Z, 9)
    # f.clabel(contours, inline=True)
    # f.colorbar()
    # axarr[0].xlabel("Probability of being in the first option")
    # axarr[0].ylabel("Probability of jaywalking in the first option")
    #
    # ax = axarr[1].gca(projection='3d')
    # ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
    # ax.view_init(10, -30)
    # axarr[1].xlabel("Probability of being in the first option")
    # axarr[1].ylabel("Probability of jaywalking in the first option")

    fig1.show()


if __name__ == '__main__':
    test_name = "test 50-50 50-50 50-50"
    with open("dense results for " + test_name, "r") as f:
        data = f.readlines()
        generate_plots(parse_dense_results(data[0], "accuracy"),
                       "Classification accuracy against " + test_name)
        generate_plots(parse_dense_results(data[0], "loss"), "Loss against " + test_name)
        generate_plots(parse_dense_results(data[0], "prob_jaywalking_when_wrong"),
                       "Actual prob of jaywalking when wrong for " + test_name)

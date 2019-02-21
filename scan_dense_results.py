import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # required


def parse_single_test(data):
    # The first line is metadata
    line = [float(x.strip()) for x in data[0].split(",")]
    first_option_probability = line[0]
    jaywalking_probability = line[3]

    # The remaining lines are loss and accuracy
    line_loss = []
    line_accuracy = []
    actual_jay_prob = []
    for j in range(1, len(data)):
        line = [float(x.strip()) for x in data[j].split(",")]
        line_loss.append(line[0])
        line_accuracy.append(line[1])
        actual_jay_prob.append(line[2])

    return sum(line_accuracy) / len(line_accuracy), sum(line_loss) / len(line_loss), \
           sum(actual_jay_prob) / len(actual_jay_prob), first_option_probability, \
           jaywalking_probability


def parse_jaywalking_block(data):
    first_option_probability = -1
    jaywalking_probability = []
    accuracy = []
    loss = []
    actual_jaywalking_probability = []

    start_index = 0
    option_match = data[start_index]
    i = 6
    while i < len(data):
        if data[i] != option_match or i == len(data) - 1:
            acc, los, actual_jay_prob, first_prob, jay_prob = parse_single_test(data[start_index:i])

            accuracy.append(acc)
            loss.append(los)
            actual_jaywalking_probability.append(actual_jay_prob)
            first_option_probability = first_prob
            jaywalking_probability.append(jay_prob)

            start_index = i
            option_match = data[i]

        i += 6

    return first_option_probability, jaywalking_probability, accuracy, loss, \
           actual_jaywalking_probability


def parse_dense_results(data):
    first_option_probability = []
    jaywalking_probability = []
    accuracy = []
    loss = []
    actual_jaywalking_probability = []

    def parse_option_from_line(index):
        return [float(x.strip()) for x in data[index].split(",")][:2]

    start_index = 0
    option_match = parse_option_from_line(start_index)
    i = 6
    while i < len(data):
        if parse_option_from_line(i) != option_match:
            first_prob, jay_prob, acc, los, act_jay_prob = parse_jaywalking_block(
                data[start_index:i])

            first_option_probability.append(first_prob)
            jaywalking_probability = jay_prob
            accuracy.append(acc)
            loss.append(los)
            actual_jaywalking_probability.append(act_jay_prob)

            start_index = i
            option_match = parse_option_from_line(i)

        i += 6

    return first_option_probability, jaywalking_probability, accuracy, loss, \
           actual_jaywalking_probability


if __name__ == '__main__':
    test_name = "test 50-50 50-50 50-50"
    with open("dense results for " + test_name, "r") as f:
        data = f.readlines()
        y_data, x_data, z_data_acc, z_data_loss, z_data_act_jay_prob = parse_dense_results(data)


        def generate_plots(x_data, y_data, z_data, title):
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


        generate_plots(x_data, y_data, z_data_acc, "Classification accuracy against " + test_name)
        generate_plots(x_data, y_data, z_data_loss, "Loss against " + test_name)
        generate_plots(x_data, y_data, z_data_act_jay_prob, "Actual jaywalking probability for "
                       + test_name)

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
    for j in range(1, len(data)):
        line = [float(x.strip()) for x in data[j].split(",")]
        line_loss.append(line[0])
        line_accuracy.append(line[1])

    return sum(line_accuracy) / len(line_accuracy), sum(line_loss) / len(line_loss), \
           first_option_probability, jaywalking_probability


def parse_jaywalking_block(data):
    first_option_probability = -1
    jaywalking_probability = []
    accuracy = []
    loss = []

    start_index = 0
    option_match = data[start_index]
    i = 6
    while i < len(data):
        if data[i] != option_match or i == len(data) - 1:
            acc, los, first_prob, jay_prob = parse_single_test(data[start_index:i])

            accuracy.append(acc)
            loss.append(los)
            first_option_probability = first_prob
            jaywalking_probability.append(jay_prob)

            start_index = i
            option_match = data[i]

        i += 6

    return first_option_probability, jaywalking_probability, accuracy, loss


def parse_dense_results(data):
    first_option_probability = []
    jaywalking_probability = []
    accuracy = []
    loss = []

    def parse_option_from_line(index):
        return [float(x.strip()) for x in data[index].split(",")][:2]

    start_index = 0
    option_match = parse_option_from_line(start_index)
    i = 6
    while i < len(data):
        if parse_option_from_line(i) != option_match:
            first_prob, jay_prob, acc, los = parse_jaywalking_block(data[start_index:i])

            first_option_probability.append(first_prob)
            jaywalking_probability = jay_prob
            accuracy.append(acc)
            loss.append(los)

            start_index = i
            option_match = parse_option_from_line(i)

        i += 6

    return first_option_probability, jaywalking_probability, accuracy, loss


if __name__ == '__main__':
    with open("dense results for test 40-60 100-0 0-100", "r") as f:
        data = f.readlines()
        y_data, x_data, z_data_acc, z_data_loss = parse_dense_results(data)


        def generate_plots(x_data, y_data, z_data, title):
            X, Y = np.meshgrid(np.array(x_data), np.array(y_data))
            Z = np.array(z_data)

            contours = plt.contour(X, Y, Z, 9)
            plt.clabel(contours, inline=True)
            plt.colorbar()
            plt.xlabel("Probability of being in the first option")
            plt.ylabel("Probability of jaywalking in the first option")
            plt.title(title)

            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
            ax.view_init(10, -30)
            plt.xlabel("Probability of being in the first option")
            plt.ylabel("Probability of jaywalking in the first option")
            plt.title(title)

            plt.show()


        generate_plots(x_data, y_data, z_data_acc,
                       "Classification accuracy against test 40-60 100-0 0-100")
        generate_plots(x_data, y_data, z_data_loss, "Loss against test 40-60 100-0 0-100")

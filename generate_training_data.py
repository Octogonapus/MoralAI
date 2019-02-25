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

import numpy as np


def generate_modeled_dilemmas(generator, max_num_people: int, num_dilemmas: int):
    dilemmas = [generator.generate_dilemma(max_num_people) for _ in range(num_dilemmas)]
    return list(map(lambda x: (x[0].export_as_list(), x[1]), dilemmas))


def generate_training_data(generator, max_num_people: int, num_dilemmas: int):
    data_and_labels = generate_modeled_dilemmas(generator, max_num_people, num_dilemmas)
    data = [np.array(x[0]) for x in data_and_labels]
    labels = [np.array(x[1]) for x in data_and_labels]
    return np.array(data), np.array(labels)

#!/bin/bash

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

function scan_results_for_data() {
    python scan_dense_results.py --test-data "$1" --data-series "accuracy" --title "Classification accuracy against " --file-suffix "_accuracy"
    python scan_dense_results.py --test-data "$1" --data-series "loss" --title "Loss against " --file-suffix "_loss"
    python scan_dense_results.py --test-data "$1" --data-series "prob_jaywalking_when_wrong" --title "Actual prob of jaywalking when wrong for " --file-suffix "_jay_prob"
}

scan_results_for_data "test 40-60 0-100 100-0"
scan_results_for_data "test 40-60 20-80 80-20"
scan_results_for_data "test 40-60 80-20 20-80"
scan_results_for_data "test 40-60 100-0 0-100"
scan_results_for_data "test 50-50 50-50 50-50"

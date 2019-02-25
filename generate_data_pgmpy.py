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

from random import choices
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

from model import Dilemma, Person, Race, LegalSex


class StatesAndProbs:
    age_states = [10, 20, 30, 40, 50, 60]
    race_states = [Race.white, Race.black, Race.asian, Race.native_american,
                   Race.other_race]
    legal_sex_states = [LegalSex.male, LegalSex.female]
    jaywalking_states = [False, True]
    driving_under_the_influence_states = [False, True]

    def __init__(self, age_probability, race_probability, legal_sex_probability,
                 jaywalking_probability, driving_under_the_influence_probability):
        self.age_probability = age_probability
        self.race_probability = race_probability
        self.legal_sex_probability = legal_sex_probability
        self.jaywalking_probability = jaywalking_probability
        self.driving_under_the_influence_probability = driving_under_the_influence_probability


def generate_cpd(variable_cardinality, evidence_cardinality):
    cpd = []
    for i in range(variable_cardinality):
        cpd.append([1.0 / variable_cardinality] * evidence_cardinality)
    return cpd


class DilemmaGenerator:
    var_option = "option"
    var_age = "age"
    var_race = "race"
    var_legal_sex = "legal_sex"
    var_jaywalking = "jaywalking"
    var_driving_under_the_influence = "driving_under_the_influence"

    age_card = 6
    race_card = 5
    legal_sex_card = 2
    jaywalking_card = 2
    driving_under_the_influence_card = 2

    def __init__(self, option_vals, age_vals=None, race_vals=None, legal_sex_vals=None,
                 jaywalking_vals=None, driving_under_the_influence_vals=None, debug=False):
        self.option_card = len(option_vals[0])
        self.cpd_option_values = option_vals

        # Parse all the input CPDs. If the user specifies one, keep it. If not, generate it as a
        # uniform distribution.

        if age_vals is not None:
            self.cpd_age_values = age_vals
        else:
            self.cpd_age_values = generate_cpd(self.age_card, self.option_card)

        if race_vals is not None:
            self.cpd_race_values = race_vals
        else:
            self.cpd_race_values = generate_cpd(self.race_card, self.option_card)

        if legal_sex_vals is not None:
            self.cpd_legal_sex_values = legal_sex_vals
        else:
            self.cpd_legal_sex_values = generate_cpd(self.legal_sex_card, self.option_card)

        if jaywalking_vals is not None:
            self.cpd_jaywalking_values = jaywalking_vals
        else:
            self.cpd_jaywalking_values = generate_cpd(self.jaywalking_card, self.option_card)

        if driving_under_the_influence_vals is not None:
            self.cpd_driving_under_the_influence_values = driving_under_the_influence_vals
        else:
            self.cpd_driving_under_the_influence_values = generate_cpd(
                self.driving_under_the_influence_card, self.option_card)

        self.model = BayesianModel([
            (self.var_option, self.var_age),
            (self.var_option, self.var_race),
            (self.var_option, self.var_legal_sex),
            (self.var_option, self.var_jaywalking),
            (self.var_option, self.var_driving_under_the_influence)
        ])

        # First or second option
        cpd_option = TabularCPD(
            variable=self.var_option,
            variable_card=self.option_card,
            values=self.cpd_option_values
        )

        # Age bracket
        cpd_age = TabularCPD(
            variable=self.var_age,
            variable_card=self.age_card,
            values=self.cpd_age_values,
            evidence=[self.var_option],
            evidence_card=[self.option_card]
        )

        # Race enum
        cpd_race = TabularCPD(
            variable=self.var_race,
            variable_card=self.race_card,
            values=self.cpd_race_values,
            evidence=[self.var_option],
            evidence_card=[self.option_card]
        )

        # Legal sex enum
        cpd_legal_sex = TabularCPD(
            variable=self.var_legal_sex,
            variable_card=self.legal_sex_card,
            values=self.cpd_legal_sex_values,
            evidence=[self.var_option],
            evidence_card=[self.option_card]
        )

        # Jaywalking boolean, 1 = True
        cpd_jaywalking = TabularCPD(
            variable=self.var_jaywalking,
            variable_card=self.jaywalking_card,
            values=self.cpd_jaywalking_values,
            evidence=[self.var_option],
            evidence_card=[self.option_card]
        )

        # Driving under the influence boolean, 1 = True
        cpd_driving_under_the_influence = TabularCPD(
            variable=self.var_driving_under_the_influence,
            variable_card=self.driving_under_the_influence_card,
            values=self.cpd_driving_under_the_influence_values,
            evidence=[self.var_option],
            evidence_card=[self.option_card]
        )

        # Associating the CPDs with the network
        self.model.add_cpds(
            cpd_option,
            cpd_age,
            cpd_race,
            cpd_legal_sex,
            cpd_jaywalking,
            cpd_driving_under_the_influence
        )

        self.model.check_model()

        if debug:
            print("Option CPD:")
            print(self.model.get_cpds(self.var_option))

            print("Age CPD:")
            print(self.model.get_cpds(self.var_age))

            print("Race CPD:")
            print(self.model.get_cpds(self.var_race))

            print("Legal sex CPD:")
            print(self.model.get_cpds(self.var_legal_sex))

            print("Jaywalking CPD:")
            print(self.model.get_cpds(self.var_jaywalking))

            print("Driving under the influence CPD:")
            print(self.model.get_cpds(self.var_driving_under_the_influence))

        # noinspection PyTypeChecker
        self.infer = VariableElimination(self.model)

        option_query = self.infer.query(["option"])["option"].values
        self.option_probability = [option_query[i] for i in range(self.option_card)]
        self.option_states = [i for i in range(self.option_card)]

        def generate_option_probabilities(option: int):
            def infer_values_given_option(variable: str):
                return self.infer.query(
                    [variable], evidence={self.var_option: option}
                )[variable].values

            def map_infer_to_list(infer):
                return [x.item() for x in infer]

            return StatesAndProbs(
                age_probability=map_infer_to_list(infer_values_given_option(self.var_age)),
                race_probability=map_infer_to_list(infer_values_given_option(self.var_race)),
                legal_sex_probability=map_infer_to_list(infer_values_given_option(
                    self.var_legal_sex)),
                jaywalking_probability=map_infer_to_list(
                    infer_values_given_option(self.var_jaywalking)),
                driving_under_the_influence_probability=map_infer_to_list(
                    infer_values_given_option(self.var_driving_under_the_influence)
                )
            )

        # Generate all possible probabilities once for this class because inferring is very
        # expensive
        self.all_probabilities = [generate_option_probabilities(state) for state in
                                  self.option_states]

    def generate_person_list(self, option_size: int, option: int):
        age_choices = choices(self.all_probabilities[option].age_states,
                              self.all_probabilities[option].age_probability,
                              k=option_size)

        race_choices = choices(self.all_probabilities[option].race_states,
                               self.all_probabilities[option].race_probability,
                               k=option_size)

        legal_sex_choices = choices(self.all_probabilities[option].legal_sex_states,
                                    self.all_probabilities[option].legal_sex_probability,
                                    k=option_size)

        jaywalking_choices = choices(self.all_probabilities[option].jaywalking_states,
                                     self.all_probabilities[option].jaywalking_probability,
                                     k=option_size)

        driving_under_the_influence_choices = choices(
            self.all_probabilities[option].driving_under_the_influence_states,
            self.all_probabilities[option].driving_under_the_influence_probability,
            k=option_size)

        return [Person(
            age=age_choices[i],
            race=race_choices[i],
            legal_sex=legal_sex_choices[i],
            jaywalking=jaywalking_choices[i],
            driving_under_the_influence=driving_under_the_influence_choices[i]
        ) for i in range(option_size)]

    def generate_dilemma(self, max_num_people: int):
        option_choices = choices(self.option_states, self.option_probability, k=max_num_people)

        # Count the number of people we want in each option and build a list where the number of
        # people in an option corresponds to the index of that option.
        option_sizes = [0] * self.option_card
        for choice in option_choices:
            option_sizes[choice] = option_sizes[choice] + 1

        # Generate a list of people for each option.
        option_people = [self.generate_person_list(option_sizes[i], i) for i in
                         range(len(option_sizes))]

        dilemma = Dilemma(option_people, max_num_people)

        # Label the leftmost maximum size option as correct.
        label = [0] * len(option_sizes)
        label[option_sizes.index(max(option_sizes))] = 1

        return dilemma, label

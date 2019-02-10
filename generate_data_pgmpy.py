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


class DilemmaGenerator:
    var_option = "option"
    var_age = "age"
    var_race = "race"
    var_legal_sex = "legal_sex"
    var_jaywalking = "jaywalking"
    var_driving_under_the_influence = "driving_under_the_influence"

    cpd_option_values = [[1 / 3, 1 / 3, 1 / 3]]
    cpd_age_values = [
        [1 / 6, 1 / 6, 1 / 6],
        [1 / 6, 1 / 6, 1 / 6],
        [1 / 6, 1 / 6, 1 / 6],
        [1 / 6, 1 / 6, 1 / 6],
        [1 / 6, 1 / 6, 1 / 6],
        [1 / 6, 1 / 6, 1 / 6]
    ]
    cpd_race_values = [
        [1 / 5, 1 / 5, 1 / 5],
        [1 / 5, 1 / 5, 1 / 5],
        [1 / 5, 1 / 5, 1 / 5],
        [1 / 5, 1 / 5, 1 / 5],
        [1 / 5, 1 / 5, 1 / 5]
    ]
    cpd_legal_sex_values = [
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5]
    ]
    cpd_jaywalking_values = [
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5]
    ]
    cpd_driving_under_the_influence_values = [
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5]
    ]

    def __init__(self, option_vals=None, age_vals=None, race_vals=None, legal_sex_vals=None,
                 jaywalking_vals=None, driving_under_the_influence_vals=None, debug=False):
        if option_vals is not None:
            self.cpd_option_values = option_vals
        if age_vals is not None:
            self.cpd_age_values = age_vals
        if race_vals is not None:
            self.cpd_race_values = race_vals
        if legal_sex_vals is not None:
            self.cpd_legal_sex_values = legal_sex_vals
        if jaywalking_vals is not None:
            self.cpd_jaywalking_values = jaywalking_vals
        if driving_under_the_influence_vals is not None:
            self.cpd_driving_under_the_influence_values = driving_under_the_influence_vals

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
            variable_card=3,
            values=self.cpd_option_values
        )

        # Age bracket
        cpd_age = TabularCPD(
            variable=self.var_age,
            variable_card=6,
            values=self.cpd_age_values,
            evidence=[self.var_option],
            evidence_card=[3]
        )

        # Race enum
        cpd_race = TabularCPD(
            variable=self.var_race,
            variable_card=5,
            values=self.cpd_race_values,
            evidence=[self.var_option],
            evidence_card=[3]
        )

        # Legal sex enum
        cpd_legal_sex = TabularCPD(
            variable=self.var_legal_sex,
            variable_card=2,
            values=self.cpd_legal_sex_values,
            evidence=[self.var_option],
            evidence_card=[3]
        )

        # Jaywalking boolean, 1 = True
        cpd_jaywalking = TabularCPD(
            variable=self.var_jaywalking,
            variable_card=2,
            values=self.cpd_jaywalking_values,
            evidence=[self.var_option],
            evidence_card=[3]
        )

        # Driving under the influence boolean, 1 = True
        cpd_driving_under_the_influence = TabularCPD(
            variable=self.var_driving_under_the_influence,
            variable_card=2,
            values=self.cpd_driving_under_the_influence_values,
            evidence=[self.var_option],
            evidence_card=[3]
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
        self.option_probability = [option_query[0].item(), option_query[1].item(),
                                   option_query[2].item()]
        self.option_states = [0, 1, 2]

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
        self.probs = [generate_option_probabilities(state) for state in self.option_states]

    def generate_person_list(self, option_size: int, option: int):
        age_choices = choices(self.probs[option].age_states, self.probs[option].age_probability,
                              k=option_size)

        race_choices = choices(self.probs[option].race_states, self.probs[option].race_probability,
                               k=option_size)

        legal_sex_choices = choices(self.probs[option].legal_sex_states,
                                    self.probs[option].legal_sex_probability, k=option_size)

        jaywalking_choices = choices(self.probs[option].jaywalking_states,
                                     self.probs[option].jaywalking_probability, k=option_size)

        driving_under_the_influence_choices = choices(
            self.probs[option].driving_under_the_influence_states,
            self.probs[option].driving_under_the_influence_probability,
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

        first_option_size = 0
        second_option_size = 0
        third_option_size = 0
        for choice in option_choices:
            if choice == 0:
                first_option_size += 1
            elif choice == 1:
                second_option_size += 1
            elif choice == 2:
                third_option_size += 1

        first_option = self.generate_person_list(first_option_size, 0)
        second_option = self.generate_person_list(second_option_size, 1)
        third_option = self.generate_person_list(third_option_size, 2)

        dilemma = Dilemma(first_option, second_option, third_option, max_num_people)

        if second_option_size > first_option_size and second_option_size > third_option_size:
            label = [0, 1, 0]
        elif third_option_size > first_option_size and third_option_size > second_option_size:
            label = [0, 0, 1]
        else:
            label = [1, 0, 0]

        return dilemma, label

from random import choices
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

from model import Dilemma, Person, Race, LegalSex


class DilemmaGenerator:
    var_option = "option"
    var_age = "age"
    var_race = "race"
    var_legal_sex = "legal_sex"
    var_jaywalking = "jaywalking"
    var_driving_under_the_influence = "driving_under_the_influence"

    cpd_option_values = [[0.5, 0.5]]
    cpd_age_values = [
        [1 / 6, 1 / 6],
        [1 / 6, 1 / 6],
        [1 / 6, 1 / 6],
        [1 / 6, 1 / 6],
        [1 / 6, 1 / 6],
        [1 / 6, 1 / 6]
    ]
    cpd_race_values = [
        [1 / 5, 1 / 5],
        [1 / 5, 1 / 5],
        [1 / 5, 1 / 5],
        [1 / 5, 1 / 5],
        [1 / 5, 1 / 5]
    ]
    cpd_legal_sex_values = [
        [0.5, 0.5],
        [0.5, 0.5]
    ]
    cpd_jaywalking_values = [
        [0.5, 0.5],
        [0.5, 0.5]
    ]
    cpd_driving_under_the_influence_values = [
        [0.5, 0.5],
        [0.5, 0.5]
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
            variable_card=2,
            values=self.cpd_option_values
        )

        # Age bracket
        cpd_age = TabularCPD(
            variable=self.var_age,
            variable_card=6,
            values=self.cpd_age_values,
            evidence=[self.var_option],
            evidence_card=[2]
        )

        # Race enum
        cpd_race = TabularCPD(
            variable=self.var_race,
            variable_card=5,
            values=self.cpd_race_values,
            evidence=[self.var_option],
            evidence_card=[2]
        )

        # Legal sex enum
        cpd_legal_sex = TabularCPD(
            variable=self.var_legal_sex,
            variable_card=2,
            values=self.cpd_legal_sex_values,
            evidence=[self.var_option],
            evidence_card=[2]
        )

        # Jaywalking boolean, 1 = True
        cpd_jaywalking = TabularCPD(
            variable=self.var_jaywalking,
            variable_card=2,
            values=self.cpd_jaywalking_values,
            evidence=[self.var_option],
            evidence_card=[2]
        )

        # Driving under the influence boolean, 1 = True
        cpd_driving_under_the_influence = TabularCPD(
            variable=self.var_driving_under_the_influence,
            variable_card=2,
            values=self.cpd_driving_under_the_influence_values,
            evidence=[self.var_option],
            evidence_card=[2]
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

    def generate_person_list(self, option_size: int, option: int):
        def infer_values_given_option(variable: str):
            return self.infer.query(
                [variable], evidence={self.var_option: option}
            )[variable].values

        def map_infer_to_list(infer):
            return [x.item() for x in infer]

        age_states = [10, 20, 30, 40, 50, 60]
        age_probability = map_infer_to_list(infer_values_given_option(self.var_age))

        race_states = [Race.white, Race.black, Race.asian, Race.native_american, Race.other_race]
        race_probability = map_infer_to_list(infer_values_given_option(self.var_race))

        legal_sex_states = [LegalSex.male, LegalSex.female]
        legal_sex_probability = map_infer_to_list(infer_values_given_option(self.var_legal_sex))

        jaywalking_states = [False, True]
        jaywalking_probability = map_infer_to_list(infer_values_given_option(self.var_jaywalking))

        driving_under_the_influence_states = [False, True]
        driving_under_the_influence_probability = map_infer_to_list(
            infer_values_given_option(self.var_driving_under_the_influence)
        )

        age_choices = choices(age_states, age_probability, k=option_size)
        race_choices = choices(race_states, race_probability, k=option_size)
        legal_sex_choices = choices(legal_sex_states, legal_sex_probability, k=option_size)
        jaywalking_choices = choices(jaywalking_states, jaywalking_probability, k=option_size)
        driving_under_the_influence_choices = choices(driving_under_the_influence_states,
                                                      driving_under_the_influence_probability,
                                                      k=option_size)

        return [Person(
            age=age_choices[i],
            race=race_choices[i],
            legal_sex=legal_sex_choices[i],
            jaywalking=jaywalking_choices[i],
            driving_under_the_influence=driving_under_the_influence_choices[i]
        ) for i in range(option_size)]

    def generate_dilemma(self, max_num_people: int):
        option_probability = self.infer.query(["option"])["option"].values
        first_option_size = round(max_num_people * option_probability[0].item())
        second_option_size = round(max_num_people * option_probability[1].item())

        first_option = self.generate_person_list(first_option_size, 0)
        second_option = self.generate_person_list(second_option_size, 1)

        dilemma = Dilemma(first_option, second_option, max_num_people)

        label = [1, 0] if first_option_size >= second_option_size else [0, 1]

        return dilemma, label


if __name__ == '__main__':
    max_num_people = 10
    dilemma_count = 100
    generator = DilemmaGenerator(
        jaywalking_vals=[
            [1 / 3, 2 / 3],
            [2 / 3, 1 / 3]
        ],
        debug=True
    )
    dilemmas = [generator.generate_dilemma(max_num_people) for _ in range(dilemma_count)]

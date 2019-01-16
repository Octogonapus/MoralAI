from random import choices
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

from model import Dilemma, Person, Race, LegalSex

if __name__ == '__main__':
    var_option = "option"
    var_age = "age"
    var_race = "race"
    var_legal_sex = "legal_sex"
    var_jaywalking = "jaywalking"
    var_driving_under_the_influence = "driving_under_the_influence"

    # Define edges
    model = BayesianModel([
        (var_option, var_age),
        (var_option, var_race),
        (var_option, var_legal_sex),
        (var_option, var_jaywalking),
        (var_option, var_driving_under_the_influence)
    ])

    # First or second option
    cpd_option = TabularCPD(
        variable=var_option,
        variable_card=2,
        values=[[0.5, 0.5]]
    )

    # Age bracket
    cpd_age = TabularCPD(
        variable=var_age,
        variable_card=6,
        values=[
            [1 / 6, 1 / 6],
            [1 / 6, 1 / 6],
            [1 / 6, 1 / 6],
            [1 / 6, 1 / 6],
            [1 / 6, 1 / 6],
            [1 / 6, 1 / 6]
        ],
        evidence=[var_option],
        evidence_card=[2]
    )

    # Race enum
    cpd_race = TabularCPD(
        variable=var_race,
        variable_card=5,
        values=[
            [1 / 5, 1 / 5],
            [1 / 5, 1 / 5],
            [1 / 5, 1 / 5],
            [1 / 5, 1 / 5],
            [1 / 5, 1 / 5]
        ],
        evidence=[var_option],
        evidence_card=[2]
    )

    # Legal sex enum
    cpd_legal_sex = TabularCPD(
        variable=var_legal_sex,
        variable_card=2,
        values=[
            [0.5, 0.5],
            [0.5, 0.5]
        ],
        evidence=[var_option],
        evidence_card=[2]
    )

    # Jaywalking boolean, 1 = True
    cpd_jaywalking = TabularCPD(
        variable=var_jaywalking,
        variable_card=2,
        values=[
            [0.5, 0.5],
            [0.5, 0.5]
        ],
        evidence=[var_option],
        evidence_card=[2]
    )

    # Driving under the influence boolean, 1 = True
    cpd_driving_under_the_influence = TabularCPD(
        variable=var_driving_under_the_influence,
        variable_card=2,
        values=[
            [0.5, 0.5],
            [0.5, 0.5]
        ],
        evidence=[var_option],
        evidence_card=[2]
    )

    # Associating the CPDs with the network
    model.add_cpds(
        cpd_option,
        cpd_age,
        cpd_race,
        cpd_legal_sex,
        cpd_jaywalking,
        cpd_driving_under_the_influence
    )

    # Check the network structure and CPDs and verify that the CPDs are correctly defined and sum
    # to 1
    model.check_model()

    print("Option CPD:")
    print(model.get_cpds(var_option))

    print("Age CPD:")
    print(model.get_cpds(var_age))

    print("Race CPD:")
    print(model.get_cpds(var_race))

    print("Legal sex CPD:")
    print(model.get_cpds(var_legal_sex))

    print("Jaywalking CPD:")
    print(model.get_cpds(var_jaywalking))

    print("Driving under the influence CPD:")
    print(model.get_cpds(var_driving_under_the_influence))

    # noinspection PyTypeChecker
    infer = VariableElimination(model)


    def generate_person_list(option_size: int, option: int):
        def infer_values_given_option(variable: str):
            return infer.query([variable], evidence={var_option: option})[variable].values

        def map_infer_to_list(infer):
            return [x.item() for x in infer]

        age_states = [10, 20, 30, 40, 50, 60]
        age_probability = map_infer_to_list(infer_values_given_option(var_age))

        race_states = [Race.white, Race.black, Race.asian, Race.native_american, Race.other_race]
        race_probability = map_infer_to_list(infer_values_given_option(var_race))

        legal_sex_states = [LegalSex.male, LegalSex.female]
        legal_sex_probability = map_infer_to_list(infer_values_given_option(var_legal_sex))

        jaywalking_states = [False, True]
        jaywalking_probability = map_infer_to_list(infer_values_given_option(var_jaywalking))

        driving_under_the_influence_states = [False, True]
        driving_under_the_influence_probability = map_infer_to_list(
            infer_values_given_option(var_driving_under_the_influence)
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


    def generate_dilemma(max_num_people: int):
        option_probability = infer.query(["option"])["option"].values
        first_option_size = round(max_num_people * option_probability[0].item())
        second_option_size = round(max_num_people * option_probability[1].item())

        first_option = generate_person_list(first_option_size, 0)
        second_option = generate_person_list(second_option_size, 1)

        dilemma = Dilemma(first_option, second_option, max_num_people)

        label = [1, 0] if first_option_size >= second_option_size else [0, 1]

        return dilemma, label


    max_num_people = 10
    dilemma_count = 100
    dilemmas = [generate_dilemma(max_num_people) for _ in range(dilemma_count)]

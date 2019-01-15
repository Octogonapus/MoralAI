from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

if __name__ == '__main__':
    # Defining the model structure. We can define the network by just passing a list of edges.
    model = BayesianModel([("option", "jaywalking")])

    # First or second option
    cpd_option = TabularCPD(variable="option", variable_card=2, values=[[0.3, 0.7]])
    cpd_jaywalking = TabularCPD(variable="jaywalking", variable_card=2, values=[[0.6, 0.4],
                                                                                [0.4, 0.6]],
                                evidence=["option"], evidence_card=[2])

    # Associating the CPDs with the network
    model.add_cpds(cpd_option, cpd_jaywalking)

    # check_model checks for the network structure and CPDs and verifies that the CPDs are correctly
    # defined and sum to 1.
    model.check_model()

    # We can now call some methods on the BayesianModel object.
    print("Option CPD:")
    print(model.get_cpds("option"))

    print("Jaywalking CPD:")
    print(model.get_cpds("jaywalking"))

    infer = VariableElimination(model)

    # We can infer a variable.
    print("Jaywalking inference:")
    print(infer.query(["jaywalking"])["jaywalking"])

    # We can compute a conditional distribution.
    print("Jaywalking inference given option=1:")
    print(infer.query(["jaywalking"], evidence={"option": 1})["jaywalking"])

    # We can also get the most probable state of a variable.
    print("Most probable states for:")
    print("Option:", infer.map_query(["option"]))
    print("Jaywalking:", infer.map_query(["jaywalking"]))
    print("Jaywalking given option=0:", infer.map_query(["jaywalking"], evidence={"option": 0}))
    print("Jaywalking given option=1:", infer.map_query(["jaywalking"], evidence={"option": 1}))

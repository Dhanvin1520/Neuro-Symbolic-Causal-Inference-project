from c2c.reader import rename_variables

PREMISE = ("Suppose there is a closed system of 3 variables, Z, Y and X. All the statistical relations among these "
           "3 variables are as follows: Z correlates with Y. Z correlates with X. However, Y and X are independent "
           "given Z.")


def test_rename_follows_premise_order():
    premise, hypothesis = rename_variables(PREMISE, "X directly causes Z.")
    assert "variables, A, B and C." in premise
    assert "B and C are independent given A." in premise
    assert hypothesis == "C directly causes A."


def test_rename_keeps_a_b_c():
    premise = PREMISE.replace("Z", "A").replace("Y", "B").replace("X", "C")
    assert rename_variables(premise, "A directly causes B.") == (premise, "A directly causes B.")

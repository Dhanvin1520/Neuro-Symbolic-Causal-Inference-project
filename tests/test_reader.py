import pandas as pd

from c2c.parsing import parse_hypothesis
from c2c.reader import ORIGINAL_TEMPLATES, WORDINGS, rename_variables, wording_examples

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


def test_wordings_read_by_parser():
    # every wording has the cause first, so the parser's x and y match the {x} and {y} we filled in
    examples = parse_hypothesis(wording_examples())
    assert (examples["x"] + examples["y"]).str.len().eq(2).all()
    assert examples["x"].ne(examples["y"]).all()


def test_validation_wordings_are_held_out():
    examples = wording_examples()
    assert set(examples.loc[examples["part"] == "validation", "hypothesis"]).isdisjoint(
        examples.loc[examples["part"] == "train", "hypothesis"])
    assert examples.groupby("relation")["part"].apply(lambda p: (p == "validation").sum()).eq(30).all()


def test_original_only():
    examples = wording_examples(original_only=True)
    assert len(examples) == len(ORIGINAL_TEMPLATES) * 30
    assert set(examples["part"]) == {"train"}


def test_no_test_words():
    words = " ".join(w for ws in WORDINGS.values() for w in ws).lower()
    for word in ["affect", "influence", "mediator", "together"]:
        assert word not in words

"""Phase 2 text helpers: rename variables to A-F, and the wordings used to train the claim readers
(notebooks 06 and 07)."""
import itertools
import re

import pandas as pd

from c2c.parsing import parse_variables

# dataset templates -> the four claim types our graph rules know
TEMPLATE_TO_RELATION = {
    "parent": "direct_cause", "child": "direct_cause",
    "non-parent ancestor": "indirect_cause", "non-child descendant": "indirect_cause",
    "has_collider": "collider", "has_confounder": "confounder",
}

# ways to say each claim. The first wording(s) are the original Corr2Cause templates, the others are written by us.
# None of them use the words of the authors' paraphrase test ("affects", "influences", "mediator", "together").
# The last wording of each claim type is kept out of training and used for validation.
# No wording starts with the article "A", because a lone capital letter is read as a variable name.
WORDINGS = {
    "direct_cause": [
        "{x} directly causes {y}.",
        "{x} has a direct effect on {y}.",
        "{x} is a direct cause of {y}.",
        "{x} directly leads to {y}.",
        "{x} causes {y} with nothing in between.",
        "{x} directly changes {y}.",
    ],
    "indirect_cause": [
        "{x} causes something else which causes {y}.",
        "{x} is a cause for {y}, but not a direct one.",
        "{x} causes {y} through another variable.",
        "{x} has an indirect effect on {y}.",
        "{x} leads to {y} by way of a third variable.",
        "{x} is an indirect cause of {y}.",
    ],
    "collider": [
        "There exists at least one collider (i.e., common effect) of {x} and {y}.",
        "{x} and {y} have a common effect.",
        "{x} and {y} both cause a third variable.",
        "There is a variable that is caused by both {x} and {y}.",
        "{x} and {y} share a common effect.",
        "One variable is an effect of {x} and also of {y}.",
    ],
    "confounder": [
        "There exists at least one confounder (i.e., common cause) of {x} and {y}.",
        "{x} and {y} have a common cause.",
        "{x} and {y} are caused by the same variable.",
        "There is a variable that is a cause of {x} and of {y}.",
        "{x} and {y} share a common cause.",
        "One variable is a cause of {x} and also of {y}.",
    ],
}

# the original Corr2Cause templates (the only wordings in the dataset)
ORIGINAL_TEMPLATES = [
    "{x} directly causes {y}.",
    "{x} causes something else which causes {y}.",
    "{x} is a cause for {y}, but not a direct one.",
    "There exists at least one collider (i.e., common effect) of {x} and {y}.",
    "There exists at least one confounder (i.e., common cause) of {x} and {y}.",
]


def rename_variables(premise, hypothesis):
    """Rename the variables to A, B, C, ... by the order the premise lists them (our entity schema).
    'Z, Y and X' becomes 'A, B and C' in both the premise and the hypothesis."""
    new_name = {old: "ABCDEF"[i] for i, old in enumerate(parse_variables(premise))}

    def swap(text):
        return re.sub(r"\b[A-Z]\b", lambda m: new_name.get(m.group(0), m.group(0)), text)

    return swap(premise), swap(hypothesis)


def wording_examples(original_only=False):
    """Every wording filled in with every ordered pair of A-F, as a table: hypothesis, relation, part.
    part is 'train' or 'validation' (the last wording of each claim type).
    original_only=True keeps only the Corr2Cause templates (all of them are 'train')."""
    rows = []
    for relation, wordings in WORDINGS.items():
        for i, wording in enumerate(wordings):
            if original_only and wording not in ORIGINAL_TEMPLATES:
                continue
            part = "validation" if i == len(wordings) - 1 else "train"
            for x, y in itertools.permutations("ABCDEF", 2):
                rows.append({"hypothesis": wording.format(x=x, y=y), "relation": relation, "part": part})
    return pd.DataFrame(rows)

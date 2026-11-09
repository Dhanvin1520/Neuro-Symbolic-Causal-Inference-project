"""Phase 2 text helpers: rename variables to A-F and map dataset templates to claim types (notebook 06)."""
import re

from c2c.parsing import parse_variables

# dataset templates -> the four claim types our graph rules know
TEMPLATE_TO_RELATION = {
    "parent": "direct_cause", "child": "direct_cause",
    "non-parent ancestor": "indirect_cause", "non-child descendant": "indirect_cause",
    "has_collider": "collider", "has_confounder": "confounder",
}


def rename_variables(premise, hypothesis):
    """Rename the variables to A, B, C, ... by the order the premise lists them (our entity schema).
    'Z, Y and X' becomes 'A, B and C' in both the premise and the hypothesis."""
    new_name = {old: "ABCDEF"[i] for i, old in enumerate(parse_variables(premise))}

    def swap(text):
        return re.sub(r"\b[A-Z]\b", lambda m: new_name.get(m.group(0), m.group(0)), text)

    return swap(premise), swap(hypothesis)

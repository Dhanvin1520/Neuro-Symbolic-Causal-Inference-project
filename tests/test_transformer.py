from transformers import AutoTokenizer

from c2c.transformer import MODEL_NAME, encode


def test_long_premise_is_cut_but_hypothesis_kept():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    premise = "A correlates with B. " * 200
    inputs = encode(tokenizer, [premise], ["C directly causes D."])
    assert inputs["input_ids"].shape[1] == 512
    text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
    assert text.endswith("c directly causes d.")


def test_short_inputs_are_padded_to_the_longest():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    inputs = encode(tokenizer, ["A directly causes B.", "A and B share a common cause."])
    assert inputs["input_ids"].shape[0] == 2
    assert inputs["attention_mask"][0].sum() < inputs["attention_mask"][1].sum()

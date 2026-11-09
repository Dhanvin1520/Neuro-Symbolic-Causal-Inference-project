"""Fine-tune DistilBERT as a text classifier with a plain PyTorch loop (notebooks 07 and 08)."""
import logging as python_logging

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.utils import logging

logging.set_verbosity_error()   # hide the "some weights are newly initialized" message, which is expected
logging.disable_progress_bar()
# hide a harmless torch note on macOS ("Redirects are currently not supported")
python_logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(python_logging.ERROR)

MODEL_NAME = "distilbert-base-uncased"
# Apple GPU (mps) on a Mac, an NVIDIA GPU (cuda) if there is one, otherwise the CPU
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


def load_model(num_labels, seed=42):
    """A pretrained DistilBERT with a new, untrained classification layer on top."""
    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels).to(DEVICE)
    return tokenizer, model


def encode(tokenizer, texts, text_pairs=None, fixed_length=False):
    """Turn sentences into token ids. With a pair (premise, hypothesis) only the premise is cut if the two
    together are longer than 512 tokens, the most DistilBERT can read. fixed_length=True pads every batch to
    512 tokens: the same shape every step keeps memory flat during long training runs."""
    return tokenizer(list(texts), None if text_pairs is None else list(text_pairs),
                     padding="max_length" if fixed_length else True,
                     truncation=True if text_pairs is None else "only_first", max_length=512,
                     return_tensors="pt").to(DEVICE)


def fine_tune(model, tokenizer, texts, labels, text_pairs=None, epochs=3, lr=3e-5, batch_size=16,
              balanced=False, seed=42, fixed_length=False):
    """Train every layer for a few epochs with AdamW. balanced=True weights the loss so the rare class counts
    as much as the common one (like class_weight="balanced" in scikit-learn). fixed_length: see encode."""
    labels = np.asarray(labels)
    texts = np.asarray(texts, dtype=object)
    text_pairs = None if text_pairs is None else np.asarray(text_pairs, dtype=object)
    weights = None
    if balanced:
        counts = np.bincount(labels)
        weights = torch.tensor(len(labels) / (len(counts) * counts), dtype=torch.float32, device=DEVICE)
    loss_function = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)

    model.train()
    for epoch in range(epochs):
        order = rng.permutation(len(labels))
        total = 0.0
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            inputs = encode(tokenizer, texts[batch], None if text_pairs is None else text_pairs[batch], fixed_length)
            loss = loss_function(model(**inputs).logits, torch.tensor(labels[batch], device=DEVICE))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total += loss.item() * len(batch)
            if DEVICE == "mps" and start % (50 * batch_size) == 0:
                torch.mps.empty_cache()   # free unused Apple GPU memory, so long runs fit on an 8 GB laptop
        print(f"epoch {epoch + 1}: mean loss {total / len(labels):.4f}")
    model.eval()


def predict_proba(model, tokenizer, texts, text_pairs=None, batch_size=32):
    """Probability of every class for every sentence (or sentence pair)."""
    texts = list(texts)
    text_pairs = None if text_pairs is None else list(text_pairs)
    model.eval()
    probabilities = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            inputs = encode(tokenizer, texts[start:start + batch_size],
                            None if text_pairs is None else text_pairs[start:start + batch_size])
            probabilities.append(model(**inputs).logits.softmax(-1).cpu().numpy())
    return np.vstack(probabilities)

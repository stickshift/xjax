"""
Provides tests of high level hugging face models. Primary goal is let us run these models through the debugger to see
what's happening under the hood.
"""

import numpy as np
import torch
from transformers import pipeline


def test_text_classification(device: torch.device):

    #
    # Givens
    #

    # I created off-the-shelf text classification pipeline
    model = pipeline("text-classification", device=device)

    #
    # Whens
    #

    # I classify a sentence as positive or negative
    outputs = model("I love ice cream")

    #
    # Thens
    #

    # The sentence should be classified as positive
    assert outputs[0]["label"] == "POSITIVE"


def test_zero_shot_classification(device: torch.device):

    #
    # Givens
    #

    # I created off-the-shelf zero-shot classification pipeline
    classifier = pipeline("zero-shot-classification", device=device)

    #
    # Whens
    #

    # I classify a sentence as {"education", "politics", "business"}
    outputs = classifier(
        "This is a course about the Transformers library",
        candidate_labels=["education", "politics", "business"],
    )

    #
    # Thens
    #

    # The sentence should be classified as education
    assert outputs["labels"][np.argmax(outputs["scores"])] == "education"

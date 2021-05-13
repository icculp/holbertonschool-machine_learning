#!/usr/bin/env python3
"""
    NLP Evaluation Metrics
"""
import numpy as np
from collections import Counter
import math
from nltk.translate.bleu_score import corpus_bleu
from fractions import Fraction


def uni_bleu(references, sentence):
    """ calculates the unigram BLEU score for a sentence:
        references is a list of reference translations
        each reference translation is a list of the words in the translation
        sentence is a list containing the model proposed sentence
        Returns: the unigram BLEU score
    """
    return corpus_bleu([references], [sentence], weights=(1, 0, 0, 0))

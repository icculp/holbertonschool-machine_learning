#!/usr/bin/env python3
import numpy as np
uni_bleu = __import__('0-uni_bleu').uni_bleu
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import sentence_bleu

references = [["the", "cat", "is", "on", "the", "mat"], ["there", "is", "a", "cat", "on", "the", "mat"]]
sentence = ["there", "is", "a", "cat", "here"]

print(uni_bleu(references, sentence))
print(np.round(corpus_bleu([references], [sentence], weights=(1, 0, 0, 0)), 10))

references = [["hello", "there", "my", "friend"]]
sentence = ["hello", "hello", "there", "there", "friend"]
print(np.round(uni_bleu(references, sentence), 10))
print(np.round(corpus_bleu([references], [sentence], weights=(1, 0, 0, 0)), 10))
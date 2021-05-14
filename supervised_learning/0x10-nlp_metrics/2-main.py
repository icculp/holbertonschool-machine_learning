#!/usr/bin/env python3
import numpy as np
cumulative_bleu = __import__('2-cumulative_bleu').cumulative_bleu
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import sentence_bleu

references = [["the", "cat", "is", "on", "the", "mat"], ["there", "is", "a", "cat", "on", "the", "mat"]]
sentence = ["there", "is", "a", "cat", "here"]

print(cumulative_bleu(references, sentence, 4))
print(sentence_bleu(references, sentence))


references = [["hello", "there", "my", "friend"]]
sentence = ["hello", "hello", "there", "there", "friend"]
print(np.round(cumulative_bleu(references, sentence, 2), 10))
print(np.round(sentence_bleu(references, sentence, weights=(.5, .5, 0, 0)), 10))
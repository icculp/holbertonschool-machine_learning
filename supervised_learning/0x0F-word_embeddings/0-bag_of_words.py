#!/usr/bin/env python3
"""
    NLP - Word Embeddings
"""


def bag_of_words(sentences, vocab=None):
    """ Creates a bag of words embedding matrix
        sentences is a list of sentences to analyze
        vocab is a list of the vocabulary words to use for the analysis
            If None, all words within sentences should be used
        Returns: embeddings, features
            embeddings ndarray (s, f) containing the embeddings
                s is the number of sentences in sentences
                f is the number of features analyzed
            features list of the features used for embeddings
    """
